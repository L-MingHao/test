import torch.utils.data as data
import os
import SimpleITK as sitk
import numpy as np
import random
import cv2
import pandas as pd
from utils.processing import crop
from utils.heatmap_generator import HeatmapGenerator

from DataAugmentation.augmentation_IVDlocation import \
    random_rotate_around_z_axis, random_translate, random_elastic_deformation, to_tensor, random_flip_3d


def landmark_extractor(landmarks):
    """
    return a list of the landmarks
    :param landmarks: pandas.Dataframe
    """
    labels = landmarks.columns[1:].tolist()  # exclude the 'axis' column
    list_landmarks = []
    for label in labels:
        list_landmarks.append(np.array(landmarks[label]))

    return list_landmarks



def read_data(case_dir):
    """
    read data from a given path
    """
    dict_images = dict()
    list_files = ['MR_512.nii.gz', 'landmarks_512.csv', 'Mask_512.nii.gz']

    for file_name in list_files:
        file_path = case_dir + '/' + file_name
        assert os.path.exists(file_path), case_dir + ' does not exist!'

        if file_name.split('.')[-1] == 'csv':
            landmarks = pd.read_csv(file_path)
            dict_images['list_landmarks'] = landmark_extractor(landmarks)
        elif file_name.split('.')[0].split('_')[0] == 'MR':
            dict_images['MR'] = sitk.ReadImage(file_path, sitk.sitkFloat32)
            dict_images['MR'] = sitk.GetArrayFromImage(dict_images['MR'])[np.newaxis, :, :, :]
        elif file_name.split('.')[0].split('_')[0] == 'Mask':
            dict_images['Mask'] = sitk.ReadImage(file_path, sitk.sitkInt16)
            dict_images['Mask'] = sitk.GetArrayFromImage(dict_images['Mask'])[np.newaxis, :, :, :]

    return dict_images


def pre_processing(dict_images, phase):
    MR = dict_images['MR']
    MR = np.clip(MR / 2048, a_min=0, a_max=1)
    Mask = dict_images['Mask']
    _, D, H, W = MR.shape

    heatmap_generator = HeatmapGenerator(image_size=(D, H, W),
                                         sigma=2.,
                                         scale_factor=1.,
                                         normalize=True,
                                         size_sigma_factor=8,
                                         sigma_scale_factor=2,
                                         dtype=np.float32)
    list_landmarks = dict_images['list_landmarks']

    index = random.randint(10, 18)
    while True in np.isnan(list_landmarks[index]):
        index = random.randint(10, 18)

    heatmap = heatmap_generator.generate_heatmap(landmark=list_landmarks[index])[np.newaxis, :, :, :]
    Mask = np.where(Mask == index + 1, 1, 0)  # just IVD

    if D > 12:
        start = random.choice([i for i in range(D - 12 + 1)])
        MR = crop(MR, start=start, end=start + 12, axis='z')
        heatmap = crop(heatmap, start=start, end=start + 12, axis='z')
        Mask = crop(Mask, start=start, end=start + 12, axis='z')

    list_images = [MR, heatmap]
    transform = {'train': train_transform, 'val': val_transform}[phase]
    list_images = transform(list_images, Mask)

    return list_images


def train_transform(list_images, Mask):

    # Random flip along z and x axis
    list_images = random_flip_3d(list_images, list_axis=(0, 2), p=0.5)

    # Random rotation
    list_images = random_rotate_around_z_axis(list_images,
                                              list_angles=(0, 3, 6, 9, -3, -6, -9),
                                              list_border_value=(0, 0, 0),
                                              list_interp=(cv2.INTER_NEAREST, cv2.INTER_NEAREST, cv2.INTER_NEAREST),
                                              p=0.3)

    list_images = random_translate(list_images, Mask,  # [MR, spine_heatmap, Mask]
                                   p=0.5,
                                   max_shift=1)

    # To torch tensor
    list_images = to_tensor(list_images)
    return list_images


def val_transform(list_images):
    list_images = to_tensor(list_images)
    return list_images


class IVDLocationDataset(data.Dataset):
    def __init__(self, catalogue, num_samples_per_epoch, phase, path):

        self.num_samples_per_epoch = num_samples_per_epoch
        self.phase = phase
        self.cases = catalogue[phase].dropna()

        self.list_case_id = [os.path.join(path, self.cases[i]) for i in range(len(self.cases))]

        random.shuffle(self.list_case_id)
        self.num_case = len(self.list_case_id)

    def __getitem__(self, index_):
        if index_ <= self.num_case - 1:
            case_id = self.list_case_id[index_]
        else:
            new_index_ = index_ - (index_ // self.num_case) * self.num_case
            case_id = self.list_case_id[new_index_]

        dict_images = read_data(case_id)
        list_images = pre_processing(dict_images, self.phase)

        return list_images  # [MR, target_heatmap]

    def __len__(self):
        return self.num_samples_per_epoch


def get_loader(catalogue, batch_size=2,
               num_samples_per_epoch=1, num_works=4,
               phase='train',
               path='../../../Data/Spine_Segmentation'):
    dataset = IVDLocationDataset(catalogue=catalogue,
                                     num_samples_per_epoch=num_samples_per_epoch,
                                     phase=phase, path=path)

    loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_works,
                             pin_memory=True)

    return loader