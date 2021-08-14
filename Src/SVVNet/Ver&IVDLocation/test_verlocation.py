# -*- encoding: utf-8 -*-
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
import cv2
if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

if os.path.abspath('../..') not in sys.path:
    sys.path.insert(0, os.path.abspath('../..'))

from Evaluate.evaluate import *
from model import SpatialConfigurationNet
from NetworkTrainer.network_trainer import *
from DataLoader.dataloader_Verlocation import landmark_extractor
from utils.heatmap_generator import HeatmapGenerator
from utils.tools import csv_to_catalogue
from utils.processing import crop
from post_processing import post_processing



def read_data(case_dir):
    """
    read data from a given path
    """
    dict_images = dict()
    list_files = ['MR_512.nii.gz', 'landmarks_512.csv', ]
    # In fact, there is no Mask during inference, so we cannot load it.

    for file_name in list_files:
        file_path = case_dir + '/' + file_name
        assert os.path.exists(file_path), case_dir + ' does not exist!'

        if file_name.split('.')[-1] == 'csv':
            landmarks = pd.read_csv(file_path)
            dict_images['list_landmarks'] = landmark_extractor(landmarks)
        elif file_name.split('.')[0].split('_')[0] == 'MR':
            dict_images['MR'] = sitk.ReadImage(file_path, sitk.sitkFloat32)
            dict_images['MR'] = sitk.GetArrayFromImage(dict_images['MR'])
        elif file_name.split('.')[0].split('_')[0] == 'Mask':
            dict_images['Mask'] = sitk.ReadImage(file_path, sitk.sitkInt16)
            dict_images['Mask'] = sitk.GetArrayFromImage(dict_images['Mask'])[np.newaxis, :, :, :]

    return dict_images


def pre_processing(dict_images):
    MR = dict_images['MR']
    MR = np.clip(MR / 2048, a_max=1, a_min=0)
    _, D, H, W = MR.shape
    MR_new = np.zeros((_, D, 256, 256))
    for i in range(D):
        MR_new[0, i, :, :] = cv2.resize(MR[0, i, :, :], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    MR = MR_new
    _, D, H, W = MR.shape

    spine_heatmap = dict_images['Heatmap']

    centroid_coordinate = [round(i) for i in ndimage.center_of_mass(spine_heatmap)]  # (0, z, y, x)
    center = centroid_coordinate[-1] - 128

    start_x = centroid_coordinate[-1] - W // 4 - 128
    end_x = centroid_coordinate[-1] + W // 4 - 128
    MR = crop(MR, start=start_x, end=end_x, axis='x')

    if D > 12:
        #   start_z = random.choice([i for i in range(D - 12 + 1)])
        start_z = 0
        MR = crop(MR, start=start_z, end=start_z + 12, axis='z')

    ## Add bottom and delete up
    MR_n = np.zeros((_, 12, 256, 128))
    MR_n[:, :, :H - 3, :] = MR[:, :, 3:, :]
    for i in range(3):
        MR_n[:, :, H - 3 + i, :] = MR[:, :, H - i - 1, :]

    MR = MR_n

    return [MR, center]


def copy_sitk_imageinfo(image1, image2):
    image2.SetSpacing(image1.GetSpacing())
    image2.SetDirection(image1.GetDirection())
    image2.SetOrigin(image1.GetOrigin())

    return image2


def to_tensor(list_images):
    for image_i in range(len(list_images)):
        list_images[image_i] = torch.from_numpy(list_images[image_i].copy()).float()
    return list_images


# Input is B*C*Z*H*W
def flip_3d(input_, list_axes):

    input_ = torch.flip(input_, list_axes)

    return input_


def test_time_augmentation(trainer, input_, TTA_mode):
    """
    :param input_: 5D tensor (B, C, D, H, W)
    """
    list_pred = []

    for list_flip_axes in TTA_mode:
        # Do Augmentation before forward
        augmented_input = flip_3d(input_, list_flip_axes)
        pred = trainer.setting.network(augmented_input)  # (B, num_classes, D, H, W) probability distribution

        # Aug back to original order
        pred = flip_3d(pred, list_flip_axes)  # probability distribution
        # print(pred.cpu().numpy().shape)

        list_pred.append(pred)  # probability distribution

    list_pred = torch.stack(list_pred)

    return torch.mean(list_pred, dim=0)


def inference(trainer, list_case_dirs, save_path, do_TTA=False):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if do_TTA:
      TTA_mode = [[], [2], [4], [2, 4]]
    else:
      TTA_mode = [[]]

    with torch.no_grad():
        trainer.setting.network.eval()
        for case_dir in tqdm(list_case_dirs):
            assert os.path.exists(case_dir), case_dir + 'does not exist!'
            case_id = case_dir.split('/')[-1]


            dict_images = read_data(case_dir)
            temp = dict_images['MR']
            FinalMask = np.zeros((len(temp), 256, 128))

            for index in range(len(temp) - 12 + 1):
                MR, center = post_processing(dict_images)
                augmented = to_tensor(MR)
                img = augmented[0]

                input = img.cuda()
                output = trainer.setting.network(input)
                output = torch.sigmoid(output).cpu().numpy()

                output1 = np.where(output > 0.5, 1, 0)
                FinalMask[index:index + 12, :, :] += output1[0]
                FinalMask = np.where(FinalMask > 0.5, 1, 0)

            if not os.path.exists('../../../Output/Ver_Location/' + case_id):
                os.mkdir('../../../Output/Ver_Location/' + case_id)
            np.save('../../../Output/{}/{}/PredVerAll.npy'.format('Ver_Location', case_id), FinalMask)
            FinalImgO = np.zeros((256, 128))
            for k in range(len(temp)):
                FinalImgO += FinalMask[k, :, :]
            FinalImgO = (FinalImgO != 0)
            FinalImgO = (FinalImgO * 255).astype('uint8')
            cv2.imwrite(os.path.join('../../../Output', 'Ver_Location', case_id, 'PredVer' + '.bmp'),
                        FinalImgO, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            torch.cuda.empty_cache()


if __name__ == "__main__":
    if not os.path.exists('../../../Data/Spine_Segmentation'):  # this is base dataset
        raise Exception('Spine_Segmentation should be prepared before testing, ' +
                        'please run prepare_3D.py and landmark generation.py')

    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_id', type=int, default=0,
                        help='GPU id used for testing (default: 0)')
    parser.add_argument('--model_path', type=str,
                        default='../../../Output/Ver_Location/best_val_evaluation_index.pkl')
    parser.add_argument('--TTA', type=int, default=1,
                        help='do test-time augmentation, default True')

    parser.add_argument('--model_type', type=str, default='SpatialConfigurationNet')
    parser.add_argument('--catalogue', type=int, default=0)
    args = parser.parse_args()

    trainer = NetworkTrainer()
    trainer.setting.project_name = 'IVD_Segmentation'
    trainer.setting.output_dir = '../../../Output/Ver_Location'

    if args.model_type == 'SpatialConfigurationNet':
        trainer.setting.network = SpatialConfigurationNet(num_labels=20)
        print('Loading SpatialConfigurationNet !')


    # Load model weights
    print(args.model_path)
    trainer.init_trainer(ckpt_file=args.model_path,
                         list_GPU_ids=[args.GPU_id],
                         only_network=True)

    # Start inference
    print('\n\n# Start inference !')

    csv_path = '../../Catalogue' + '/' + str(args.catalogue) + '.csv'
    catalogue = csv_to_catalogue(csv_path)
    path = '../../../Data/Spine_Segmentation'
    cases = catalogue['test'].dropna()
    list_case_dirs = [os.path.join(path, cases[i]) for i in range(len(cases))]


    print('\n\n# Start evaluation !')
    inference(trainer, list_case_dirs, save_path=os.path.join(trainer.setting.output_dir, 'Prediction'),
              do_TTA=args.TTA)

