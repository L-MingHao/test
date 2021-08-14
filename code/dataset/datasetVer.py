import os
import random
import numpy as np 
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt

import cv2
import torch 
import torch.utils.data
import SimpleITK as sitk
import sys
sys.path.append("..")
from utils.processing import crop
from utils.heatmap_generator import HeatmapGenerator
from network.SCnet import SpatialConfigurationNet
import torch

# Usage of Ver train and Ver prediction
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_names, img_dir, num_classes=1, transform=None, train=True):
        """
        Args:
            img_names (list): Image Name.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        """
        self.img_names = img_names
        self.img_dir = img_dir
        self.num_classes = num_classes
        self.transform = self.train_transform
        self.train = train

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]


        # data read
        img_path = os.path.join(self.img_dir, img_name)

        dict_images = self.read_data(img_path)
        list_images = self.pre_processing(dict_images)
        
        if self.train == True:
            augmented = self.train_transform(list_images)
            img = augmented[0]
            mask = augmented[1]
        else:
            augmented = self.val_transform(list_images)
            img = list_images[0]
            mask = list_images[1]
        
        landmark = dict_images['list_landmarks']
        return img, mask, landmark, {'img_name': img_name} 
#        return list_images[-1], {'img_name': img_name} 

    def train_transform(self, list_images):
        # list_images = [Input, Label(gt_dose), possible_dose_mask]
        # Random flip
        list_images = random_flip_3d(list_images, list_axis=(0, 2), p=0.8)
    
        # Random rotation
        list_images = random_rotate_around_z_axis(list_images,
                                                  list_angles=(0, 3, 6, 9, -3, -6, -9),
                                                  list_border_value=(0, 0, 0),
                                                  list_interp=(cv2.INTER_NEAREST, cv2.INTER_NEAREST, cv2.INTER_NEAREST),
                                                  p=0.3)
    
#         Random translation, but make use the region can receive dose is remained
        list_images = random_translate(list_images,  # [MR, Mask]
                            mask=list_images[1][0, :, :, :],  # Mask
                            p=0.8,
                            max_shift=8,
                            list_pad_value=[0, 0, 0])
    
        # To torch tensor
        list_images = to_tensor(list_images)
        return list_images
    
    def val_transform(self, list_images):
        list_images = to_tensor(list_images)
        return list_images


    def read_data(self,case_dir):
        """
        read data from a given path.
        """
        dict_images = {}
        list_files = ['MR.nii.gz', 'landmarks.csv', 'Mask.nii.gz']
        for file_name in list_files:
            file_path = case_dir + '/' + file_name
#            assert os.path.exists(file_path), case_dir + ' does not exist!'

            if file_name.split('.')[-1] == 'csv':
                landmarks = pd.read_csv(file_path)
                dict_images['list_landmarks'] = self.landmark_extractor(landmarks)
            elif file_name.split('.')[0] == 'MR':
                dict_images['MR'] = sitk.ReadImage(file_path, sitk.sitkFloat32)
                dict_images['MR'] = sitk.GetArrayFromImage(dict_images['MR'])[np.newaxis, :, :, :]
            else:
                dict_images['Mask'] = sitk.ReadImage(file_path, sitk.sitkInt16)
                dict_images['Mask'] = sitk.GetArrayFromImage(dict_images['Mask'])[np.newaxis, :, :, :]

        return dict_images

    def landmark_extractor(self, landmarks):
        """
        Return a list of the landmarks
        """
        labels = landmarks.columns[1:].tolist()  # exclude the 'axis' column
        list_landmarks = []
        for label in labels:
            list_landmarks.append(np.array(landmarks[label]))

        return list_landmarks

    def pre_processing(self, dict_images):
        MR = dict_images['MR']
        MR = np.clip(MR / 2048, a_max=1, a_min=0)
        _, D, H, W = MR.shape
        
        MASK = dict_images['Mask']
#        Mask_new = np.zeros((1, D, H, W), np.int16)
        
        if self.num_classes == 1:
            # ver and IVDs 
            ver = [2, 3, 4, 5, 6, 7, 8, 9, 10]
            IVDs = [11, 12, 13, 14, 15, 16, 17, 18, 19]
            Mask_new = np.where((MASK == 1), 1, MASK)
            for i in range(len(ver)):    
                Mask_new = np.where((Mask_new == ver[i]), 1, Mask_new)
            for j in range(len(IVDs)):
                Mask_new = np.where((Mask_new == IVDs[j]), 0, Mask_new)
        else:
            Mask_new = (MASK != 0)
        

        heatmap_generator = HeatmapGenerator(image_size=(D, H, W),
                                             sigma=2,
                                             spine_heatmap_sigma=20,
                                             scale_factor=3.,
                                             normalize=True,
                                             size_sigma_factor=10,
                                             sigma_scale_factor=2,
                                             dtype=np.float32)
        
        spine_heatmap = heatmap_generator.generate_spine_heatmap(list_landmarks=dict_images['list_landmarks'])
        heatmaps = heatmap_generator.generate_heatmaps(list_landmarks=dict_images['list_landmarks'])
        centroid_coordinate = [round(i) for i in ndimage.center_of_mass(spine_heatmap)]  # (0, z, y, x)

        start_x = centroid_coordinate[-1] - W // 4
        end_x = centroid_coordinate[-1] + W // 4
        MR = crop(MR, start=start_x, end=end_x, axis='x')
        spine_heatmap = crop(spine_heatmap, start=start_x, end=end_x, axis='x')
        heatmaps = crop(heatmaps, start_x, end=end_x, axis='x')
        MASK = crop(MASK, start=start_x, end=end_x, axis='x')
        Mask_new = crop(Mask_new, start=start_x, end=end_x, axis='x')

        if D > 12:
            start_z = random.choice([i for i in range(D - 12 + 1)])
            MR = crop(MR, start=start_z, end=start_z + 12, axis='z')
            spine_heatmap = crop(spine_heatmap, start=start_z, end=start_z + 12, axis='z')
            heatmaps = crop(heatmaps, start_z, end=start_z + 12, axis='z')
            MASK = crop(MASK, start=start_z, end=start_z + 12, axis='z')
            Mask_new = crop(Mask_new, start=start_z, end=start_z + 12, axis='z')
            
            
        ## Add bottom and delete up
        MR_n = np.zeros((_, 12, 256, 128))
        Mask_n = np.zeros((_, 12, 256, 128))

        MR_n[:,:,:H-2,:] = MR[:,:,2:,:]
        Mask_n[:,:,:H-2,:] = Mask_new[:,:,2:,:]
        for i in range(2):
            MR_n[:,:,H-2+i,:] = MR[:,:,H-i-1,:]
            Mask_n[:,:,H-2+i,:] = Mask_new[:,:,H-i-1,:]
        
        MR = MR_n
        Mask_new = Mask_n
        # FIXME crop patches
#        MR_ALL = []
#        MASK_ALL = []
#        if self.train == False:
#            for start_y in [0, H//4 , H//2]:
#                end_y = start_y + H // 2
#                MR_ALL.append(crop(MR, start=start_y, end=end_y, axis='y'))
#                MASK_ALL.append(crop(Mask_new, start=start_y, end=end_y, axis='y'))
#            start_y = random.choice((0, H // 4, H // 2))
#        else:
#            start_y = random.randint(0, H//2)
#        end_y = start_y + H // 2
#        MR = crop(MR, start=start_y, end=end_y, axis='y')
#        MASK = crop(MASK, start=start_y, end=end_y, axis='y')
#        spine_heatmap = crop(spine_heatmap, start=start_y, end=end_y, axis='y')
#        heatmaps = crop(heatmaps, start_y, end=end_y, axis='y')
#        Mask_new = crop(Mask_new, start=start_y, end=end_y, axis='y')
        
#        return [MR, spine_heatmap, heatmaps, MASK, Mask_new, MR_ALL ,MASK_ALL, centroid_coordinate[-1]]  # (1, 12, 256, 256), (1, 12, 256, 256), (19, 12, 256, 256)
        
        return [MR, Mask_new]
#        return [MR, spine_heatmap, heatmaps, MASK, Mask_new, centroid_coordinate[-1]]

# No Usage
class Dataset2(torch.utils.data.Dataset):
    def __init__(self, img_names, img_dir, transform=None):
        self.img_ids = img_names
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
#        print(os.path.join(self.img_dir, img_id + '/Fianl.jpg'))
        img = cv2.imread(os.path.join(self.img_dir, img_id + '/Fianl.jpg'),cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.img_dir, img_id + '/map.jpg'),cv2.IMREAD_GRAYSCALE)

        
        img = img.astype('float32') / 255
        mask = mask.astype('float32') / 255
        return img[np.newaxis, :, :], mask[np.newaxis, :, :], {'img_id': img_id}

# Random flip
def random_flip_3d(list_images, list_axis=(0, 1, 2), p=0.5):
    if random.random() <= p:
        if 0 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, ::-1, :, :]
        if 1 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, :, ::-1, :]
        if 2 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, :, :, ::-1]

    return list_images


# Random rotation using OpenCV
def random_rotate_around_z_axis(list_images,
                                list_angles,
                                list_interp,
                                list_border_value,
                                p=0.5):
    if random.random() <= p:
        # Randomly pick an angle list_angles
        _angle = random.sample(list_angles, 1)[0]
        # Do not use random scaling, set scale factor to 1
        _scale = 1.

        for image_i in range(len(list_images)):
            for chan_i in range(list_images[image_i].shape[0]):
                for slice_i in range(list_images[image_i].shape[1]):
                    rows, cols = list_images[image_i][chan_i, slice_i, :, :].shape
                    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), _angle, scale=_scale)
                    list_images[image_i][chan_i, slice_i, :, :] = \
                        cv2.warpAffine(list_images[image_i][chan_i, slice_i, :, :],
                                       M,
                                       (cols, rows),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=list_border_value[image_i],
                                       flags=list_interp[image_i])
    return list_images


# Random translation
def random_translate(list_images, mask, p, max_shift=20, list_pad_value=(0, 0, 0)):
    if random.random() <= p:
        exist_mask = np.where(mask > 0)
        ori_z, ori_h, ori_w = list_images[0].shape[1:]  # MR

        bz = min(max_shift - 1, np.min(exist_mask[0]))
        ez = max(ori_z - 1 - max_shift, np.max(exist_mask[0]))
        bh = min(max_shift - 1, np.min(exist_mask[1]))
        eh = max(ori_h - 1 - max_shift, np.max(exist_mask[1]))
        bw = min(max_shift - 1, np.min(exist_mask[2]))
        ew = max(ori_w - 1 - max_shift, np.max(exist_mask[2]))

        for image_i in range(len(list_images)):
            list_images[image_i] = list_images[image_i][:, bz:ez + 1, bh:eh + 1, bw:ew + 1]

        # Pad to original size
        list_images = random_pad_to_size_3d(list_images,
                                            target_size=[ori_z, ori_h, ori_w],
                                            list_pad_value=list_pad_value)
    return list_images


# To tensor, images should be C*Z*H*W
def to_tensor(list_images):
    for image_i in range(len(list_images)):
        list_images[image_i] = torch.from_numpy(list_images[image_i].copy()).float()
    return list_images


# Pad
def random_pad_to_size_3d(list_images, target_size, list_pad_value):
    _, ori_z, ori_h, ori_w = list_images[0].shape[:]
    new_z, new_h, new_w = target_size[:]

    pad_z = new_z - ori_z
    pad_h = new_h - ori_h
    pad_w = new_w - ori_w

    pad_z_1 = random.randint(0, pad_z)
    pad_h_1 = random.randint(0, pad_h)
    pad_w_1 = random.randint(0, pad_w)

    pad_z_2 = pad_z - pad_z_1
    pad_h_2 = pad_h - pad_h_1
    pad_w_2 = pad_w - pad_w_1

    output = []
    for image_i in range(len(list_images)):
        _image = list_images[image_i]
        output.append(np.pad(_image,
                             ((0, 0), (pad_z_1, pad_z_2), (pad_h_1, pad_h_2), (pad_w_1, pad_w_2)),
                             mode='constant',
                             constant_values=list_pad_value[image_i])
                      )
    return output
if __name__ == '__main__':
    img_names = os.listdir('./RawData')
#    img_names = ['Case27']
    d = Dataset(img_names, "./RawData" )
#    for i in range(12):
#        plt.matshow(d[0][1][0,i,:,:], cmap=plt.cm.gray)
    hashtable = dict()
    for i in range(len(d)):
        hashtable[d[i][-1]['img_name']] = d[i][-2]
        print(i)
    np.save('../outputs/data/LocPos.npy', hashtable)
#    train_loader = torch.utils.data.DataLoader(
#        d,
#        batch_size=1,
#        shuffle=True,
#        num_workers=1,
#        drop_last=True)
            
    
#    model = SpatialConfigurationNet(num_labels=1)
##    model = model.cuda()
#    model.load_state_dict(torch.load("model-2021-05-19.pth"))
#    model = model.cuda()
#    model.eval() 
#    with torch.no_grad():
#        for input, target, _  in train_loader:
#            input = input.cuda()
#            target = target.cuda()
#            output = model(input)[0]
#            output2 = torch.sigmoid(output)
#            output_ = output2 > 0.5
#            target_ = target[0,:,:,:,:] > 0.5
#            smooth = 1e-5
#            intersection = (output_ & target_).sum()
#            union = (output_ | target_).sum()  
#            
#            print((intersection + smooth) / (union + smooth))
#            output = output.detach().numpy()
#            output2 = output2.detach().numpy()
#            for k in range(12):
#                plt.matshow(target_[0,k,:,:], cmap=plt.cm.gray)
#                plt.savefig("target{}.png".format(k))
#    #            plt.show()
#                plt.matshow(output[0,k,:,:], cmap=plt.cm.gray)
#                plt.savefig("output{}.png".format(k))
#    #            plt.show()
#                plt.matshow(output2[0,k,:,:], cmap=plt.cm.gray)
#                plt.savefig("output2{}.png".format(k))
#            plt.show()
#    model = model.cuda()
#    for opoch in range(10):
#        for input, target, _  in train_loader:
    #        input = input.cuda()
    #        target = target.cuda()
#            plt.matshow(input[0,0,0,:,:], cmap=plt.cm.gray)
#            plt.show()
        # compute output 
#        output = model(input).detach().numpy()
#        print(target2.shape)
#        print(target3.shape)
#        for k in range(12):
#            plt.matshow(input[0,0,k,:,:], cmap=plt.cm.gray)
#            plt.show()
#            plt.matshow(output[0,0,k,:,:], cmap=plt.cm.gray)
#            plt.show()
#            plt.matshow(target[0,0,k,:,:], cmap=plt.cm.gray)
#            plt.show()
#        targetss = torch.sum(target2, axis=2)
#        plt.matshow(output[0,0,:,:], cmap=plt.cm.gray)
#        plt.show()