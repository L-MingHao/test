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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_names, img_dir, num_classes=1, transform=None, train=True, start=0):
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
        self.train = train
        self.center = 128
        self.start = 0

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        # data read
        img_path = os.path.join(self.img_dir, img_name)

        dict_images = self.read_data(img_path)
        list_images = self.pre_processing(dict_images)
        
        augmented = self.val_transform(list_images)
        img = augmented[0]
        
        return img, self.center, {'img_name': img_name} 
    
    def val_transform(self, list_images):
        list_images = to_tensor(list_images)
        return list_images


    def read_data(self,case_dir):
        """
        read data from a given path.
        """
        dict_images = {}
        list_files = ['MR_512.nii.gz', 'pred_heatmap.nii.gz']
        for file_name in list_files:
            file_path = case_dir + '/' + file_name
            if file_name.split('.')[0] == 'MR_512':
                dict_images['MR'] = sitk.ReadImage(file_path, sitk.sitkFloat32)
                dict_images['MR'] = sitk.GetArrayFromImage(dict_images['MR'])[np.newaxis, :, :, :]
            elif file_name.split('.')[0] == 'pred_heatmap':
                dict_images['Heatmap'] = sitk.ReadImage(file_path, sitk.sitkFloat32)
                dict_images['Heatmap'] = sitk.GetArrayFromImage(dict_images['Heatmap'])[np.newaxis, :, :, :]                

        return dict_images

    def pre_processing(self, dict_images):
        MR = dict_images['MR']
        MR = np.clip(MR / 2048, a_max=1, a_min=0)
        _, D, H, W = MR.shape
        MR_new = np.zeros((_, D, 256, 256))
        for i in range(D):
            MR_new[0,i,:,:] = cv2.resize(MR[0,i,:,:], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        MR  = MR_new
        _, D, H, W = MR.shape
        
        spine_heatmap = dict_images['Heatmap']

        centroid_coordinate = [round(i) for i in ndimage.center_of_mass(spine_heatmap)]  # (0, z, y, x)
        self.center = centroid_coordinate[-1] - 128
        
        start_x = centroid_coordinate[-1] - W // 4 - 128
        end_x = centroid_coordinate[-1] + W // 4 - 128
        MR = crop(MR, start=start_x, end=end_x, axis='x')

        if D > 12:
#            start_z = random.choice([i for i in range(D - 12 + 1)])
            start_z = self.start
            MR = crop(MR, start=start_z, end=start_z + 12, axis='z')
            
        ## Add bottom and delete up
        MR_n = np.zeros((_, 12, 256, 128))
        MR_n[:,:,:H-3,:] = MR[:,:,3:,:]
        for i in range(3):
            MR_n[:,:,H-3+i,:] = MR[:,:,H-i-1,:]
        
        MR = MR_n
        
        return [MR]


# To tensor, images should be C*Z*H*W
def to_tensor(list_images):
    for image_i in range(len(list_images)):
        list_images[image_i] = torch.from_numpy(list_images[image_i].copy()).float()
    return list_images

if __name__ == '__main__':
    img_names = os.listdir('./TestData')
#    img_names = ['Case27']
    d = Dataset(img_names, "./TestData" )
    hashtable = dict()
    for i in range(len(d)):
        os.makedirs(os.path.join('../outputs/testdata',d[i][-1]['img_name']), exist_ok=True)
        sitk.WriteImage(sitk.GetImageFromArray(d[i][0][0,:,:,:]),'../outputs/testdata/{}/cropMR_256.nii.gz'.format(d[i][-1]['img_name']))
        hashtable[d[i][-1]['img_name']] = d[i][-2]
        print(i)
    np.save('../outputs/testdata/LocPos.npy', hashtable)
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