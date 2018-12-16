from torch.utils.data import Dataset
from os.path import join,exists
from PIL import Image
import torch
import os
import numpy as np 
import torchvision.transforms as tt
import data.seg_transforms as st
import PIL
import random


class SegList(Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        if self.phase == 'train':
            data = [Image.open(join(self.data_dir, self.image_list[index]))]
            data.append(Image.open(join(self.data_dir, self.label_list[index])))
            data = list(self.transforms(*data))
            data = [data[0],data[1].long()]
            return tuple(data)
        
        if self.phase == 'val':
            pic = sorted(os.listdir(join(self.data_dir, self.image_list[index])))[0]
            img = Image.open(join(self.data_dir, self.image_list[index],pic))
            w,h = img.size

            image = torch.zeros(len(os.listdir(join(self.data_dir, self.image_list[index]))),3,h,w)
            label = torch.zeros(len(os.listdir(join(self.data_dir, self.image_list[index]))),h,w)
            imt = torch.zeros(len(os.listdir(join(self.data_dir, self.image_list[index]))),h,w)

            for i,pic_name in enumerate(sorted(os.listdir(join(self.data_dir, self.image_list[index])))):
                data = [Image.open(join(self.data_dir, self.image_list[index],pic_name))]

                imt_3 = torch.from_numpy(np.array(data[0]).transpose(2,0,1))

                imt_i = imt_3[1]
                imt[i,:,:] = imt_i

                label_name = str(int(pic_name.split('.')[0])+1)+'.bmp'
                #label_name = pic_name
                data.append(Image.open(join(self.data_dir, self.label_list[index],label_name)))
                data = list(self.transforms(*data))
                image[i,:,:,:] = data[0]
                label[i,:,:] = data[1]

            return (image,label.long(),imt)


        elif self.phase == 'test':
            # for identity image size
            pic = sorted(os.listdir(join(self.data_dir, self.image_list[index])))[0]
            img = Image.open(join(self.data_dir, self.image_list[index],pic))
            w,h = img.size

            image = torch.zeros(len(os.listdir(join(self.data_dir, self.image_list[index]))),3,h,w)

            for i,pic_name in enumerate(sorted(os.listdir(join(self.data_dir, self.image_list[index])),key=lambda x:int(x[:-4]))):
                data = [Image.open(join(self.data_dir, self.image_list[index],pic_name))]
                data = list(self.transforms(*data))
                image[i,:,:,:] = data[0]

            return image,self.image_list[index][:-4].split('/')[-1]

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        
        if self.phase != 'test':
            label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]

        if self.phase != 'test':
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)
        
        if self.phase == 'train':
            print('Total train image is : %d'%len(self.image_list))
        else:
            print('Total val pid is : %d'%len(self.image_list))
