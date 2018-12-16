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

class SegList3D(Dataset):
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
        if self.phase in ['val','train']:
            image = torch.zeros(32,1,128,128,128)
            label = torch.zeros(32,128,128,128)

            for m in range(4):
                for n in range(8):
                    box = (m*128,n*128,(m+1)*128,(n+1)*128)
                    for i,pic_name in enumerate(sorted(os.listdir(join(self.data_dir, self.image_list[index])),key=lambda x:int(x[:-4]))):
                        data = [Image.open(join(self.data_dir, self.image_list[index],pic_name)).crop(box)]
                        label_name = pic_name
                        data.append(Image.open(join(self.data_dir, self.label_list[index],label_name)).crop(box))
                        data = list(self.transforms(*data))
                        image[m+4*n,:,i,:,:] = data[0]
                        label[m+4*n,i,:,:] = data[1]

            # order = [i for i in range(32)]
            # order_shuffle = random.shuffle(order)
            # print(order)
            # print(order_shuffle)
            # image_shuffle = torch.zeros(32,1,128,128,128)
            # label_shuffle = torch.zeros(32,128,128,128)
            # for j in order:
            #     image_shuffle[o,:,:,:,:] = image[order_shuffle[j]]
            #     label_shuffle[o,:,:,:] = label[order_shuffle[j]]

            return (image,label.long())



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

class SegList_3D(Dataset):
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
        
        if self.phase == 'train' or 'val':
            # pic = sorted(os.listdir(join(self.data_dir, self.image_list[index])))[0]
            # img = Image.open(join(self.data_dir, self.image_list[index],pic))
            w,h = 256,256

            image = torch.zeros(1,h,w,128)
            label = torch.zeros(h,w,128)
            for i,pic_name in enumerate(sorted(os.listdir(join(self.data_dir, self.image_list[index])))):
                data = [Image.open(join(self.data_dir, self.image_list[index],pic_name)).resize((w,h))]
                label_name = pic_name
                data.append(Image.open(join(self.data_dir, self.label_list[index],label_name)).resize((w,h),PIL.Image.NEAREST))
                data = list(self.transforms(*data))
                image[:,:,:,i] = data[0]
                label[:,:,i] = data[1]

            return (image,label.long())


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



class DetList(Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

        self.resize = tt.Resize(512)

    def __getitem__(self, index):
        if self.phase == 'train':
            data = [Image.open(join(self.data_dir, self.image_list[index]))]
            class_num = label2class(Image.open(join(self.data_dir, self.label_list[index])))
            data = list(self.transforms(*data))
            data = [data[0],class_num]
            return tuple(data)

        elif self.phase == 'val':
            image = torch.zeros(len(os.listdir(join(self.data_dir, self.image_list[index]))),3,224,224)
            label = torch.zeros(len(os.listdir(join(self.data_dir, self.image_list[index]))),3)
            for i,pic_name in enumerate(sorted(os.listdir(join(self.data_dir, self.image_list[index])))):
                data = [Image.open(join(self.data_dir, self.image_list[index],pic_name))]
                label_name = str(int(pic_name.split('.')[0])+1)+'.bmp'
                class_num = label2class(Image.open(join(self.data_dir, self.label_list[index],label_name)))
                data = list(self.transforms(*data))
                image[i,:,:,:] = data[0]
                label[i,:] = class_num

            return (image,label)


        elif self.phase == 'test':
            # for identity image size
            pic = sorted(os.listdir(join(self.data_dir, self.image_list[index])))[0]
            img = Image.open(join(self.data_dir, self.image_list[index],pic))
            w,h = img.size

            image = torch.zeros(len(os.listdir(join(self.data_dir, self.image_list[index]))),3,h,w)
            image_det = torch.zeros(len(os.listdir(join(self.data_dir, self.image_list[index]))),3,224,224)

            for i,pic_name in enumerate(sorted(os.listdir(join(self.data_dir, self.image_list[index])),key=lambda x:int(x[:-4]))):
                data = [Image.open(join(self.data_dir, self.image_list[index],pic_name))]
                data_det = [(Image.open(join(self.data_dir, self.image_list[index],pic_name))).resize((224,224))]
                data = list(self.transforms(*data))
                data_det = list(self.transforms(*data_det))
                image[i,:,:,:] = data[0]
                image_det[i,:,:,:] = data_det[0]

            return image,self.image_list[index][:-4].split('/')[-1],image_det

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

class M_List(Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        if self.phase != 'test':
            data = [Image.open(join(self.data_dir, self.image_list[index]))]
            class_num = label2class(Image.open(join(self.data_dir, self.label_list[index])))
            data.append(Image.open(join(self.data_dir, self.label_list[index])))
            data = list(self.transforms(*data))
            data = [data[0],data[1].long(),class_num]
            return tuple(data)


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

'''
Background：0

PED: 128 3

SRF: 191 2

REA: 255 1
'''

def label1class(image):
    label = np.unique(np.array(image))
    if label.sum() == 128:
        class_num = torch.tensor([1])
    else:
        class_num = torch.tensor([0])

    return class_num.float()


def label2class(image):
    label = np.unique(np.array(image))
    if label.sum() == 0:
        class_num = torch.tensor([0,0,0])
    elif label.sum() == 255:#np.array([0,255]):
        class_num = torch.tensor([1,0,0])
    elif label.sum() == 191:#np.array([0,191]):
        class_num = torch.tensor([0,1,0])
    elif label.sum() == 128:#np.array([0,128]):
        class_num = torch.tensor([0,0,1])
    elif label.sum() == 255+191: #np.array([0,255,191]):
        class_num = torch.tensor([1,1,0])
    elif label.sum() == 255+128:#np.array([0,255,128]):
        class_num = torch.tensor([1,0,1])
    elif label.sum() == 191+128:#np.array([0,191,128]):
        class_num = torch.tensor([0,1,1])
    elif label.sum() == 255+191+128:#np.array([0,255,191,128]):
        class_num = torch.tensor([1,1,1])

    return class_num.float()

