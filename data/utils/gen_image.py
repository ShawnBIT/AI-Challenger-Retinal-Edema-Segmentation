import cv2
import numpy as np
import os 
import os.path as osp
from os.path import exists
import numpy as np



# generate the mask label 0,1
def gen_label(ant_path,label_path):

    for pid in sorted(os.listdir(ant_path)):
        if pid.startswith('.'):
            continue
        pic_list = sorted(os.listdir(osp.join(ant_path,pid)))
        for i,pic_name in enumerate(pic_list):
            img = cv2.imread(osp.join(ant_path,pid,pic_list[i]),cv2.IMREAD_ANYCOLOR) 
            img[img != 0] = 1
            save_dir = osp.join(label_path,pid)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(osp.join(save_dir,'%02d.png'%i),img)

# generate img_mask
def gen_img_mask(img_path,image_path,root):

    for pid in sorted(os.listdir(img_path)):
        print(pid)
        if pid.startswith('.'):
            continue
        pic_list = sorted(os.listdir(osp.join(img_path,pid)))
        for i,pic_name in enumerate(pic_list):
            img = cv2.imread(osp.join(img_path,pid,pic_list[i]),cv2.IMREAD_ANYCOLOR)
            mask = cv2.imread(osp.join(root,'mask',pid,pic_list[i]),cv2.IMREAD_ANYCOLOR)
            mask[mask != 0] = 1
            img_mask = img*mask
            save_dir = osp.join(image_path,pid)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(osp.join(save_dir,'%02d.png'%i),img_mask)

# generate the 3D image
def gen_image(img_path,image_path):

    for j,pid in enumerate(sorted(os.listdir(img_path))):
        print(j)
        if pid.startswith('.'):
            continue
        pic_list = sorted(os.listdir(osp.join(img_path,pid)),key=lambda x:int(x[:-4]))
        for i,pic_name in enumerate(pic_list):
        
            if i == 0:
                img1 = cv2.imread(osp.join(img_path,pid,pic_list[i]),cv2.IMREAD_ANYCOLOR)
                img2 = cv2.imread(osp.join(img_path,pid,pic_list[i]),cv2.IMREAD_ANYCOLOR)
                img3 = cv2.imread(osp.join(img_path,pid,pic_list[i+1]),cv2.IMREAD_ANYCOLOR)
            elif i == len(pic_list) - 1:

                img1 = cv2.imread(osp.join(img_path,pid,pic_list[i-1]),cv2.IMREAD_ANYCOLOR)
                img2 = cv2.imread(osp.join(img_path,pid,pic_list[i]),cv2.IMREAD_ANYCOLOR)
                img3 = cv2.imread(osp.join(img_path,pid,pic_list[i]),cv2.IMREAD_ANYCOLOR)
            else:
                img1 = cv2.imread(osp.join(img_path,pid,pic_list[i-1]),cv2.IMREAD_ANYCOLOR)
                img2 = cv2.imread(osp.join(img_path,pid,pic_list[i]),cv2.IMREAD_ANYCOLOR)
                img3 = cv2.imread(osp.join(img_path,pid,pic_list[i+1]),cv2.IMREAD_ANYCOLOR)
            mix_img = np.array([img1,img2,img3])
            mix_img = mix_img.swapaxes(2,0)
            mix_img = mix_img.swapaxes(0,1)
            save_dir = osp.join(image_path,pid)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(osp.join(save_dir,'%03d.png'%i),mix_img)



def main():
    #root_list = ['../dataset/blood/202','../dataset/blood/203','../dataset/blood/224','../dataset/blood/249','../dataset/blood/259','../dataset/blood/277']
    root_list = ['../dataset/ori/Edema_trainingset','../dataset/ori/Edema_validationset','../dataset/ori/Edema_testset']  # 277 is written protected. chmod 777 277 to solve the problem
    #root_list = ['../dataset/train']
    for root in root_list:
        #gen_img_mask(osp.join(root,'img'),osp.join(root,'img_mask'),root)
        gen_image(osp.join(root,'original_images'),osp.join(root,'trans3channel_images'))
        print('dataset #{} has been generated!'.format(root))

# img_path = './img'
# image_path = './image'

# label_path = './label'
# ant_path = './ant'

if __name__ == '__main__':
    main()

        
