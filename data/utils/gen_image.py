import cv2
import numpy as np
import os 
import os.path as osp
from os.path import exists
import numpy as np

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
    root_list = ['../dataset/Edema_trainingset','../dataset/Edema_validationset','../dataset/Edema_testset'] 
    for root in root_list:
        gen_image(osp.join(root,'original_images'),osp.join(root,'trans3channel_images'))
        print('dataset #{} has been generated!'.format(root))


if __name__ == '__main__':
    main()

        
