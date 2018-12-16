import os
import os.path as osp
from os.path import exists

def gen_traintxt(root,stage,kind,data_path):
    f = open(osp.join(data_path,'train_'+kind+'s.txt'),'a')
    for pic_path in sorted(os.listdir(osp.join(root,stage,kind))):
        for pic_name in sorted(os.listdir(osp.join(root,stage,kind,pic_path)),key=lambda x:int(x[:-4])):
                f.write(osp.join(stage,kind,pic_path,pic_name))
                f.write('\n')
    #f.close()

def gen_valtxt(root,stage,kind,data_path):
    f = open(osp.join(data_path,'val_'+kind+'s.txt'),'w')
    for pic_path in sorted(os.listdir(osp.join(root,stage,kind))):
            f.write(osp.join(stage,kind,pic_path))
            f.write('\n')
    #f.close()

def gen_testtxt(root,stage,kind,data_path):
    f = open(osp.join(data_path,'test_'+kind+'s.txt'),'w')
    for pic_path in sorted(os.listdir(osp.join(root,stage,kind))):
            f.write(osp.join(stage,kind,pic_path))
            f.write('\n')

def gen_trainvaltxt(root,stage,kind,data_path):
    f = open(osp.join(data_path,'train_'+kind+'s.txt'),'a')
    for pic_path in sorted(os.listdir(osp.join(root,stage,kind))):
            f.write(osp.join(stage,kind,pic_path))
            f.write('\n')
    #f.close()

def main():
    root = '../dataset/ori'
    #stage_list = ['202','203','224','249','259','277']
    stage_list = ['test']
    kind_list = ['image']
    data_path = '../data_path/ori_3D_1'
    if not exists(data_path):
        os.makedirs(data_path)
    for stage in stage_list:
        for kind in kind_list:
            #print(stage)
            if stage == 'train':
                gen_traintxt(root,stage,kind,data_path)
            else:
                gen_traintxt(root,stage,kind,data_path)
            # if stage == 'train':
            #     gen_traintxt(root,stage,kind,data_path)
            # elif stage == 'val' or 'test':
            #     gen_valtxt(root,stage,kind,data_path)


if __name__ == '__main__':
    main()

        