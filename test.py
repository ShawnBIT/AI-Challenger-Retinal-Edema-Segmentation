#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@Author:ShawnWang

##### System library #####
import os
import os.path as osp
from os.path import exists
import argparse
import json
import logging
import time
import numpy as np
##### pytorch library #####
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
##### My own library #####
import data.seg_transforms as st
from data.Seg_dataset import SegList
from utils.logger import Logger
from models.net_builder import net_builder
from utils.utils import AverageMeter, zip_dir

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger_vis = logging.getLogger(__name__)
logger_vis.setLevel(logging.DEBUG)
  
###### test ########

def test(args, test_data_loader, model,result_path):
    model.eval()
    batch_time = AverageMeter()
    end = time.time()

    for iter, (image,image_name) in enumerate(test_data_loader):
        # batchsize = 1 ,so squeeze dim 1
        image = image.squeeze()
        image_name = image_name[0]
        
        with torch.no_grad():
            # batch test for memory reduce
            batch = 8
            pred_seg = torch.zeros(image.shape[0],image.shape[2],image.shape[3])
            pred_cls = torch.zeros(image.shape[0],3)
            for i in range(0,image.shape[0],batch):
                start_id = i
                end_id = i + batch
                if end_id > image.shape[0]:
                    end_id = image.shape[0]
                image_batch = image[start_id:end_id,:,:,:]
                image_var = Variable(image_batch).cuda()
                # model forward
                output_seg,output_cls = model(image_var)
                _, pred_batch = torch.max(output_seg, 1)
                pred_seg[start_id:end_id,:,:] = pred_batch.cpu().data
                pred_cls[start_id:end_id,:] = output_cls.cpu().data
            
            pred_seg = pred_seg.numpy().astype('uint8') # predict label
            pred_det = pred_cls.numpy().astype('float32')

            batch_time.update(time.time() - end)
        # save seg result 
        if args.seg:
            save_dir = osp.join(result_path, 'segment')
            if not exists(save_dir):
                os.makedirs(save_dir)
            np.save(osp.join(save_dir,  image_name+'_labelMark_volumes'), pred_seg)
            print('save segment result, finished!')
        # save cls result 
        if args.det:
            save_dir = osp.join(result_path, 'segment')
            if not exists(save_dir):
                os.makedirs(save_dir)
            np.save(osp.join(save_dir,  image_name+'_labelMark_detections'), pred_det)
            print('save detection result, finished!')

        end = time.time()
        logger_vis.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(iter, len(test_data_loader), batch_time=batch_time))


def test_fusion(args, test_data_loader, model_list,result_path):
    for model in model_list:
        model.eval()
    batch_time = AverageMeter()
    end = time.time()

    for iter, (image,image_name) in enumerate(test_data_loader):
        # batchsize = 1 ,so squeeze dim 1
        image = image.squeeze()
        image_name = image_name[0]
        
        with torch.no_grad():
            # batch test for memory reduce
            batch = 8
            pred_seg = torch.zeros(image.shape[0],image.shape[2],image.shape[3])
            pred_cls = torch.zeros(image.shape[0],3)
            for i in range(0,image.shape[0],batch):
                start_id = i
                end_id = i + batch
                if end_id > image.shape[0]:
                    end_id = image.shape[0]
                image_batch = image[start_id:end_id,:,:,:]
                image_var = Variable(image_batch).cuda()

                Output_Seg = Variable(torch.zeros(batch,4,image.shape[2],image.shape[3])).cuda()
                Output_Cls = Variable(torch.zeros(batch,3)).cuda()
                # wangshen model forward
                weight = torch.tensor([0.5,0.5]).cuda()
                for j,model in enumerate(model_list):
                    output_seg,output_cls = model(image_var)
                    Output_Seg += weight[j]*torch.exp(output_seg)
                    Output_Cls += weight[j]*output_cls
              
                _, pred_batch = torch.max(Output_Seg, 1)
                pred_seg[start_id:end_id,:,:] = pred_batch.cpu().data
                pred_cls[start_id:end_id,:] = Output_Cls.cpu().data
            
            pred_seg = pred_seg.numpy().astype('uint8') # predict label
            pred_det = pred_cls.numpy().astype('float32')

            batch_time.update(time.time() - end)
            
        if args.seg:
            save_dir = osp.join(result_path, 'segment')
            if not exists(save_dir):
                os.makedirs(save_dir)
            np.save(osp.join(save_dir,  image_name+'_labelMark_volumes'), pred_seg)
            print('save segment result, finished!')

        if args.det:
            save_dir = osp.join(result_path, 'segment')
            if not exists(save_dir):
                os.makedirs(save_dir)
            np.save(osp.join(save_dir,  image_name+'_labelMark_detections'), pred_det)
            print('save detection result, finished!')

        end = time.time()
        logger_vis.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(iter, len(test_data_loader), batch_time=batch_time))


def test_seg(args,result_path):
    print('Loading test model ...')
    if args.fusion:
        # 1
        net_1 = net_builder('unet_nested')
        net_1 = nn.DataParallel(net_1).cuda()
        checkpoint_1 = torch.load('result/ori_3D/train/unet_nested/checkpoint/model_best.pth.tar')
        net_1.load_state_dict(checkpoint_1['state_dict'])
        # 2
        net_2 = net_builder('unet')
        net_2 = nn.DataParallel(net_2).cuda()
        checkpoint_2 = torch.load('result/ori_3D/train/unet/checkpoint/model_best.pth.tar')
        net_2.load_state_dict(checkpoint_2['state_dict'])

        net = [net_1,net_2]
    else:
        net = net_builder(args.seg_name)
        net = nn.DataParallel(net).cuda()
        checkpoint = torch.load(args.seg_path)
        net.load_state_dict(checkpoint['state_dict'])
        #print('model loaded!')
        
    info = json.load(open(osp.join(args.list_dir,'info_test.json'), 'r'))
    normalize = st.Normalize(mean=info['mean'], std=info['std'])
    
    t = []
    if args.resize:
        t.append(st.Resize(args.resize))
    t.extend([st.ToTensor(),
              normalize])
    dataset = SegList(args.data_dir, 'test', st.Compose(t), list_dir=args.list_dir)
    test_loader = torch.utils.data.DataLoader(
        dataset,batch_size=1, shuffle=False, num_workers=args.workers,pin_memory=False
    )

    cudnn.benchmark = True
    if args.fusion:
        test_fusion(args, test_loader,net,result_path)
    else:
        test(args, test_loader,net,result_path)

def parse_args():
    # Testing settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data-dir', default=None, required=True)
    parser.add_argument('-l', '--list-dir', default=None,
                        help='List dir to look for train_images.txt etc. '
                             'It is the same with --data-dir if not set.')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--seg-name', dest='seg_name',help='seg model',default=None, type=str) 
    parser.add_argument('--det-name', dest='det_name',help='det model',default=None, type=str) 
    parser.add_argument('--seg-path',help='pretrained model test',default='./', type=str) 
    parser.add_argument('--det-path',help='pretrained model test',default='./', type=str) 
    parser.add_argument('--seg',action='store_true')
    parser.add_argument('--det',action='store_true')
    parser.add_argument('--fusion',action='store_true')
    parser.add_argument('--resize', default=0, type=int)
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    task_name = args.list_dir.split('/')[-1]
    ##### logger setting #####
    result_path = osp.join('result',task_name,'test', 'final_CHUNXIN')
    #result_path = osp.join('result',data_name,'comp', args.name)
    if not exists(result_path):
        os.makedirs(result_path)
    test_seg(args,result_path)
    # zip submission
    print('Submission zip generating ... ')
    zip_dir(osp.join(result_path,'segment'),osp.join(result_path,'submission.zip'))
    print('Submission zip generated ^_^ ')
      
if __name__ == '__main__':
    main()
