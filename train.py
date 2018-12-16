#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:ShawnWang
##### System library #####
import os
import os.path as osp
from os.path import exists
import argparse
import json
import logging
import time
import numpy as np
import shutil
##### pytorch library #####
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
##### My own library #####
import data.seg_transforms as dt
from data.Seg_dataset import SegList
from utils.logger import Logger
from models.net_builder import net_builder
from utils.loss import loss_builder
from utils.utils import compute_average_dice,AverageMeter, save_checkpoint,count_param,target_seg2target_cls
from utils.utils import aic_fundus_lesion_segmentation,aic_fundus_lesion_classification,compute_segment_score,compute_single_segment_score
# logger vis
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger_vis = logging.getLogger(__name__)
logger_vis.setLevel(logging.DEBUG)

###### train #######

def adjust_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(args,train_loader, model, criterion, optimizer, epoch,print_freq=10):
   # set the AverageMeter 
    batch_time = AverageMeter()
    losses = AverageMeter()
    dice = AverageMeter()
    Dice_1 = AverageMeter()
    Dice_2 = AverageMeter()
    Dice_3 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    correct,total  = 0,0
    for i, (input, target) in enumerate(train_loader):
        if i > 20:
            break
        # transfer seg target to cls target
        target_seg = target.numpy()
        target_cls = target_seg2target_cls(target_seg).cuda()
        # Variable
        input_var = Variable(input).cuda()
        target_var_seg = Variable(target).cuda()
        target_var_cls = Variable(target_cls).cuda()
        # forward
        output_seg,output_cls = model(input_var)
        # loss 
        loss_1 = criterion[0](output_seg, target_var_seg)
        loss_2 = criterion[1](output_seg, target_var_seg)
        loss_3 = criterion[2](output_cls,target_var_cls)
        loss = loss_1 + loss_2 + loss_3
        losses.update(loss.data, input.size(0))
        # metric dice for seg
        _, pred_seg = torch.max(output_seg, 1)
        pred_seg = pred_seg.cpu().data.numpy()
        label_seg = target_var_seg.cpu().data.numpy()
        dice_score,dice_1,dice_2,dice_3 = compute_average_dice(pred_seg.flatten(),label_seg.flatten())
        # metric acc for cls
        pred_cls = (output_cls > 0.5)
        label_cls = target_var_cls.cpu().data.numpy()
        total += target_var_cls.size(0)*3
        correct += pred_cls.eq(target_var_cls.byte()).sum().item()
        # update dice
        dice.update(dice_score)
        Dice_1.update(dice_1)
        Dice_2.update(dice_2)
        Dice_3.update(dice_3)
        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # logger vis
        if i % print_freq == 0:
            logger_vis.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        #'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                        'Dice {dice.val:.4f} ({dice.avg:.4f})\t'
                        'Dice_1 {dice_1.val:.6f} ({dice_1.avg:.4f})\t'
                        'Dice_2 {dice_2.val:.6f} ({dice_2.avg:.4f})\t'
                        'Dice_3 {dice_3.val:.6f} ({dice_3.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,dice = dice,dice_1=Dice_1,dice_2=Dice_2,dice_3=Dice_3))
            print('loss :',loss.cpu().data.numpy())
            print('Acc: %.3f%% (%d/%d)'% (100.*correct/total, correct, total))

    return losses.avg,dice.avg,Dice_1.avg,Dice_2.avg,Dice_3.avg

def train_seg(args,result_path,logger):

    for k, v in args.__dict__.items():
        print(k, ':', v)
    # load the net
    net = net_builder(args.name,args.model_path,args.pretrained)
    model = torch.nn.DataParallel(net).cuda()
    param = count_param(model)
    print('###################################')
    print('Model #%s# parameters: %.2f M' % (args.name,param/1e6))
    # set the loss criterion
    criterion = loss_builder(args.loss)
    # Data loading code
    info = json.load(open(osp.join(args.list_dir, 'info.json'), 'r'))
    normalize = dt.Normalize(mean=info['mean'],std=info['std'])
    # data transforms
    t = []
    if args.resize:
        t.append(dt.Resize(args.resize))
    if args.random_rotate > 0:
        t.append(dt.RandomRotate(args.random_rotate))
    if args.random_scale > 0:
        t.append(dt.RandomScale(args.random_scale))
    if args.crop_size:
        t.append(dt.RandomCrop(args.crop_size))
    t.extend([dt.Label_Transform(),
              dt.RandomHorizontalFlip(),
              dt.ToTensor(),
              normalize])
    train_loader = torch.utils.data.DataLoader(
        SegList(args.data_dir, 'train', dt.Compose(t),list_dir=args.list_dir),batch_size=args.batch_size, 
        shuffle=True, num_workers=args.workers,pin_memory=True, drop_last=True)

    # define loss function (criterion) and pptimizer
    if args.optimizer == 'SGD':    #SGD optimizer
        optimizer = torch.optim.SGD(net.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam': #Adam optimizer
        optimizer = torch.optim.Adam(net.parameters(),
                                    args.lr,
                                    betas=(0.9, 0.99),
                                    weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_dice = 0
    start_epoch = 0

    # load the pretrained model
    if args.model_path:
        print("=> loading pretrained model '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_dice = checkpoint['best_dice']
            dice_epoch = checkpoint['dice_epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # main training
    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args,optimizer, epoch)
        logger_vis.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        loss,dice_train,dice_1,dice_2,dice_3 = train(args,train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        dice_val,dice_11,dice_22,dice_33,dice_list,auc,auc_1,auc_2,auc_3 = val_seg(args,model)
        # save best checkpoints
        is_best = dice_val > best_dice
        best_dice = max(dice_val, best_dice)
        checkpoint_dir = osp.join(result_path,'checkpoint')
        if not exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_latest = checkpoint_dir+'/checkpoint_latest.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'dice_epoch':dice_val,
            'best_dice': best_dice,
        }, is_best, checkpoint_dir,filename=checkpoint_latest)
        if args.save_every_checkpoint:
            if (epoch + 1) % 1 == 0:
                history_path = checkpoint_dir+'/checkpoint_{:03d}.pth.tar'.format(epoch + 1)
                shutil.copyfile(checkpoint_latest, history_path)

        logger.append([epoch,dice_train,dice_val,auc,dice_1,dice_11,dice_2,dice_22,dice_3,dice_33,auc_1,auc_2,auc_3])


####### validation ###########

def val(args,eval_data_loader, model):
    model.eval()
    batch_time = AverageMeter()
    dice = AverageMeter()
    end = time.time()
    dice_list = []
    Dice_1 = AverageMeter()
    Dice_2 = AverageMeter()
    Dice_3 = AverageMeter()
    ret_segmentation = []
    
    for iter, (image, label, _) in enumerate(eval_data_loader):
        # batchsize = 1 ,so squeeze dim 1
        image = image.squeeze(dim=0)
        label = label.squeeze(dim=0)

        target_seg = label.numpy()
        target_cls = target_seg2target_cls(target_seg)
        
        with torch.no_grad():
            # batch test for memory reduce
            batch = 16
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
            # merice dice for seg
            pred_seg = pred_seg.numpy().astype('uint8') 
            batch_time.update(time.time() - end)
            label_seg = label.numpy().astype('uint8')
            ret = aic_fundus_lesion_segmentation(label_seg,pred_seg)
            ret_segmentation.append(ret)
            dice_score = compute_single_segment_score(ret)
            dice_list.append(dice_score)
            dice.update(dice_score)
            Dice_1.update(ret[1])
            Dice_2.update(ret[2])
            Dice_3.update(ret[3])
            # metric auc for cls
            ground_truth = target_cls.numpy().astype('float32')
            prediction = pred_cls.numpy().astype('float32') 
            if iter == 0:
                detection_ref_all = ground_truth
                detection_pre_all = prediction
            else:
                detection_ref_all = np.concatenate((detection_ref_all, ground_truth), axis=0)
                detection_pre_all = np.concatenate((detection_pre_all, prediction), axis=0)
            
        end = time.time()
        logger_vis.info('Eval: [{0}/{1}]\t'
                    'Dice {dice.val:.3f} ({dice.avg:.3f})\t'
                    'Dice_1 {dice_1.val:.3f} ({dice_1.avg:.3f})\t'
                    'Dice_2 {dice_2.val:.3f} ({dice_2.avg:.3f})\t'
                    'Dice_3 {dice_3.val:.3f} ({dice_3.avg:.3f})\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), dice = dice,dice_1 = Dice_1,dice_2 = Dice_2,dice_3 = Dice_3,batch_time=batch_time))

    # compute average dice for seg
    final_seg,seg_1,seg_2,seg_3 = compute_segment_score(ret_segmentation)
    print('### Seg ###')
    print('Final Seg Score:{}'.format(final_seg))
    print('Final Seg_1 Score:{}'.format(seg_1))
    print('Final Seg_2 Score:{}'.format(seg_2))
    print('Final Seg_3 Score:{}'.format(seg_3))
    # compute average auc for cls
    ret_detection = aic_fundus_lesion_classification(detection_ref_all, detection_pre_all, num_samples=len(eval_data_loader)*128)
    auc = np.array(ret_detection).mean()
    print('AUC :',auc)
    auc_1 = ret_detection[0]
    auc_2 = ret_detection[1]
    auc_3 = ret_detection[2]

    return final_seg,seg_1,seg_2,seg_3,dice_list,auc,auc_1,auc_2,auc_3

def val_seg(args,model):
    info = json.load(open(osp.join(args.list_dir, 'info.json'), 'r'))
    normalize = dt.Normalize(mean=info['mean'], std=info['std'])

    t = []
    if args.resize:
        t.append(dt.Resize(args.resize))
    if args.crop_size:
        t.append(dt.RandomCrop(args.crop_size)) 
    t.extend([dt.Label_Transform(),
              dt.ToTensor(),
              normalize])

    dataset = SegList(args.data_dir, 'val', dt.Compose(t),list_dir=args.list_dir)
    val_loader = torch.utils.data.DataLoader(dataset,batch_size=1, shuffle=False, 
        num_workers=args.workers,pin_memory=False)

    cudnn.benchmark = True
    dice_avg,dice_1,dice_2,dice_3,dice_list,auc,auc_1,auc_2,auc_3 = val(args,val_loader, model)

    return dice_avg,dice_1,dice_2,dice_3,dice_list,auc,auc_1,auc_2,auc_3


def parse_args():
    parser = argparse.ArgumentParser(description='train')
    # config
    parser.add_argument('-d', '--data-dir', default=None, required=True)
    parser.add_argument('-l', '--list-dir', default=None,
                        help='List dir to look for train_images.txt etc. '
                             'It is the same with --data-dir if not set.')
    parser.add_argument('--name', dest='name',help='change model',default=None, type=str) 
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-j', '--workers', type=int, default=8)
    # Train Setting
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--loss',help='change model',default='wce', type=str) 
    parser.add_argument('-o', '--optimizer',default='Adam', type=str) 
    # Data Transform
    parser.add_argument('--random-rotate', default=0, type=int)
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--resize', default=0, type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    # Pretrain and Checkpoint
    parser.add_argument('-p','--pretrained', type=bool)
    parser.add_argument('--model-path', default=None,type=str)
    parser.add_argument('--save-every-checkpoint', action='store_true')

    args = parser.parse_args()

    return args

def main():
    ##### config #####
    args = parse_args()
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print('torch version:',torch.__version__)
    ##### logger setting #####
    task_name = args.list_dir.split('/')[-1]
    pretrained = 'pre' if args.pretrained else 'nopre'
    result_path = osp.join('result',task_name,'train',args.name + '_' + pretrained + '_' + args.loss + '_' +str(args.lr) + '_github')
    if not exists(result_path):
        os.makedirs(result_path)
    resume = True if args.resume else False
    logger = Logger(osp.join(result_path,'dice_epoch.txt'), title='dice',resume=resume)
    #if not resume:
    logger.set_names(['Epoch','Dice_Train','Dice_Val','AUC','Dice_1','Dice_11','Dice_2','Dice_22','Dice_3','Dice_33','AUC_1','AUC_2','AUC_3',])
    # train_seg
    train_seg(args,result_path,logger)
      
if __name__ == '__main__':
    main()
