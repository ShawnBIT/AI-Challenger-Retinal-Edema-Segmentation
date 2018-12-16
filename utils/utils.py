import numpy as np
import torch
import os
import os.path as osp
import cv2
import scipy.misc as misc
import shutil
from skimage import measure
import math
import traceback
from sklearn import metrics
import zipfile


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best,checkpoint_path,filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(checkpoint_path,'model_best.pth.tar'))

def save_dice_single(is_best, filename='dice_single.txt'):
    if is_best:
        shutil.copyfile(filename, 'dice_best.txt')


def compute_dice_score(predict, gt, forground = 1):
    score = 0
    count = 0
    assert(predict.shape == gt.shape)
    overlap = 2.0 * ((predict == forground)*(gt == forground)).sum()
    #print('overlap:',overlap)
    
    return (overlap + 0.001) / (((predict == forground).sum() + (gt == forground).sum()) + 0.001)
  

def compute_average_dice(predict, gt, class_num = 4):
    Dice = 0
    Dice_list = []

    for i in range(1,class_num):
        predict_copy = predict.copy()
        gt_copy = gt.copy()
        predict_copy[predict_copy != i] = 0
        gt_copy[gt_copy != i] = 0
        dice = compute_dice_score(predict_copy, gt_copy, forground = i)
        Dice += dice
        Dice_list.append(dice)
    return Dice/(class_num - 1),Dice_list[0],Dice_list[1],Dice_list[2]



def compute_score(predict, gt, forground = 1):
    score = 0
    count = 0
    assert(predict.shape == gt.shape)
    overlap = ((predict == forground)*(gt == forground)).sum()
    #print('overlap:',overlap)
    if(overlap > 0):
        return 2*overlap / ((predict == forground).sum() + (gt == forground).sum()),overlap /  (predict == forground).sum(), overlap / (gt == forground).sum(),overlap / ((predict == forground).sum() + (gt == forground).sum() - overlap)
        # dice,precsion,recall
    else:
        return 0,0,0,0

def eval_seg(predict, gt, forground = 1):
    assert(predict.shape == gt.shape)
    Dice = 0
    Precsion = 0
    Recall = 0
    Jaccard = 0
    n = predict.shape[0]
    for i in range(n):
        dice,precsion,recall,jaccard = compute_score(predict[i],gt[i])
        Dice += dice
        Precsion += precsion
        Recall += recall
        Jaccard += jaccard

    return Dice/n,Precsion/n,Recall/n,Jaccard/n

def aic_fundus_lesion_classification(ground_truth, prediction, num_samples=128):
    """
    Classification task auc metrics.
    :param ground_truth: numpy matrix, (num_samples, 3)
    :param prediction: numpy matrix, (num_samples, 3)
    :param num_samples: int, default 128
    :return list:[AUC_1, AUC_2, AUC_3]
    """
    # assert (ground_truth.shape == (num_samples, 3))
    # assert (prediction.shape == (num_samples, 3))

    try:
        ret = [0.5, 0.5, 0.5]
        for i in range(3):
            fpr, tpr, thresholds = metrics.roc_curve(ground_truth[:, i], prediction[:, i], pos_label=1)
            ret[i] = metrics.auc(fpr, tpr)

        
        # fpr, tpr, thresholds = metrics.roc_curve(ground_truth[:,0], prediction[:,0], pos_label=1)
        # ret = metrics.auc(fpr, tpr)
    except Exception as e:
        traceback.print_exc()
        print("ERROR msg:", e)
        return None
    return ret

def aic_fundus_lesion_segmentation(ground_truth, prediction, num_samples=128):
    """
    Detection task auc metrics.
    :param ground_truth: numpy matrix, (num_samples, 1024, 512)
    :param prediction: numpy matrix, (num_samples, 1024, 512)
    :param num_samples: int, default 128
    :return list:[Dice_0, Dice_1, Dice_2, Dice_3]
    """
    #assert (ground_truth.shape == (num_samples, 1024, 512))
    #assert (prediction.shape == (num_samples, 1024, 512))

    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    try:
        ret = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            mask1 = (ground_truth == i)
            mask2 = (prediction == i)
            if mask1.sum() != 0:
                ret[i] = 2 * ((mask1 * (ground_truth == prediction)).sum()) / (mask1.sum() + mask2.sum())
            else:
                ret[i] = float('nan')
    except Exception as e:
        traceback.print_exc()
        print("ERROR msg:", e)
        return None
    return ret

def compute_segment_score(ret_segmentation,cubes=15):
    REA_segementation, SRF_segementation, PED_segementation = 0.0, 0.0, 0.0
    n1, n2, n3 = 0, 0, 0
    for i in range(cubes):
        if not math.isnan(ret_segmentation[i][1]):
            REA_segementation += ret_segmentation[i][1]
            n1 += 1
        if not math.isnan(ret_segmentation[i][2]):
            SRF_segementation += ret_segmentation[i][2]
            n2 += 1
        if not math.isnan(ret_segmentation[i][3]):
            PED_segementation += ret_segmentation[i][3]
            n3 += 1

    REA_segementation /= n1
    SRF_segementation /= n2
    PED_segementation /= n3
    avg_segmentation = (REA_segementation + SRF_segementation + PED_segementation) / 3

    return avg_segmentation,REA_segementation,SRF_segementation,PED_segementation

def compute_single_segment_score(ret_segmentation):
    REA_segementation, SRF_segementation, PED_segementation = 0.0, 0.0, 0.0
    n1, n2, n3 = 0, 0, 0
    
    if not math.isnan(ret_segmentation[1]):
        REA_segementation += ret_segmentation[1]
        n1 += 1
    if not math.isnan(ret_segmentation[2]):
        SRF_segementation += ret_segmentation[2]
        n2 += 1
    if not math.isnan(ret_segmentation[3]):
        PED_segementation += ret_segmentation[3]
        n3 += 1

    avg_segmentation = (REA_segementation + SRF_segementation + PED_segementation) / (n1+n2+n3)

    return avg_segmentation


def rebuild_tensor_v2():
    try:
        torch._utils._rebuild_tensor_v2
    except AttributeError:
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor
        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def target_seg2target_cls(array):
    class_num_all = torch.zeros(array.shape[0],3)
    for i in range(array.shape[0]):
        array_np = np.unique(array[i])
        label = array_np[1:] + 1
        if label.sum() == 0:
            class_num = torch.tensor([0,0,0])
        elif label.sum() == 2:#np.array([0,255]):
            class_num = torch.tensor([1,0,0])
        elif label.sum() == 3:#np.array([0,191]):
            class_num = torch.tensor([0,1,0])
        elif label.sum() == 4:#np.array([0,128]):
            class_num = torch.tensor([0,0,1])
        elif label.sum() == 5: #np.array([0,255,191]):
            class_num = torch.tensor([1,1,0])
        elif label.sum() == 6:#np.array([0,255,128]):
            class_num = torch.tensor([1,0,1])
        elif label.sum() == 7:#np.array([0,191,128]):
            class_num = torch.tensor([0,1,1])
        elif label.sum() == 9:#np.array([0,255,191,128]):
            class_num = torch.tensor([1,1,1])
        class_num_all[i,:] = class_num

    return class_num_all.float()

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def zip_dir(dirname,zipfilename):
    filelist = []
    if osp.isfile(dirname):
        filelist.append(dirname)
    else :
        for root, dirs, files in os.walk(dirname):
            for name in files:
                filelist.append(osp.join(root, name))
        
    zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
    for tar in filelist:
        arcname = tar[len(dirname):]
        #print arcname
        zf.write(tar,arcname)
    zf.close()















