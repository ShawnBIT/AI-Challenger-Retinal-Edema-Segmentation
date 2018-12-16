#!/bin/bash

# train
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train.py --name unet_nested -d ./data/dataset/ -l ./data/data_path --batch-size 16 -j 16 --epochs 100 -o Adam --lr 0.001 --lr-mode poly --momentum 0.9 --loss mix_33
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train.py --name unet  -d ./data/dataset/ -l ./data/data_path --batch-size 32 -j 32 --epochs 100 -o Adam --lr 0.001 --step 20 --momentum 0.9 --loss mix_3

# eval
# CUDA_VISIBLE_DEVICES=2 python3 eval.py -d ./data/dataset/ -l ./data/data_path -j 32 --vis --fusion 
# CUDA_VISIBLE_DEVICES=2 python3 eval.py -d ./data/dataset/ -l ./data/data_path -j 32 --vis --seg-name unet_nested --seg-path result/ori_3D/train/unet_nested_nopre_mix_33_NEW_multi_2_another/checkpoint/model_best.pth.tar


# test
# CUDA_VISIBLE_DEVICES=2 python3 test.py -d ./data/dataset/ -l ./data/data_path -j 32 --seg --det --fusion
# CUDA_VISIBLE_DEVICES=2 python3 test.py -d ./data/dataset/ -l ./data/data_path -j 32 --seg --det --seg-name unet_nested --seg-path result/ori_3D/train/unet_nested_nopre_mix_33_NEW_multi_2_another/checkpoint/model_best.pth.tar

