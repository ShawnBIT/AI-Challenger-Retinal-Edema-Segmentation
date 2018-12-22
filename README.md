# AI-Challenger-Retinal-Edema-Segmentation
> Our team ranked fourth in objective results and ranked first in subjective results. Finally, we got the fourth place in this challenge.

We build an end-to-end **multi-task framework** that can simultaneously detect and segment retinal edema lesions. We use the latest **UNet++** model to better integrate high-level and low-level features for segmentation and add a classification head at the highest level feature map for detection. For two types of  small lesions, we use a novel **exponential logarithmic loss** function to enhance the segmentation performance. Meanwhile, we introduce the **dilated convolution module**, which significantly increases the receptive field of the model and improves the segmentation performance of big lesions. More importantly, only random horizontal flip data augmentation is needed and no need for post-processing.

Finally, the dice of single model on the test set is **0.736**. The dice of fusion model on the test set is **0.744** and the detection AUC is **0.986**. The memory of inference stage is **7.3G(TITAN Xp)** when we set batch is 8 and the inference time is **9.5s** per patient.


  |Original Image|Ground Truth| UNet| UNet++| Dialted UNet++|UNet&UNet++ Fusion| 
  |---|---|---|---|---|---
<p align="center">
  <img src="https://github.com/ShawnBIT/AI-Challenger-Retinal-Edema-Segmentation/blob/master/materials/figures/0010.gif" width="800"/>  
  <img src="https://github.com/ShawnBIT/AI-Challenger-Retinal-Edema-Segmentation/blob/master/materials/figures/0014.gif" width="800"/>
</p>



## 1. Getting Started

Clone the repo:

  ```bash
  git clone https://github.com/ShawnBIT/AI-challenger-Retinal-Edema-Segmentation.git
  ```

#### Requirements
 ```
python>=3.6
torch>=0.4.0
torchvision
argparse
numpy
pillow
scipy
scikit-image
sklearn
 ```
 Install all dependent libraries:
  ```bash
  pip install -r requirements.txt
  ```
 
## 2. Data Prepare 

#### Data Download

#### Data Pre-process


## 3. Usage
To train the model:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train.py --name unet_nested -d ./data/dataset/ -l ./data/data_path --batch-size 16 -j 16 --epochs 100 -o Adam --lr 0.001 --lr-mode poly --momentum 0.9 --loss mix_33
  
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train.py --name unet  -d ./data/dataset/ -l ./data/data_path --batch-size 32 -j 32 --epochs 100 -o Adam --lr 0.001 --step 20 --momentum 0.9 --loss mix_3
  ```
To evaluate a single model:
```bash
CUDA_VISIBLE_DEVICES=2 python3 eval.py -d ./data/dataset/ -l ./data/data_path -j 32 --vis --seg-name unet_nested --seg-path result/ori_3D/train/unet_nested_nopre_mix_33_NEW_multi_2_another/checkpoint/model_best.pth.tar
  ```
To evaluate the fusion model:
```bash
CUDA_VISIBLE_DEVICES=2 python3 eval.py -d ./data/dataset/ -l ./data/data_path -j 32 --vis --fusion 
  ```
To test a single model:
```bash
CUDA_VISIBLE_DEVICES=2 python3 test.py -d ./data/dataset/ -l ./data/data_path -j 32 --seg --det --seg-name unet_nested --seg-path result/ori_3D/train/unet_nested_nopre_mix_33_NEW_multi_2_another/checkpoint/model_best.pth.tar
  ```
To test the fusion model:
```bash
CUDA_VISIBLE_DEVICES=2 python3 test.py -d ./data/dataset/ -l ./data/data_path -j 32 --seg --det --fusion
  ```
  
## 4. Results

| Model |Multi-Task| Params| Loss| Val_Dice| Val_Auc| Test_Dice| Test_Auc|Checkpoint|
|---|---|---|---|---|---|---|---|---
| ResNet18(*pre)  |No| 11.18M | BCE| -    | 0.971| -| 0.904| link |
| UNet            |No| 2.47M  | WCE+Dice| 0.772| -| 0.683| -|link|
| UNet            |Yes| 2.47M | WCE+Dice+BCE| 0.785(+1.3%)|0.985(+1.4%) |0.701(+1.8%) |- | link|
| UNet++          |Yes| 2.95M | WCE+Dice+BCE| 0.784(+1.2%)|0.986(+1/5%)| -| -| link|
| UNet++          |Yes| 2.95M | WCE+ELDice+BCE|0.799(+2.7%)|0.989(+1.8%)|0.736(+5.3%) | -|link|
| Dialted UNet++  |Yes| 5.32M | WCE+ELDice+BCE|**0.807(+3.5%)**|0.978(+0.6%) | | |link|
| Fusion(*)      |-|- | -|0.805(+3.3%)|**0.991(2%)**|**0.744(6.1%)**|**0.986(+8.2%)**|link|


## 5. Main Technique Analysis

## 6. Future Work

## 7. To do
- [x] Add Presentation PPT
- [ ] Add Dataset source
- [ ] Add Visualization demo
- [x] Add Usage
- [ ] Add Main technique analysis
- [x] Add Results
- [ ] Add Future work
- [x] Add Reference
- [ ] Add Personal experience

## 8. Acknowledge
 * GPU support of [DeepWise](http://www.deepwise.com/) 
 * Mentor Prof. [Yizhou Wang](http://www.idm.pku.edu.cn/staff/wangyizhou/)'s guidence
 * The host，[AI challenger](https://challenger.ai/) platform

## 9. Reference
#### Paper
 * [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/pdf/1807.10165.pdf) (2018 MICCAI)
 * [3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes](https://arxiv.org/pdf/1809.00076.pdf) (2018 MICCAI oral)
 * [D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf) (2018 CVPR workshop)
 * [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf) (2015 MICCAI)

#### Code
 * https://github.com/fyu/drn (Our framework style mianly refered to this repository.)
 * https://github.com/ozan-oktay/Attention-Gated-Networks (Our model style mianly refered to this repository.）
  
