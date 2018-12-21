# AI-Challenger-Retinal-Edema-Segmentation
> Our team ranked fourth in objective results and ranked first in subjective results. Finally, we got the fourth place in this challenge.


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

To evaluate the model and save the visualization result:

To test the model and generate the result zip file:


## 4. Results

## 5. Main Technique Analysis

## 6. Future Work

## 7. To do
- [x] Add Presentation PPT
- [ ] Add Dataset source
- [ ] Add Visualization demo
- [ ] Add Usage
- [ ] Add Main technique analysis
- [ ] Add Results
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
  
