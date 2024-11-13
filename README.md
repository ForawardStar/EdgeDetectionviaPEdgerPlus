# PEdger++

This is the implementation of our paper ``PEdger++: Practical Edge Detection via Assembling Cross Information"

# Brief Introduction
Edge detection, serving as a crucial component in numerous vision-based applications, aims to effectively extract object boundaries and/or salient edges from natural images. To be viable for broad deployment across devices with varying computational capacities, edge detectors shall balance high accuracy with low computational complexity. This paper addresses the challenge of achieving that balance: {how to efficiently capture discriminative features without relying on large-size and sophisticated models}. We propose PEdger++, a collaborative learning framework designed to reduce computational costs and model sizes while improving edge detection accuracy. The core principle of our PEdger++ is that cross-information derived from  heterogeneous  architectures, diverse training moments, and multiple parameter samplings, is beneficial to enhance learning from an ensemble perspective. Extensive ablation studies together with experimental comparisons on the BSDS500, NYUD and Multicue datasets demonstrate the effectiveness of our approach, both quantitatively and qualitatively, showing clear improvements over existing methods.  We also provide multiple versions of the model with varying computational requirements, highlighting PEdger++'s adaptability with respect to different resource constraints.


# Environment Installation
Our model is based on Pytorch, which can be installed following the instructions of the official website: https://pytorch.org/

Or, you can install the packages using requirement.txt, through running:
```pip install -r requirement.txt```


# Preparing Data
Download the augmented BSDS and PASCAL VOC datasets from:

http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz

http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz

Download the augmented NYUD dataset from:

https://pan.baidu.com/s/1J5z6235tv1xef3HXTaqnKg Extraction Code:t2ce

After downloading the datasets, put them in 'data/' folder.

# Training
Before starting the training process, the training and validation set should be split through runing:

```python splittrainval.py```

If you want to train our PEdger++, change the data path of training images, and then run:

```python main.py```

# Testing
If you want to test our pre-trained model, put the test images into the 'data/test/' folder, and then run:

```python test_edge.py```

Our pre-trained models of PEdger++ without any pre-training, have already been stored in the "models/" folder, i.e., models/checkpoint.pth, and the relative path to the 'models/checkpoint.pth' is specified in the "test_edge.py" file. Simply running 'test_edge.py' can take the images in "data/test/" folder as inputs, and output the detected edges.

If you want to evaluate the performance of PEdger++ w/ VGG16 and PEdger++ w/ ResNet50, please run:


```python test_edge_VGG16.py```


```python test_edge_ResNet50.py```

Before running 'test_edge_VGG16.py' and 'test_edge_ResNet50.py', the pre-trained models should be downloaded from the link: 
 https://pan.baidu.com/s/129W63l5nyYMZnE_8-0C9lA?pwd=dcfv Extraction Code: dcfv 



Our pre-computed edge maps are available at this link: https://pan.baidu.com/s/1TgyJ84oqJVASpA59jhbTeQ?pwd=1ia4

# Evaluation
The matlab code for evaluation can be downloaded in https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html. Before evaluation, the non-maximum suppression should be done through running ``edge_nms.m" in https://github.com/yun-liu/RCF.

# Related Projects
MuGE (CVPR 2024) & UAED (CVPR 2023): https://github.com/ZhouCX117/UAED_MuGE

DiffusionEdge (AAAI 2024): https://github.com/GuHuangAI/DiffusionEdge

EDTER (CVPR 2022): https://github.com/mengyangpu/edter

PEdger (ACM MM 2023): https://github.com/ForawardStar/PEdger

PiDiNet (ICCV 2021 & TPAMI 2023): https://github.com/hellozhuo/pidinet

BDCN (CVPR 2019): https://github.com/pkuCactus/BDCN

RCF (CVPR 2017): https://mmcheng.net/rcfEdg

HED (ICCV 2015): https://github.com/s9xie/hed 
