# RPVNet

## Introdution
This is an **Non-official** implement of **RPVNet:A Deep and Efficient Range-Point-Voxel Fusion Network for LiDAR
Point Cloud Segmentation**. It is a fusion network based on three kinds of representation including Range image，Point Clouds and Voxels. For each Range or Voxel branch, it consist of a stem, four upblocks and four downblock. And point branch only uses four simple mlps bringing great efficiency and ability to extract fine-grained geometric features. What's more, Gated Fusion Module(GFM) was designed to adaptively measurse the importance of feature for each branch.

<div align="center">
    <a href="https://github.com/GuoPingPan/RPVNet"><img width="500px" height="auto" src="demo/rpvnet.png"></a>
</div>

</br>

> More details for this job for searching this [paper](https://arxiv.org/abs/2103.12978).
>
> Chinese comments Version: https://github.com/GuoPingPan/RPVNet/tree/chinese

</br>

## Dependencies
- python 3
- pyyaml
- argparse
- torch
- torchvision
- tqdm
- numpy
- torchsparse

> torchsparse can be install by flowing [this](https://github.com/mit-han-lab/torchsparse)

</br>

## Data Preparation
SemanticKITTI

Please follow the instructions from [here](http://www.semantic-kitti.org/) to download the SemanticKITTI dataset (both KITTI Odometry dataset and SemanticKITTI labels) and extract all the files in the sequences folder to /dataset/semantic-kitti. You shall see 22 folders 00, 01, …, 21; each with subfolders named velodyne and labels.
```
dataset
 - sequences
  - 00
    - velodyne
      - 000000.bin
      - 000001.bin
      - ...
    - labels
      - 000000.label
      - 000001.label
      - ...
      
    - poses.txt
      
```

</br>

## Quick Start

1. ensure the dependencies
2. train
```
python train.py -d dataset/sequences  --log <path to save model> [--ckpt <path to pretrained model> --freeze_layers --device <cpu or cuda>]
```
3. test
```
python inference.py -d dataset/sequences  -ckpt <path to model> [--device <cpu or cuda>]
```

</br>

## Reference
- paper:  [RPVNet: A Deep and Efficient Range-Point-Voxel Fusion Network for LiDAR Point Cloud Segmentation](https://arxiv.org/abs/2103.12978)
- spvnas:  https://github.com/mit-han-lab/spvnas
- SalsaNext:  https://github.com/TiagoCortinhal/SalsaNext
- kprnet:  https://github.com/DeyvidKochanov-TomTom/kprnet
