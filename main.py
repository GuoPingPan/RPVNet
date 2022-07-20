'''
Network Setup. As shown in Figure 3, both
range and voxel branch are Unet-like architecture with
a stem, four down-sampling and four up-sampling
stages, and the dimensions of these 9 stages are
32; 64; 128; 256; 256; 128; 128; 64; 32, respectively. For
range branch, the input range-image size is 64 × 2048 on
SemanticKITTI dataset, and 32 × 2048 as initial size for
nuScenes dataset, then resized to 64×2048 to keep the same
as SemanticKITTI. As for voxel branch, the voxel resolution is 0:05m for the experiments in Sec. 4.2 and Sec. 4.3.
For point branch, it consists of four per-point MLPs with
dimensions: 32; 256; 128; 32.

Training and Inference Details. We employ the commonly used cross-entropy as loss in training. Our RPVNet
is trained from scratch in an end-to-end manner with the
ADAM or SGD optimizer. For the SemanticKITTI dataset,
the model uploaded to the leaderboard was trained with
SGD, batch size 12, learning rate 0.24 for 60 epochs on
2 GPUs, which is kept the same as SPVCNN [30] for
fair comparison. This setup takes around 100 hours on 2
Tesla V100 GPUs. For the other experiments, including
nuScenes dataset and ablation studies, we core the entire
network with ADAM, batch size 40, learning rate 0.003 for
80 epochs on 8 Tesla V100 GPUs. The cosine annealing
learning rate strategy is adopted for the learning rate decay.
During training, we utilize the widely used data augmentation strategy of segmentation, including global scaling with a random scaling factor sampled from [0:95; 1:05],
and global rotation around the Z axis with a random angle. We also conduct the proposed instance cut-mixup sampling strategy to fine-tune the network in the last 10 training
epoch. To note that, in the voxelization process, we set the
max number of voxels to 84000 for training, and all voxels
for inference.
'''

'''
We conduct
extensive control experiments, including single view(R, P,
V), fusion of two views(RP, PV), and fusion of all three
views(RPV). To note that, only 1/4 training data was used
in order to speed up the pace of training, and the mIoU was
reported on SemanticKITTI validation set(sequence 08) and
nuScenes validation set, moreover Macs and latency were
tested on SemnaticKITTI dataset.
'''
