import time

import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributed
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import DataLoader

from core.dataset.semantic_kitti import SemanticKITTIInternal
from core.models.rpvnet import RPVnet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

class Trainer():
    def __init__(self,dataset_dir,model_cfg,data_cfg,log_dir,pretrained_dir):

        self.batch_size = model_cfg['train']['batch_size']
        self.gpus = model_cfg['train']['batch_size']
        self.lr = model_cfg['train']['lr']
        self.loss = model_cfg['train']['loss']
        self.optimizer = model_cfg['train']['optimizer']
        self.epochs = model_cfg['train']['epochs']
        self.mode = ['train','val','test']

        self.model = RPVnet(
            vsize=model_cfg['voxel_size'],
            cr=model_cfg['cr'],
            cs=model_cfg['dimension_of_stages'],
            num_classes = model_cfg['num_classes']
        )


        data = SemanticKITTIInternal(
            root=dataset_dir,
            voxel_size=data_cfg['voxel_size'],
            range_size=data_cfg['range_size'],
            sample_stride=data_cfg['sample_stride'],
            split=data_cfg['split']['train'],
            max_voxels = data_cfg['max_voxels'],
            label_name_mapping = data_cfg['label_name_mapping'],
            kept_labels = data_cfg['kept_labels']
        )


        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.fastest = True
            if torch.cuda.device_count() == 1:
                # data.device = 'cuda'
                self.gpus = 0
                self.dataloader = DataLoader(
                    data,
                    batch_size=self.batch_size,
                    collate_fn=data.collate_fn
                )
                # self.model.cuda()

            elif torch.cuda.device_count() > 1:
                torch.distributed.init_process_group(backend='nccl',world_size=self.gpus)
                local_rank = torch.distributed.get_rank() # gpu index
                torch.cuda.set_device(local_rank)
                sampler = DistributedSampler(
                    data,
                    rank=local_rank,
                    shuffle= True
                )
                self.dataloader = DataLoader(
                    data,
                    batch_size= self.batch_size,
                    sampler = sampler,
                    collate_fn=data.collate_fn
                )
                self.model = torch.nn.parallel.DistributedDataParallel(self.model,device_ids=local_rank)

        else:
            self.gpus = None
            self.dataloader = DataLoader(
                data,
                batch_size=self.batch_size,
                collate_fn=data.collate_fn
            )
            self.model = self.model

        self.criterion = getattr(nn,model_cfg['train']['loss'])(ignore_index=model_cfg['train']['ignore_index'])
        self.optimizer = getattr(optim,model_cfg['train']['optimizer'])(params=self.model.parameters(),lr=model_cfg['train']['lr'])




    def train(self):

        for epoch in range(self.epochs):

            loss = self.train_epoch()
            print('loss',loss)




    def train_epoch(self):

        if self.gpus:
            torch.cuda.empty_cache()

        self.model.train()

        loss_all = 0
        print(self.batch_size)
        for batch,data in enumerate(self.dataloader):
            lidar = data['lidar']
            label = data['label']
            image = data['image']
            py = data['py']
            px = data['px']
            if self.gpus:
                lidar.cuda()
                label.cuda()
                image.cuda()
                for x in px: x.cuda()
                for y in py: y.cuda()

            print("t1",time.time())
            outputs = self.model(lidar,image,py,px)
            print("t2",time.time())

            if outputs.requires_grad:
                loss = self.criterion(outputs,label.F.long())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_all += loss.item()
            print('loss',loss.item())

        return loss_all/self.batch_size