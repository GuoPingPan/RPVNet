import os
import sys
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import DataLoader,BatchSampler

from core.dataset.semantic_kitti import SemanticKITTIInternal
from core.models.rpvnet import RPVnet
from core.evaluator import MeanIoU

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

class Trainer():
    def __init__(self,args,model_cfg,data_cfg,):

        self.batch_size = model_cfg['train']['batch_size']
        self.gpus = model_cfg['train']['gpus']
        self.lr = model_cfg['train']['lr']
        self.loss = model_cfg['train']['loss']
        self.optimizer = model_cfg['train']['optimizer']
        self.epochs = model_cfg['train']['epochs']
        self.mode = ['train','val','test']
        self.log_dir = args.log
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')


        self.info = {
            'epochs': self.epochs,
            'epoch': 0,
            'train_loss': 0,
            'train_acc': 0,
            'train_iou': 0,
            "valid_loss": 0,
            "valid_acc": 0,
            "valid_iou": 0,
            "best_train_iou": 0,
            "best_val_iou": 0
        }


        self.model = RPVnet(
            vsize=model_cfg['voxel_size'],
            cr=model_cfg['cr'],
            cs=model_cfg['dimension_of_stages'],
            num_classes = model_cfg['num_classes']
        )

        self.criterion = getattr(nn,model_cfg['train']['loss'])(ignore_index=model_cfg['train']['ignore_index'])
        self.optimizer = getattr(optim,model_cfg['train']['optimizer'])(params=self.model.parameters(),lr=model_cfg['train']['lr'])
        self.evaluator = MeanIoU(model_cfg['num_classes'],model_cfg['ignore_index'])
        data = SemanticKITTIInternal(
            root=args.dataset_dir,
            voxel_size=data_cfg['voxel_size'],
            range_size=data_cfg['range_size'],
            sample_stride=data_cfg['sample_stride'],
            split=data_cfg['split']['train'],
            max_voxels = data_cfg['max_voxels'],
            label_name_mapping = data_cfg['label_name_mapping'],
            kept_labels = data_cfg['kept_labels']
        )

        if args.checkpoint is not None:
            state = torch.load(args.checkpoint,map_location=self.device)
            print(state['info'])
            self.epochs = state['info']['epoch']
            self.info['epochs'] = self.epochs
            self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            # self.load_static_dict()
        # todo
        # if args.freeze_layer:
        #     pass

        nw = min([os.cpu_count(), self.batch_size, 8])

        if self.device == 'cpu':
            self.gpus = None
            self.dataloader = DataLoader(data,
                                         batch_size=self.batch_size,
                                         num_workers=nw,
                                         collate_fn=data.collate_fn)

        elif torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.fastest = True
            # 单卡
            if torch.cuda.device_count() == 1:
                self.gpus = 1
                self.dataloader = DataLoader(data,
                                             batch_size=self.batch_size,
                                             num_workers=nw,
                                             collate_fn=data.collate_fn)
                self.model.to(self.device)
            # 多卡
            elif torch.cuda.device_count() > 1:
                self.init_distributed_mode(args)
                self.rank = args.rank
                self.world_size = args.world_szie
                self.lr = self.lr*self.world_size
                self.gpus = torch.cuda.device_count()

                self.sampler = DistributedSampler(data,rank=self.rank,shuffle= True)
                batch_sampler = BatchSampler(self.sampler,batch_size=self.batch_size//2,drop_last=True)
                self.dataloader = DataLoader(data,
                                             pin_memory=True, # 将数据加载到gpu中
                                             num_workers=nw,
                                             sampler=batch_sampler,
                                             collate_fn=data.collate_fn)
                # 这里device默认是cuda,因为在init进程的时候已经创建set_device了,因此这里会自动分配gpu
                self.model.to(self.device)
                if args.checkpoint is None:
                    # 由于这里要保证每个模型的初始权重相同,因此要先保存再重新加载
                    if self.rank == 0:
                        ckpt_path = os.path.join(self.log_dir,'initial.ckpt')
                        if not os.path.exists(ckpt_path):
                            torch.save({
                                'state_dict': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'info': None
                            },ckpt_path)

                    dist.barrier()
                    state_dict = torch.load(self.log_dir,ckpt_path)
                    self.model.load_state_dict(state_dict['state_dict'],strict=True)
                    self.optimizer.load_state_dict(state_dict['optimizer'])

                    # todo
                    # if args.freeze_layer:
                    #     pass
                # 这里的args.gpu应该是当前进程使用的gpu,而上面的device是'cuda'
                self.model = torch.nn.parallel.DistributedDataParallel(self.model,device_ids=[args.gpu])




    def save_checkpoint(self,state_dict,log,shuffix=''):
        torch.save()


    # def load_static_dict(self):
    #     state_dict = torch.load(self.pretrained,map_location=self.device)
    #     self.model.load_state_dict(state_dict['state_dict'],strict=True,)
    #     self.optimizer.load_state_dict(state_dict['optimizer'])
    #     self.info = state_dict['info']
    #     print(f'info: {self.info}')

    def cleanup(self):
        dist.destroy_process_group()

    def train(self):

        for epoch in range(self.epochs):
            if self.device == 'cuda':
                # 这里会根据每个epoch生成不同的种子来打乱数据
                self.sampler.set_epoch(epoch)

            loss,iou,miou,acc = self.train_each_epoch()

            if self.rank == 0:
                print(f'Epoch:[{epoch:>3d}/{self.epochs:>3d}]'
                      f'   Mean Loss:{loss}   mIoU:{miou}   Accuary:{acc}')

            if miou > self.info['best_train_iou']:
                print(f'Best mean iou in training set so far, save model!')
                self.info['epoch'] = epoch
                self.info['train_loss'] = loss
                self.info['train_acc'] = acc
                self.info['train_iou'] = iou
                self.info['best_train_iou'] =miou
                state_dict = {
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'info': self.info
                }
                if (self.gpus > 1 and self.rank == 0):
                    self.save_checkpoint(state_dict,self.log_dir,shuffix=f'rpvnet_m{miou:3f}_cr{self.model.cr:2f}_vs{self.model.vsize:2f}.ckpt')
                elif self.gpus == 1 or self.device == 'cpu':
                    self.save_checkpoint(state_dict,self.log_dir,shuffix=f'rpvnet_m{miou:3f}_cr{self.model.cr:2f}_vs{self.model.vsize:2f}.ckpt')

        self.cleanup()

    def init_distributed_mode(self,args):
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # 单机多卡的时候 RANK = LOCAL_RANK
            # 多机多卡的时候 RANK 代表 全局下的第几个进程
            #             LOCAL 代表 当前机器下的第几个进程
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])

        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        else:
            print('Not using distributed mode')
            args.distributed = False
            return

        args.distributed = True

        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
        print('| Distributed init (rank {}): {}'.format(args.rank, 'env://'), flush=True)
        # 多个进程的world_size一样,但是rank不一样
        dist.init_process_group(backend=args.dist_backend,world_size=args.world_size, rank=args.rank)
        dist.barrier()

    def reduce_value(self,value,average=True):

        with torch.no_grad():
            dist.all_reduce(value)
            if average:
                value /= self.world_size

            return value

    def train_each_epoch(self,epoch):


        if self.gpus > 1 and self.rank == 0:
            self.dataloader = tqdm(self.dataloader,file=sys.stdout)
        elif self.gpus == 1 or self.device == 'cpu':
            self.dataloader = tqdm(self.dataloader,file=sys.stdout)

        torch.cuda.empty_cache()
        self.model.train()
        self.optimizer.zero_grad()
        self.evaluator.reset()

        mean_loss = torch.zeros(1).to(self.device)

        for batch,data in enumerate(self.dataloader):
            lidar,label,image = data['lidar'],data['label'],data['image']
            py,px = data['py'],data['px']

            # 因为pin_memory已将数据放到gpu中
            if self.gpus:
                lidar.cuda(),label.cuda(),image.cuda()
                for x in px: x.cuda()
                for y in py: y.cuda()

            outputs = self.model(lidar,image,py,px)

            loss = self.criterion(outputs,label.F.long())
            loss.backward()

            loss = self.reduce_value(loss,average=True)
            mean_loss = (mean_loss*batch + loss.detach()) / (batch + 1)
            iou,miou,acc = self.evaluator(outputs,label.F.long())

            assert torch.isfinite(loss),f'ERROR: non-finite loss, ending training! {loss}'

            if self.rank == 0:
                print(f'Epoch:[{epoch:>3d}/{self.epochs:>3d}]'
                      f'   Mean Loss:{mean_loss}   mIoU:{miou}   Accuary:{acc}')


            self.optimizer.step()
            self.optimizer.zero_grad()

        # 等待所有进程计算完毕
        if self.device != torch.device('cpu'):
            torch.cuda.synchronize(self.device)

        return mean_loss,self.evaluator.epoch_miou()