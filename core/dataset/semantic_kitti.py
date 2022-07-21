import os
import os.path

import numpy as np
import torch

from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
import torchsparse.nn.functional as F
# from torchsparse.utils.collate import sparse_collate_fn

from core.dataset.collate import sparse_collate_fn


__all__ = ['SemanticKITTIInternal']

class SemanticKITTIInternal:

    def __init__(self,
                 root,
                 voxel_size,
                 max_voxels,
                 split,
                 label_name_mapping,
                 kept_labels,
                 mode = 'train',
                 range_size = None,
                 sample_stride=1,
                 submit=False,
                 # device='cpu',
                 **kwargs):

        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.range_size = tuple(range_size)
        self.max_voxels = max_voxels
        self.sample_stride = sample_stride
        # self.device = device
        self.mode = mode

        # 如果提交则加上val一起进行训练
        if submit:
            self.split.append('08')

        # 组合所有序列的bin数据路径
        self.files = []
        for seq in self.split:
            seq_files = sorted(
                os.listdir(os.path.join(self.root, seq, 'velodyne')))
            seq_files = [
                os.path.join(self.root, seq, 'velodyne', x) for x in seq_files
            ]
            self.files.extend(seq_files)

        # 根据数据步长进行采样
        if self.sample_stride > 1:
            self.files = self.files[::self.sample_stride]

        # 构建label id 和 label name的映射
        reverse_label_name_mapping = {} # 从label name -> cnt
        self.label_map = np.zeros(260) # 从 原始label -> cnt
        cnt = 0
        for label_id in label_name_mapping:
            if label_id > 250:
                if label_name_mapping[label_id].replace('moving-',
                                                        '') in kept_labels:
                    self.label_map[label_id] = reverse_label_name_mapping[
                        label_name_mapping[label_id].replace('moving-', '')]
                else:
                    self.label_map[label_id] = 255
            elif label_id == 0:
                self.label_map[label_id] = 255
            else:
                if label_name_mapping[label_id] in kept_labels:
                    self.label_map[label_id] = cnt
                    reverse_label_name_mapping[
                        label_name_mapping[label_id]] = cnt
                    cnt += 1
                else:
                    self.label_map[label_id] = 255

        self.reverse_label_name_mapping = reverse_label_name_mapping
        self.num_classes = cnt
        self.angle = 0.0

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # todo 是否可以加入shuffle操作
        # 应该在dataloader里面设置就行
        assert os.path.exists(self.files[index]),f'the [{self.files[index]}] doesn\'t exist.'

        with open(self.files[index], 'rb') as b:
            block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        block = np.zeros_like(block_)

        # 数据增强操作
        if self.mode == 'train':
            # todo 如何去解决旋转会导致索引越界
            # theta = np.random.uniform(0, 2 * np.pi)
            theta = self.angle
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),np.cos(theta), 0],
                                [0, 0, 1]])

            block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor

        else:
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),np.sin(theta), 0],
                                      [-np.sin(theta),np.cos(theta), 0],
                                      [0, 0, 1]])

            block[:, :3] = np.dot(block_[:, :3], transform_mat)

        # 反射率
        block[:, 3] = block_[:, 3]

        # 读取label
        label_file = self.files[index].replace('velodyne', 'labels').replace('.bin', '.label')
        assert os.path.exists(label_file),f'the [{label_file}] doesn\'t exist.'

        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros(block.shape[0]).astype(np.int32)

        # 截取32位
        labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)
        # print(np.where(labels_==255))

        points_xyz = block[:,:3]
        # 根据 x,y,z 分别最小的坐标值
        points_xyz -= points_xyz.min(0,keepdims=1)

        points_refl = block[:,3]
        self.data = {}
        self.point_valid_index = None
        self.do_voxel_projection(block,points_xyz,labels_)
        self.do_range_projection(points_xyz,points_refl)

        self.data['filename'] = self.files[index]
        # self.data['device'] = self.device

        return self.data

    def do_voxel_projection(self,feat,points_xyz, label):

        # points_xyz是所有点的坐标
        # 求得voxel的坐标
        pc = np.round(points_xyz / self.voxel_size).astype(np.int32)
        # 根据 x,y,z 分别最小的坐标值
        # pc -= pc.min(0, keepdims=1)

        # 这个函数并不改变pc,pc还是12000个
        _, inds, inverse_map = sparse_quantize(pc, return_index=True,
                                              return_inverse=True)
        # print(inds.shape)
        # 在训练的时候将会剔除某些voxel,因此在这个过程中需要将voxel里面对应的点云也去掉
        # 采用torchsparse来加速
        if self.mode == 'train':
            if len(inds) > self.max_voxels:
                inds = np.random.choice(inds,self.max_voxels,replace=False)
                pc_ = pc[inds]
                all_point = torch.concat([torch.from_numpy(pc),torch.zeros(pc.shape[0]).reshape(-1,1)],dim=1).int()
                voxel_valid = torch.concat([torch.from_numpy(pc_),torch.zeros(pc_.shape[0]).reshape(-1,1)],dim=1).int()

                old_hash = F.sphash(all_point) # 120000+
                sparse_hash = F.sphash(voxel_valid) # max 84000

                self.point_valid_index = F.sphashquery(old_hash,sparse_hash)

                # 测试为什么点的很少,因为一个除了voxel,一个没除
                # print(self.point_valid_index[self.point_valid_index!=-1].shape)
                # points_xyz = points_xyz[self.point_valid_index!=-1]
                # print(points_xyz.shape)
                # new_float_coord = torch.cat(
                #     [(torch.from_numpy(points_xyz)) / 0.05, torch.zeros(size=(points_xyz.shape[0],1))], 1)
                # pc_hash = F.sphash(torch.floor(new_float_coord).int())
                # spa = torch.unique(pc_hash)
                # print(spa.shape)
                # exit()

        # todo 要不要裁剪点
        # if self.point_valid_index.shape[0] > self.num_points :
        #     # 没有重复采样replace=False
        #     self.point_valid_index = np.random.choice(self.point_valid_index,self.num_points,replace=False)
        #
        # print(self.point_valid_index)
        # self.point_valid_index = set(self.point_valid_index)
        # print(self.point_valid_index)
        # self.point_valid_index.remove(0)

        # print(self.point_valid_index != -1)



        coord_,label_,feat_ = (points_xyz[self.point_valid_index != -1],
                               label[self.point_valid_index != -1],
                               feat[self.point_valid_index != -1]) \
                        if self.point_valid_index is not None else (points_xyz,label,feat)

        # print(feat_.shape,coord_.shape,label_.shape,points_xyz.shape,label.shape)
        # (80000, 4)(106616, 3)(106616, )(124231, 3)(124231, )

        # 这里存进去的坐标都是浮点数
        self.data['lidar'] = SparseTensor(feats=feat_,coords=coord_)
        self.data['label'] = SparseTensor(feats=label_,coords=coord_)
        # self.data['target_mapped'] = SparseTensor(feats=label,coords=points_xyz)
        # self.data['inverse_map'] = SparseTensor(feats=inverse_map,coords=points_xyz)



    def do_range_projection(self,points_xyz, points_refl):

        H,W = self.range_size if self.range_size is not None else (64,2048)

        points_xyz,points_refl = (points_xyz[self.point_valid_index != -1],
                                  points_refl[self.point_valid_index != -1]) \
                     if self.point_valid_index is not None else (points_xyz,points_refl)

        # 计算每个点的模长作为深度
        depth = np.linalg.norm(points_xyz,2,axis=1)

        # get scan components
        scan_x = points_xyz[:, 0]
        scan_y = points_xyz[:, 1]
        scan_z = points_xyz[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, -scan_x)
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]


        # 　https://blog.csdn.net/u013698770/article/details/54632047/
        # np.nonzero返回数组中的非零索引

        # 目的是找到64线的交接点,就是从 >0.8到<0.2的这个跳转点,因为是64线lidar,因此有64个点
        new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1

        # 对于每个点都给定对应的位置
        proj_y = np.zeros_like(proj_x)

        # 对应位置赋1
        proj_y[new_raw] = 1

        # 累加求得每个点在哪条线上
        proj_y = np.cumsum(proj_y)
        # todo 如何去解决旋转会导致索引越界
        # print(proj_y.max())
        # print(np.where(proj_y>=65.0))

        # scale to image size using angular resolution
        # 这里 -0.001 能保证后面取floor落在0~2047的范围
        proj_x = proj_x * W - 0.001

        px = proj_x.copy()
        py = proj_y.copy()

        proj_x = np.floor(proj_x).astype(np.int32)
        proj_y = np.floor(proj_y).astype(np.int32)
        # print(proj_y.max())

        # order in decreasing depth
        # 使得深度从大到小排列,同一位置将会使用距离小的点的距离来填充
        order = np.argsort(depth)[::-1]

        depth = depth[order]
        reflectivity = points_refl[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # 这里也需要对point进行排序,因为px,py是需要和point进行绑定的,从range到point过程中要对应
        # todo 疑似kprnet中有bug
        # self.data['lidar'] = SparseTensor(feats=feat_,coords=coord_)
        # self.data['label'] = SparseTensor(feats=label_,coords=coord_)
        # self.data['inverse_map'] = SparseTensor(feats=inverse_map,coords=points_xyz)
        self.data['lidar'].F = self.data['lidar'].F[order]
        self.data['lidar'].C = self.data['lidar'].C[order]
        self.data['label'].F = self.data['label'].F[order]
        self.data['label'].C = self.data['lidar'].C
        # 这里如果打榜提交数据要符合输入数据顺序我们才需要进行inverse_order的操作
        # self.data['inverse_order'] =

        proj_range = np.zeros((H, W))
        # 逆深度
        proj_range[proj_y, proj_x] = 1.0 / depth

        proj_reflectivity = np.zeros((H, W))
        proj_reflectivity[proj_y, proj_x] = reflectivity

        # nomalize values to -10 and 10
        depth_image = 25 * (proj_range - 0.4)
        refl_image = 20 * (proj_reflectivity - 0.5)

        # import matplotlib.pyplot as plt
        # plt.imshow(depth_image)
        # plt.imshow(refl_image)
        # plt.show()



        # 默认在channel维度进行拼接了
        range_image = np.stack([depth_image,refl_image]).astype(np.float32)

        px = px[np.newaxis,:]
        py = py[np.newaxis,:]
        py = 2. * (py / H - 0.5)
        px = 2. * (px / W - 0.5)

        self.data['image'] = range_image
        self.data['py'] = py
        self.data['px'] = px
        # print(range_image.shape,py.shape,px.shape)
        # (2, 64, 2048)(1, 103683)(1, 103683)

    @staticmethod
    def collate_fn(inputs):
        '''
        self.data['lidar'] = SparseTensor(feats=feat_,coords=coord_)
        self.data['label'] = SparseTensor(feats=label_,coords=coord_)
        self.data['filename'] = self.files[index]
        self.data['image'] = range_image
        self.data['py'] = py
        self.data['px'] = px
        '''
        return sparse_collate_fn(inputs)

'''-------------------------------------------------------------------------------'''
# import yaml
# data_cfg = yaml.safe_load(open('../../config/semantic-kitti.yaml', 'r'))
# #
# data = SemanticKITTIInternal(
#     root='/home/pgp/xialingying/dataset',
#     voxel_size=data_cfg['voxel_size'],
#     range_size=data_cfg['range_size'],
#     sample_stride=data_cfg['sample_stride'],
#     split=data_cfg['split']['train'],
#     max_voxels=data_cfg['max_voxels'],
#     label_name_mapping=data_cfg['label_name_mapping'],
#     kept_labels=data_cfg['kept_labels']
# )
# #
# int_out = data.__getitem__(1)
# int_out = data.collate_fn([int_out])
#
# lidar = int_out['lidar']
# image = int_out['image']
# label = int_out['label']
# py = int_out['py']
# px = int_out['px']
#
# print(lidar.F.shape,lidar.C.shape)
# print(label.F.shape,label.C.shape)
# print(np.where(label.F!=255))
#
# from core.models.rpvnet import RPVnet
#
# model = RPVnet(
#     cr=1,
#     vsize=0.05,
#     cs = [32,64,128,256,256,128,128,64,32],
#     num_classes=19
# )
#
# loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
#
# out = model(lidar,image,py,px)
# loss = loss_fn(out,label.F.long())
# print(loss)



# print(np.where(int_out['label'].F!=255))
# # lidar,label,image,px,py,device = int_out
# # print(lidar)
# dataloader = torch.utils.data.DataLoader(
#     data,
#     batch_size=2,
#     collate_fn=data.collate_fn
# )

# inputs = next(iter(dataloader))
# print(inputs)