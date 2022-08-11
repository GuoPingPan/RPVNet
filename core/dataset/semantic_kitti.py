import os
import os.path

import numpy as np
import torch

from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
import torchsparse.nn.functional as F

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
                 **kwargs):

        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.range_size = tuple(range_size)
        self.max_voxels = max_voxels
        self.sample_stride = sample_stride
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

        assert os.path.exists(self.files[index]),f'the [{self.files[index]}] doesn\'t exist.'

        # todo 是否可以加入shuffle操作 [解]:直接在Dataloader中进行shuffer就行
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
        # todo 注意除了装有bin文件的文件夹,在路径中不能有多个velodyne
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
        # 根据 x,y,z 分别减去最小的坐标值
        points_xyz_norm = points_xyz - points_xyz.min(0,keepdims=1)

        points_refl = block[:,3]
        self.data = {}
        self.point_valid_index = None
        self.do_voxel_projection(block,points_xyz_norm,labels_)
        self.do_range_projection(points_xyz,points_refl)

        self.data['filename'] = self.files[index]
        # self.data['device'] = self.device

        return self.data

    def do_voxel_projection(self,feat,points_xyz, label):
        #                             points_xyz是所有点的坐标

        # 求得voxel的坐标
        pc = np.round(points_xyz / self.voxel_size).astype(np.int32)

        # 根据 x,y,z 分别最小的坐标值,在之前已经进行了
        # pc -= pc.min(0, keepdims=1)

        # 这个函数并不改变pc,pc还是12000个
        _, inds, inverse_map = sparse_quantize(pc, return_index=True,
                                              return_inverse=True)

        # todo 在训练的时候将会剔除某些voxel,因此在这个过程中需要将voxel里面对应的点云也去掉
        # todo 采用torchsparse来加速
        if self.mode == 'train':
            if len(inds) > self.max_voxels:
                inds = np.random.choice(inds,self.max_voxels,replace=False)
                pc_ = pc[inds]
                all_point = torch.concat([torch.from_numpy(pc),torch.zeros(pc.shape[0]).reshape(-1,1)],dim=1).int()
                voxel_valid = torch.concat([torch.from_numpy(pc_),torch.zeros(pc_.shape[0]).reshape(-1,1)],dim=1).int()

                old_hash = F.sphash(all_point) # 120000+
                sparse_hash = F.sphash(voxel_valid) # max 84000

                self.point_valid_index = F.sphashquery(old_hash,sparse_hash)


        # todo 固定点的个数有助于将 px,py 变成一个规则的 tensor,加快 r2p,p2r
        # if self.point_valid_index.shape[0] > self.num_points :
        #     pass

        coord_,label_,feat_ = (points_xyz[self.point_valid_index != -1],
                               label[self.point_valid_index != -1],
                               feat[self.point_valid_index != -1]) \
                        if self.point_valid_index is not None else (points_xyz,label,feat)


        # 这里存进去的坐标都是浮点数
        self.data['lidar'] = SparseTensor(feats=feat_,coords=coord_)
        self.data['label'] = SparseTensor(feats=label_,coords=coord_)


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



        # np.nonzero返回数组中的非零索引 [https://blog.csdn.net/u013698770/article/details/54632047/]
        # 目的是找到64线的交接点,就是从 >0.8到<0.2的这个跳转点,因为是64线lidar,因此有64个点
        new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
        proj_y = np.zeros_like(proj_x)
        proj_y[new_raw] = 1

        # 累加求得每个点在哪条线上
        proj_y = np.cumsum(proj_y)
        # todo 如何去解决旋转会导致索引越界
        # print(np.where(proj_y>=65.0))

        # scale to image size using angular resolution
        # 这里 -0.001 能保证后面取floor落在0~2047的范围
        proj_x = proj_x * W - 0.001

        px = proj_x.copy()
        py = proj_y.copy()

        proj_x = np.floor(proj_x).astype(np.int32)
        proj_y = np.floor(proj_y).astype(np.int32)

        ''' 
        v1: using the closet point's depth 
        
        # order in decreasing depth
        # 使得深度从大到小排列,同一位置将会使用距离小的点的距离来填充
        order = np.argsort(depth)[::-1]

        depth = depth[order]
        reflectivity = points_refl[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]


        proj_range = np.zeros((H, W))
        # 逆深度
        proj_range[proj_y, proj_x] = 1.0 / depth

        proj_reflectivity = np.zeros((H, W))
        proj_reflectivity[proj_y, proj_x] = reflectivity

        '''

        ''' 
        v2: using the average points' depth 
        '''
        proj_range = np.zeros((H,W)) + 1e-5
        proj_cumsum = np.zeros((H,W)) + 1e-5
        proj_reflectivity = np.zeros((H, W))
        proj_range[proj_y,proj_x] += depth
        proj_cumsum[proj_y,proj_x] += 1
        proj_reflectivity[proj_y, proj_x] += points_refl


        # inverse depth
        proj_range = proj_cumsum / proj_range

        proj_reflectivity = proj_reflectivity / proj_cumsum


        # nomalize values to -10 and 10
        depth_image = 25 * (proj_range - 0.4)
        refl_image = 20 * (proj_reflectivity - 0.5)


        # 默认在channel维度进行拼接了
        range_image = np.stack([depth_image,refl_image]).astype(np.float32)

        px = px[np.newaxis,:]
        py = py[np.newaxis,:]
        py = 2. * (py / H - 0.5)
        px = 2. * (px / W - 0.5)

        self.data['image'] = range_image
        self.data['py'] = py
        self.data['px'] = px

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
