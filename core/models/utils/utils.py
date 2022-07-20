import time

import numpy as np
import torch
import torchsparse.nn.functional as F
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets
from torch.nn.functional import grid_sample

# __all__ = ['initial_voxelize', 'point_to_voxel', 'voxel_to_point']


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, after_res):

    # print(init_res,after_res)
    # 其实在转voxel之间在构造数据集过程中已经进行了一次转voxel
    # 这里是为了保证voxel的大小和模型的大小一样
    new_float_coord = torch.cat(
        [z.C[:, :3]  / after_res, z.C[:, -1].view(-1, 1)], 1)

    # 使用 pybind11 将hash模块利用c++实现
    # todo 这里将floor改成了round
    pc_hash = F.sphash(torch.round(new_float_coord).int())
    # print(pc_hash.shape,new_float_coord.shape)

    # 将其变成唯一的 hash,只有唯一的voxel编码
    sparse_hash = torch.unique(pc_hash)
    # print(pc_hash.shape,sparse_hash.shape)
    # exit()

    # 获得pc_hash在sparse_hash中的index
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    # print(idx_query)

    # 这里是对idx_query中从0开始计数
    # tensor([2, 0, 0, 0, 1, 2, 2, 2, 4, 3, 2])
    # tensor([3, 1, 5, 1, 1], dtype=torch.int32)
    counts = F.spcount(idx_query.int(), len(sparse_hash))


    # 按照idx_query的0,1,2...选出对应的voxel
    # 这里已经将所有在同一个voxel里面的点都加和取平均了,其实这一步感觉可以加速,因为落在同一个voxel的点其坐标都一样
    inserted_coords = F.spvoxelize(torch.round(new_float_coord), idx_query,
                                   counts)
    # print('insert',inserted_coords.shape)
    # print(inserted_coords)

    # 根本没处理
    # print(pc_hash.shape,inserted_coords.shape)
    # exit()

    # 转成 int
    inserted_coords = torch.round(inserted_coords).int()

    # feature也挑选出来
    # 这里已经将所有在同一个voxel里面的点的feature都加和取平均了
    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    # print('feat',inserted_feat.shape)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)

    # 根据坐标上色
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    # 将对应的索引存到point中
    z.additional_features['idx_query'][(1,1,1)] = idx_query
    z.additional_features['counts'][(1,1,1)] = counts
    # z.C = new_float_coord

    # 这里只保留了唯一的voxel,从idx_query的0,1,...往后索引,通过idx_query可以索引到对应的point
    return new_tensor.to(z.F.device)


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    # 防止get错误
    if z.additional_features is None or z.additional_features['idx_query'] is None \
            or z.additional_features['idx_query'].get(x.s) is None:

        pc_hash = F.sphash(
            torch.cat([
                # floor -> round    点的坐标永远不变,因此只需要/x.s
                torch.round(z.C[:, :3] / x.s[0]).int(),
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = F.sphash(x.C)

        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])

        # todo 这里其实只有 x.s = 1,1,1就是在末尾的一个融合块才用上了,其他用不上,而且这个在initialize的时候已经加入了
        # z.additional_features['idx_query'][x.s] = idx_query
        # z.additional_features['counts'][x.s] = counts
    else:

        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False,cnt=0):

    if z.idx_query is None or z.weights is None or z.idx_query.get(x.s) is None \
            or z.weights.get(x.s) is None:

        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)

        # kernel_hash,生成了当前坐标包括8个偏移量的编码
        old_hash = F.sphash(
            torch.cat([
                torch.round(z.C[:, :3] / x.s[0]).int(),
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)


        # voxel中偏移量的编码
        pc_hash = F.sphash(x.C.to(z.F.device))


        # 找到每个偏移量在voxel中对应的位置,也就是对于每个point会有8个临近的voxel
        idx_query = F.sphashquery(old_hash, pc_hash)

        weights = F.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).transpose(0, 1).contiguous()

        idx_query = idx_query.transpose(0, 1).contiguous()

        # 最近邻插值只考虑点所落在的voxel里面,会带来一个弊端就是同一个grid中所有的point都共享相同的一个权重
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1

        # 空间中的八个
        new_feat = F.spdevoxelize(x.F, idx_query, weights)
        # todo 这里只需要feature就行了
        # new_tensor = PointTensor(new_feat,z.C)

        # new_tensor.additional_features = z.additional_features
        # new_tensor.idx_query[x.s] = idx_query
        # new_tensor.weights[x.s] = weights
        if x.s == (1,1,1):
            # 这个idx_query和additional_features中的idx_query是不一样的,这里是八个点
            z.idx_query[x.s] = idx_query
            z.weights[x.s] = weights
        # print(idx_query.shape,weights.shape)
        # exit()
    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))

        # new_tensor = PointTensor(new_feat,z.C)
        # new_tensor.additional_features = z.additional_features

    return new_feat

def range_to_point(x,px,py):

    r2p = []

    # todo 这里要是想快点只能是进行固定训练点的数量
    # t1 = time.time() #0.01*batch_size
    for batch,(p_x,p_y) in enumerate(zip(px,py)):
        pypx = torch.stack([p_x,p_y],dim=2)
        resampled = grid_sample(x[batch].unsqueeze(0),pypx.unsqueeze(0))
        # print(resampled.squeeze().permute(1,0).shape)
        r2p.append(resampled.squeeze().permute(1,0))

    # stack和concat的区别就是是否会增加新的特征维度,stack则是在指定的dim增加一个新的维度
    return torch.concat(r2p,dim=0)

    # print(time.time()-t1)

def point_to_range(range_shape,pF,px,py):
    H, W = range_shape
    cnt = 0
    r = []
    for batch,(p_x,p_y) in enumerate(zip(px,py)):
        image = torch.zeros(size=(H,W,pF.shape[1]))
        p_x = torch.floor((p_x/2. + 0.5) * W).long()
        p_y = torch.floor((p_y/2. + 0.5) * H).long()
        image[p_y,p_x] = pF[cnt,cnt+p_x.shape[0]]

        r.append(image.permute(2,0,1))
        cnt += p_x.shape[0]

    return torch.stack(r,dim=0)

