from typing import Any, List

import numpy as np
import torch

from torchsparse import SparseTensor

__all__ = ['sparse_collate', 'sparse_collate_fn']


def sparse_collate(inputs: List[SparseTensor]) -> SparseTensor:
    # 输入是一个同name的SparseTensor组成的列表,比如说batch_size个'laser'
    coords, feats = [], []
    stride = inputs[0].stride

    for k, x in enumerate(inputs):
        # 转torch
        if isinstance(x.coords, np.ndarray):
            x.coords = torch.tensor(x.coords)
        if isinstance(x.feats, np.ndarray):
            x.feats = torch.tensor(x.feats)

        assert isinstance(x.coords, torch.Tensor), type(x.coords)
        assert isinstance(x.feats, torch.Tensor), type(x.feats)
        # 保证所有输入的stride都是相同的
        assert x.stride == stride, (x.stride, stride)

        input_size = x.coords.shape[0]
        batch = torch.full((input_size, 1),
                           k,
                           device=x.coords.device,
                           dtype=torch.int)
        # 对每个coord加上在batch中的编号
        coords.append(torch.cat((x.coords, batch), dim=1))
        # 将特征组合在一起
        feats.append(x.feats)

    # 根据点数n进行拼接
    coords = torch.cat(coords, dim=0)
    feats = torch.cat(feats, dim=0)
    # 又重新放入SparseTensor,其实SparseTensor并不发挥任何功能,只是充当一个容器
    output = SparseTensor(coords=coords, feats=feats, stride=stride)
    return output


def sparse_collate_fn(inputs: List[Any]) -> Any:
    # print(inputs)
    # inputs是一个列表[]中装着dataset中__getitem__得到的batch_size个数据
    # 如batch_size=8 [{'laser':torchsparse.tensor.SparseTensor,...},...*8个]

    # todo 2这里device
    # device = inputs[0].get('device')

    # 如果输入是一个 dict则遍历所有的key,value,否则直接输出
    if isinstance(inputs[0], dict):
        output = {}
        for name in inputs[0].keys():
            # 如果是一个列表则继续递归调用 sparse_collate_fn,并没有使用
            if isinstance(inputs[0][name], dict):
                output[name] = sparse_collate_fn(
                    [input[name] for input in inputs])

            # 对px,py由于尺寸动态变化,因此单独处理
            # todo 加入到 sparsetensor 里面进行处理变尺度问题,但是需要最后一个维度为batch
            # todo 但这样会造成索引变慢
            elif name in ['px','py']:
                output[name] = [torch.from_numpy(input[name]) for input in inputs]

            # todo 1这里device
            # elif name == 'device':
            #     pass

            # 如果是一个np,输入中所有的相同name(key)的输入进行拼接
            elif isinstance(inputs[0][name], np.ndarray):
                output[name] = torch.stack(
                    [torch.tensor(input[name])for input in inputs], dim=0)

            # 如果是一个torch,输入中所有的相同name(key)的输入进行拼接
            elif isinstance(inputs[0][name], torch.Tensor):
                output[name] = torch.stack([input[name] for input in inputs],
                                           dim=0)
            # 如果是SparseTensor则调用自己的函数处理
            elif isinstance(inputs[0][name], SparseTensor):
                output[name] = sparse_collate([input[name] for input in inputs])
            else:
                output[name] = [input[name] for input in inputs]
        return output
    else:
        return inputs
