''' test quit '''
# a = input()
# b = 'a';
# if 1:
#     if b == 'a':
#         if a == 'n':
#             quit()
#
#     print(b)
import os

import torch.distributed

''' test yaml '''
# import yaml
#
# a = yaml.safe_load(open('semantic-kitti.yaml','r'),)
# a = yaml.load(open('semantic-kitti.yaml','r'),Loader=yaml.FullLoader)
# # print(a)
# a = yaml.safe_load(open('config/semantic-kitti.yaml','r'),)
# # print(a)
# label = a['label_name_mapping']
# kept = a['kept_labels']
# print(label)
# # print(type(kept))
#
# import torch.backends.cudnn as cudnn
# cudnn.benchmark = True
# cudnn.enabled = True
# a,b,c = tuple([1,2,34])
# print(a,b,c)

# dirname,filename = os.path.split(os.path.abspath(__file__))
# print(dirname,filename)
#
# os.mkdir('fdasf')

# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '-d',
#     # type=str,
#     action='store_true',
#     default='afsa',
#     help='fa'
# )
#
# arg = parser.parse_args()
#
# print(arg.d)


# import os
#
#
# assert os.path.exists('./3') ,'fsad'
#
#
# fa = 10

# print(torch.distributed.is_nccl_available())

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(1,1,kernel_size=3,padding=2),
    nn.ReLU(inplace=True)
)

a = torch.randn(size=(1,1,5,5))
print(a)
model(a)
# print(out)
print(a)

