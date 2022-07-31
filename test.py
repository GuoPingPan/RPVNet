import time

# a = list(np.random.randint(90000,size=(12000)))
# b = list(np.random.randint(90000,size=(80000)))
#
# t1 = time.time()
# idx = [ id if id in b else 0 for id in a]
# print(time.time()-t1)
# print(idx)

# depth = np.random.randn(100).reshape(-1,10)
# rgb = np.random.randn(100).reshape(-1,10)
# out = np.stack([depth,rgb])
# print(out.shape)

# a = [torch.randn(size=(10,10)) for i in range(10)]
# out = torch.stack(a,dim=0)
# print(out.shape)
#
# out1 = torch.concat(a,dim=0)
# print(out1.shape)

from core.dataset.semantic_kitti import SemanticKITTIInternal
import warnings
warnings.filterwarnings("ignore")


import yaml
from torch.utils.data import DataLoader
data_cfg = yaml.safe_load(open('config/semantic-kitti.yaml', 'r'))
#
data = SemanticKITTIInternal(
    root='/home/pgp/xialingying/dataset',
    voxel_size=data_cfg['voxel_size'],
    range_size=data_cfg['range_size'],
    sample_stride=data_cfg['sample_stride'],
    split=data_cfg['split']['train'],
    max_voxels=data_cfg['max_voxels'],
    label_name_mapping=data_cfg['label_name_mapping'],
    kept_labels=data_cfg['kept_labels']
)
dataloader = DataLoader(data,
                             batch_size=2,
                             num_workers=4,
                             collate_fn=data.collate_fn,
                             shuffle=False)
#
int_out = data.__getitem__(1)
int_out = data.collate_fn([int_out])

int_out = next(iter(dataloader))


from core.models.rpvnet import RPVnet
from thop import profile
# print(int_out)
lidar = int_out['lidar']
image = int_out['image']
py = int_out['py']
px = int_out['px']

# print(py.shape)
# exit()

model = RPVnet(
    cr=1,
    vsize=0.05,
    cs = [32,64,128,256,256,128,128,64,32],
    num_classes=19
)
# t1 = time.time()
for i in range(5):
    out = model(lidar,image,py,px)
    print('*'*100)
print('*'*100)
out = model(lidar, image, py, px)

# flops,params = profile(model,(lidar,image,py,px))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
# print(time.time()-t1)

# dataloader = torch.utils.data.DataLoader(
#     a,
#     batch_size=2,
#     collate_fn=a.collate_fn
# )

# inputs = next(iter(dataloader))
# print(inputs)