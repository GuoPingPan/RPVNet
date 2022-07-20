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

a = SemanticKITTIInternal(root='/home/pgp/velodyne',
                          voxel_size=0.05,
                          num_points=80000,
                          sample_stride=1,
                          split='core',
                          device = 'cpu',
                          google_mode=False)

int_out = a.__getitem__(1)
int_out = a.collate_fn([int_out])
print(int_out)
exit()

from core.models.rpvnet import RPVnet
from thop import profile
# print(int_out)
lidar = int_out['lidar']
image = int_out['image']
py = int_out['py']
px = int_out['px']

model = RPVnet(cr=1,num_classes=19,device='cpu').to('cpu')
t1 = time.time()
# out = model(lidar,image,py,px)
flops,params = profile(model,(lidar,image,py,px))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
print(time.time()-t1)

# dataloader = torch.utils.data.DataLoader(
#     a,
#     batch_size=2,
#     collate_fn=a.collate_fn
# )

# inputs = next(iter(dataloader))
# print(inputs)