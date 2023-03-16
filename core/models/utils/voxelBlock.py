import torchsparse
import torchsparse.nn as spnn
import torch.nn as nn

class BasicConvolutionBlock(nn.Sequential):
    def __init__(self,in_channels,out_channels,kernel_size=3,
                 stride=1,dilation=1):
        module = [
            spnn.Conv3d(in_channels,out_channels,kernel_size=kernel_size,
                        dilation=dilation,stride=stride),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True)
        ]
        super(BasicConvolutionBlock, self).__init__(*module)


class BasicDeconvolutionBlock(nn.Sequential):
    '''
        The difference with ConvolutionBlock is [add transposed=True]
    '''
    def __init__(self,in_channels,out_channels,kernel_size=3,
                 stride=1):
        module = [
            spnn.Conv3d(in_channels,out_channels,kernel_size=kernel_size,
                        stride=stride,transposed=True),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True)
        ]
        super(BasicDeconvolutionBlock,self).__init__(*module)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(in_channels,out_channels,kernel_size=kernel_size,
                        dilation=dilation,stride=stride),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
            spnn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                        dilation=dilation,stride=1),
            spnn.BatchNorm(out_channels),
        )

        if in_channels == out_channels and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(in_channels, out_channels, kernel_size=1,
                            dilation=1,stride=stride),
                spnn.BatchNorm(out_channels),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        # todo: What's the difference between adding with relu and adding with relu first
        # It makes a difference for LeakyReLU because of hyperparameters, 
        # but should make no difference if hyperparameters are the same
        # Plain ReLU, seems to make no difference
        out = self.relu(self.net(x) + self.downsample(x))
        return out

class DownVoxelStage(nn.Sequential):
    def __init__(self,in_channels,out_channels,
                 b_kernel_size=2,b_stride=2,b_dilation=1,
                 kernel_size=3, stride=1, dilation=1
                 ):
        module = [
            BasicConvolutionBlock(in_channels,in_channels,b_kernel_size,
                             b_stride,b_dilation),
            ResidualBlock(in_channels,out_channels,kernel_size,
                          stride,dilation),
            ResidualBlock(out_channels, out_channels, kernel_size,
                          stride, dilation)
        ]
        super(DownVoxelStage,self).__init__(*module)

class UpVoxelStage(nn.Module):
    def __init__(self, in_channels, out_channels,skip_channels,
                 b_kernel_size=2, b_stride=2,
                 kernel_size=3, stride=1, dilation=1):
        super(UpVoxelStage, self).__init__()

        self.bdb = BasicDeconvolutionBlock(in_channels, out_channels, b_kernel_size,
                                  b_stride)
        self.skip_res= nn.Sequential(
            ResidualBlock(out_channels+skip_channels, out_channels, kernel_size,
                          stride, dilation),
            ResidualBlock(out_channels, out_channels, kernel_size,
                          stride, dilation)
        )

    def forward(self,x,skip):

        out = self.bdb(x)
        out = torchsparse.cat([out,skip])
        out = self.skip_res(out)

        return out