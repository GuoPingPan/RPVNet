import torch
import torch.nn as nn
import torch.nn.functional as F


class ResContextBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ResContextBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=(1,1))
        self.act1 = nn.LeakyReLU(True)

        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),padding=1)
        self.act2 = nn.LeakyReLU(True)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),padding=2,dilation=2)
        self.act3 = nn.LeakyReLU(True)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):

        shortcut = self.act1(self.conv1(x))
        x = self.bn1(self.act2(self.conv2(shortcut)))
        x = self.bn2(self.act3(self.conv3(shortcut)))

        out = x + shortcut
        return out



class Block1Res(nn.Module):
    def __init__(self,  in_channels, out_channels):
        super(Block1Res, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.act1 = nn.LeakyReLU(True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),padding=1)
        self.act2 = nn.LeakyReLU(True)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU(True)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU(True)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv5 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU(True)
        self.bn4 = nn.BatchNorm2d(out_channels)

    def forward(self,x):

        shortcut = self.act1(self.conv1(x))

        cat1 = self.bn1(self.act2(self.conv2(x)))
        cat2 = self.bn2(self.act3(self.conv3(cat1)))
        cat3 = self.bn3(self.act4(self.conv4(cat2)))

        cat = torch.concat([cat1,cat2,cat3],dim=1)
        cat = self.bn4(self.act5(self.conv5(cat)))
        out = shortcut + cat

        return out

class Block2(nn.Sequential):
    def __init__(self,dropout_rate=0.2,kernel_size=3,pooling=True,drop_out=True):
        module = [
            nn.Dropout2d(p=dropout_rate) if drop_out else nn.Identity(),
            nn.AvgPool2d(kernel_size=kernel_size,stride=2,padding=1) if pooling else nn.Identity(),
        ]
        super(Block2, self).__init__(*module)


class Block4(nn.Module):
    def __init__(self, in_channels, out_channels,skip_channels,upscale_factor=2,dropout_rate=0.2,drop_out=True):
        super(Block4, self).__init__()

        self.upscale = nn.PixelShuffle(upscale_factor=upscale_factor)

        self.conv1 = nn.Conv2d(in_channels//(upscale_factor**2) + skip_channels, out_channels, (3, 3), padding=1)
        self.act1 = nn.LeakyReLU(True)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), dilation=2, padding=2)
        self.act2 = nn.LeakyReLU(True)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, (2, 2), dilation=2, padding=1)
        self.act3 = nn.LeakyReLU(True)
        self.bn3 = nn.BatchNorm2d(out_channels)


        self.conv4 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=(1, 1))
        self.act4 = nn.LeakyReLU(True)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(p=dropout_rate) if drop_out else nn.Identity()

    def forward(self,x,skip):

        x = self.upscale(x)
        x = self.dropout(x)

        upcat = torch.concat([x,skip],dim=1)
        upcat = self.dropout(upcat)


        cat1 = self.bn1(self.act1(self.conv1(upcat)))
        cat2 = self.bn2(self.act2(self.conv2(cat1)))
        cat3 = self.bn3(self.act3(self.conv3(cat2)))

        cat = torch.concat([cat1,cat2,cat3],dim=1)

        out = self.bn4(self.act4(self.conv4(cat)))
        out =self.dropout(out)

        return out