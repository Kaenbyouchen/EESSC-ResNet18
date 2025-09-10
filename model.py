import torch
import torch.nn as nn
from torch.nn import functional as F


"""
把ResNet18的残差卷积单元作为一个Block，这里分为两种：一种是CommonBlock，另一种是SpecialBlock，最后由ResNet18统筹调度
其中SpecialBlock负责完成ResNet18中带有虚线（升维channel增加和下采样操作h和w减少）的Block操作
其中CommonBlock负责完成ResNet18中带有实线的直接相连相加的Block操作
注意ResNet18中所有非shortcut部分的卷积kernel_size=3， padding=1，仅仅in_channel, out_channel, stride的不同 
注意ResNet18中所有shortcut部分的卷积kernel_size=1， padding=0，仅仅in_channel, out_channel, stride的不同
CommonBlock: Straightly added 
SpecialBlock: Using 1*1 Conv to downsample and then add up
"""


class CommonBlock(nn.Module):   #First two blocks(without 1*1 downsample)
    def __init__(self, in_channel, out_channel, stride):        # each CommonBlock 2 conv,stride=1
        super(CommonBlock, self).__init__()   #definition of conv and bn
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x                                            # Straightly connect (without downsample)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)       # first conv
        x = self.bn2(self.conv2(x))                             # second conv (no need relu)

        x += identity                                           # add up 1st and 2nd then relu
        return F.relu(x, inplace=True)                          # output


class SpecialBlock(nn.Module):                                  # include 1*1 downsample
    def __init__(self, in_channel, out_channel, stride):        # stride[2,1]
        super(SpecialBlock, self).__init__()
        self.change_channel = nn.Sequential(                    # change_channel:downsample, dimentional increase
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = self.change_channel(x)                       # downsample

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)                          # output


class ResNet18(nn.Module):
    def __init__(self, classes_num,in_ch=3):   # can set different chaannels (RGB:3 RAW:1 PACKED:4)
        super(ResNet18, self).__init__()
        self.prepare = nn.Sequential(           #
            nn.Conv2d(in_ch, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(            # two CommonBlock
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(            # one SpecialBlock, one CommonBlock
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),
            CommonBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            SpecialBlock(256, 512, [2, 1]),
            CommonBlock(512, 512, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, classes_num)
        )

    def forward(self, x):
        x = self.prepare(x)

        x = self.layer1(x)          # four blocks
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)   # unfold x to fc
        x = self.fc(x)

        return x

