# full assembly of the sub-parts to form the complete net

# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):

        super(UNet, self).__init__()
        self.down1 = down(n_channels, 32)
        self.AdditionalInput1 = AdditionalInput(2, 3, 64)
        # self.single_conv1 = single_conv(3, 64)

        self.x1_out = outconv(32, n_classes)

        self.down2 = down(96, 64)
        self.AdditionalInput2 = AdditionalInput(4, 3, 128)
        # self.single_conv2 = single_conv(3, 128)

        self.x2_out = outconv(64, n_classes)

        self.down3 = down(192, 128)
        self.AdditionalInput3 = AdditionalInput(8, 3, 256)

        # self.single_conv3 = single_conv(3, 256)

        self.x3_out = outconv(128, n_classes)

        self.down4 = down(384, 256)

        self.x4_out = outconv(256, n_classes)

        self.down5 = down(256, 256)
        self.global_adaptiveMaxPool1=global_adaptiveMaxPool()
        self.conv5 = double_conv(256,512)

        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.up4 = up(64, 32)

        self.out6 = outconv(256, n_classes)
        self.out7 = outconv(128, n_classes)
        self.out8 = outconv(64, n_classes)
        self.out9 = outconv(32, n_classes)

    def forward(self, x):

        #input1 = x   # 400x400

        input2 = self.AdditionalInput1(x)  #200x200
        input3 = self.AdditionalInput2(x)  # 100x100
        input4 = self.AdditionalInput3(x)  # 50x50

        ###first level
        x1_conv, x1_downsample = self.down1(x)  #32x200x200

        # x1_out = self.x1_out(x1_conv)

        ####second level
        input2 = torch.cat([input2, x1_downsample], dim=1)
        x2_conv, x2_downsample = self.down2(input2) #64x100x100

        # x2_out = self.x2_out(x2_conv)

        ####3rd level
        input3 = torch.cat([input3, x2_downsample], dim=1)
        x3_conv, x3_downsample = self.down3(input3)  #128x50x50

        # x3_out = self.x3_out(x3_conv)

        ### 4th level
        input4 = torch.cat([input4, x3_downsample], dim=1)
        x4_conv, x4_downsample = self.down4(input4) #256x25x25

        # x4_out = self.x4_out(x4_conv)

        ### 5th level
        x5 = self.conv5(x4_downsample) #512x25x25

        ## global average pooling
        # suppose x is your feature map with size N*C*H*W
        encoder_feature= self.global_adaptiveMaxPool1(x5)
        # encoder_feature = encoder_feature.reshape(encoder_feature.shape[0], encoder_feature.shape[1])
        # now x is of size N*C

        ### -4th level
        x6 = self.up1(x5, x4_conv) #256x50x50

        side6 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)(x6)

        ### -3th level
        x7 = self.up2(x6, x3_conv) #128x100x100
        side7 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)(x7)

        x8 = self.up3(x7, x2_conv)
        side8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x8)

        x9 = self.up4(x8, x1_conv)

        out6 = self.out6(side6)
        out7 = self.out7(side7)
        out8 = self.out8(side8)
        out9 = self.out9(x9)

        my_list=[out6, out7, out8, out9]
        out10 = torch.mean(torch.stack(my_list), dim=0)

        return encoder_feature, out6, out7, out8, out9, out10

class global_adaptiveMaxPool(nn.Module):
    def __init__(self):
        super(global_adaptiveMaxPool, self).__init__()
        self.globalmaxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        x = self.globalmaxpool(x)
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class AdditionalInput(nn.Module):
    def __init__(self,poolsize, in_ch, out_ch):
        super(AdditionalInput, self).__init__()
        self.AddiInput =nn.Sequential(
             nn.AvgPool2d(poolsize),
             nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )

    def forward(self, x):
        x = self.AddiInput(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = double_conv(in_ch, out_ch)
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):
        x_conv = self.mpconv(x)
        x_downsample = self.downsample(x_conv)
        return x_conv, x_downsample


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        self.conv = double_conv(in_ch, out_ch)

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, input_1, input_2):
        input_1 = self.up(input_1)
        
        # input is CHW
        diffY = input_2.size()[2] - input_1.size()[2]
        diffX = input_2.size()[3] - input_1.size()[3]

        input_1 = F.pad(input_1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([input_2, input_1], dim=1)
        x = self.conv(x)
        return x


class up5(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up5, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        self.conv = double_conv(in_ch, out_ch)

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

    def forward(self, x):
        x = self.up5(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1),
                                  nn.Sigmoid()
                                  )

    def forward(self, x):
        x = self.conv(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = UNet(3, 3).cuda()
    input = torch.zeros((1, 3, 512, 512)).cuda()
    output = model(input)
    print(output.shape)
    print(count_parameters(model))