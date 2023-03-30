import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class UNetSAR(nn.Module):
    def __init__(self):
        super(UNetSAR,self).__init__()
        self.enc_conv1 = nn.Conv2d(1,16, kernel_size=3, stride=1) # 1 in-channel (the ARC image), 16 out channels
        self.enc_conv2 = nn.Conv2d(16,32, kernel_size=3, stride=1) # 16 in-channels, 32 out channels
        # self.enc_conv3 = nn.Conv1d(128,256, kernel_size=4, stride=2)
        # self.enc_conv4 = nn.Conv1d(256,512, kernel_size=4, stride=2)
        # self.enc_conv5 = nn.Conv1d(512,512, kernel_size=4, stride=2)
        # self.enc_conv6 = nn.Conv1d(512,512, kernel_size=4, stride=2)
        # self.enc_conv7 = nn.Conv1d(512,512, kernel_size=4, stride=2)
        # self.enc_conv8 = nn.Conv1d(512,512, kernel_size=4, stride=2)

        self.enc_bn1 = nn.BatchNorm2d(16)
        self.enc_bn2 = nn.BatchNorm2d(32)
        # self.enc_bn3 = nn.BatchNorm1d(256)
        # self.enc_bn4 = nn.BatchNorm1d(512)
        # self.enc_bn5 = nn.BatchNorm1d(512)
        # self.enc_bn6 = nn.BatchNorm1d(512)
        # self.enc_bn7 = nn.BatchNorm1d(512)
        # self.enc_bn8 = nn.BatchNorm1d(512)

        # self.dec_conv8 = nn.ConvTranspose1d(512,512, kernel_size=4, stride=2, output_padding=1)
        # self.dec_conv7 = nn.ConvTranspose1d(1024,512, kernel_size=4, stride=2, output_padding=1)
        # self.dec_conv6 = nn.ConvTranspose1d(1024,512, kernel_size=4, stride=2, output_padding=1)
        # self.dec_conv5 = nn.ConvTranspose1d(1024,512, kernel_size=4, stride=2)
        # self.dec_conv4 = nn.ConvTranspose1d(1024,256, kernel_size=4, stride=2, output_padding=1)
        # self.dec_conv3 = nn.ConvTranspose1d(512,128, kernel_size=4, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(32,16, kernel_size=3, stride=1)#, output_padding=1)
        self.dec_conv1 = nn.ConvTranspose2d(32,1, kernel_size=3, stride=1)

        self.dec_bn2 = nn.BatchNorm2d(16)
        # self.dec_bn3 = nn.BatchNorm1d(128)
        # self.dec_bn4 = nn.BatchNorm1d(256)
        # self.dec_bn5 = nn.BatchNorm1d(512)
        # self.dec_bn6 = nn.BatchNorm1d(512)
        # self.dec_bn7 = nn.BatchNorm1d(512)
        # self.dec_bn8 = nn.BatchNorm1d(512)

        # self.fc = nn.Linear(1000,1000)

    def forward(self, x):
        x1 = F.leaky_relu(self.enc_bn1(self.enc_conv1(x)), negative_slope=0.2)
        x2 = F.leaky_relu(self.enc_bn2(self.enc_conv2(x1)), negative_slope=0.2)
        # x3 = F.leaky_relu(self.enc_bn3(self.enc_conv3(x2)), negative_slope=0.2)
        # x4 = F.leaky_relu(self.enc_bn4(self.enc_conv4(x3)), negative_slope=0.2)
        # x5 = F.leaky_relu(self.enc_bn5(self.enc_conv5(x4)), negative_slope=0.2)
        # x6 = F.leaky_relu(self.enc_bn6(self.enc_conv6(x5)), negative_slope=0.2)
        # x7 = F.leaky_relu(self.enc_bn7(self.enc_conv7(x6)), negative_slope=0.2)
        # # x8 = F.leaky_relu(self.enc_bn8(self.enc_conv8(x7)), negative_slope=0.2)
        # x8 = F.leaky_relu(self.enc_conv8(x7), negative_slope=0.2)

        # xd1 = F.leaky_relu(self.dec_bn8(self.dec_conv8(x8)), negative_slope=0.2)
        # xd2 = F.leaky_relu(self.dec_bn7(self.dec_conv7(torch.cat([xd1,x7], dim=1))), negative_slope=0.2)
        # xd3 = F.leaky_relu(self.dec_bn6(self.dec_conv6(torch.cat([xd2,x6], dim=1))), negative_slope=0.2)
        # xd4 = F.leaky_relu(self.dec_bn5(self.dec_conv5(torch.cat([xd3,x5], dim=1))), negative_slope=0.2)
        # xd5 = F.leaky_relu(self.dec_bn4(self.dec_conv4(torch.cat([xd4,x4], dim=1))), negative_slope=0.2)
        # xd6 = F.leaky_relu(self.dec_bn3(self.dec_conv3(torch.cat([xd5,x3], dim=1))), negative_slope=0.2)
        xd1 = F.leaky_relu(self.dec_bn2(self.dec_conv2(x2)), negative_slope=0.2)
        xd2 = self.dec_conv1(torch.cat([xd1,x1], dim=1))
        # x_out = self.fc(xd2)
        return xd2

class SegNetSAR(nn.Module):
    def __init__(self):
        super(SegNetSAR,self).__init__()
        self.enc_conv1 = nn.Conv1d(1,64, kernel_size=4, stride=2)
        self.enc_conv2 = nn.Conv1d(64,128, kernel_size=4, stride=2)
        self.enc_conv3 = nn.Conv1d(128,256, kernel_size=4, stride=2)
        self.enc_conv4 = nn.Conv1d(256,512, kernel_size=4, stride=2)
        self.enc_conv5 = nn.Conv1d(512,512, kernel_size=4, stride=2)
        self.enc_conv6 = nn.Conv1d(512,512, kernel_size=4, stride=2)
        self.enc_conv7 = nn.Conv1d(512,512, kernel_size=4, stride=2)
        self.enc_conv8 = nn.Conv1d(512,512, kernel_size=4, stride=2)

        self.enc_bn1 = nn.BatchNorm1d(64)
        self.enc_bn2 = nn.BatchNorm1d(128)
        self.enc_bn3 = nn.BatchNorm1d(256)
        self.enc_bn4 = nn.BatchNorm1d(512)
        self.enc_bn5 = nn.BatchNorm1d(512)
        self.enc_bn6 = nn.BatchNorm1d(512)
        self.enc_bn7 = nn.BatchNorm1d(512)
        self.enc_bn8 = nn.BatchNorm1d(512)

        self.dec_conv8 = nn.ConvTranspose1d(512,512, kernel_size=4, stride=2, output_padding=1)
        self.dec_conv7 = nn.ConvTranspose1d(512,512, kernel_size=4, stride=2, output_padding=1)
        self.dec_conv6 = nn.ConvTranspose1d(512,512, kernel_size=4, stride=2, output_padding=1)
        self.dec_conv5 = nn.ConvTranspose1d(512,512, kernel_size=4, stride=2)
        self.dec_conv4 = nn.ConvTranspose1d(512,256, kernel_size=4, stride=2, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose1d(256,128, kernel_size=4, stride=2)
        self.dec_conv2 = nn.ConvTranspose1d(128,64, kernel_size=4, stride=2, output_padding=1)
        self.dec_conv1 = nn.ConvTranspose1d(64,1, kernel_size=4, stride=2)

        self.dec_bn2 = nn.BatchNorm1d(64)
        self.dec_bn3 = nn.BatchNorm1d(128)
        self.dec_bn4 = nn.BatchNorm1d(256)
        self.dec_bn5 = nn.BatchNorm1d(512)
        self.dec_bn6 = nn.BatchNorm1d(512)
        self.dec_bn7 = nn.BatchNorm1d(512)
        self.dec_bn8 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = F.leaky_relu(self.enc_bn1(self.enc_conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.enc_bn2(self.enc_conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.enc_bn3(self.enc_conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.enc_bn4(self.enc_conv4(x)), negative_slope=0.2)
        x = F.leaky_relu(self.enc_bn5(self.enc_conv5(x)), negative_slope=0.2)
        x = F.leaky_relu(self.enc_bn6(self.enc_conv6(x)), negative_slope=0.2)
        x = F.leaky_relu(self.enc_bn7(self.enc_conv7(x)), negative_slope=0.2)
        # x = F.leaky_relu(self.enc_bn8(self.enc_conv8(x)), negative_slope=0.2)
        x = F.leaky_relu(self.enc_conv8(x), negative_slope=0.2)

        x = F.leaky_relu(self.dec_bn8(self.dec_conv8(x)), negative_slope=0.2)
        x = F.leaky_relu(self.dec_bn7(self.dec_conv7(x)), negative_slope=0.2)
        x = F.leaky_relu(self.dec_bn6(self.dec_conv6(x)), negative_slope=0.2)
        x = F.leaky_relu(self.dec_bn5(self.dec_conv5(x)), negative_slope=0.2)
        x = F.leaky_relu(self.dec_bn4(self.dec_conv4(x)), negative_slope=0.2)
        x = F.leaky_relu(self.dec_bn3(self.dec_conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.dec_bn2(self.dec_conv2(x)), negative_slope=0.2)
        x = self.dec_conv1(x)
        return x
