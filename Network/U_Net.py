import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 1、 必须很简单
# 2、 必须很高效
# 3、
# 4、
class PositionUNet(nn.Module):
    def __init__(self, size_x, size_y, aFFT, rFFT):
        super(PositionUNet, self).__init__()

        # Focus on Lower Feature
        # -50 -25 -10 ... and more
        # Focus on Middle Feature
        # -5   0   5 ... and more
        # Focus on Higher Feature
        # 10   25  50 ... and more

        self.size_x = size_x
        self.size_y = size_y
        self.NanFFT = aFFT
        self.NranFFT = rFFT

        # up/down sampled size
        self.sx_0 = size_x
        self.sy_0 = size_y
        self.sx_1 = size_x // 2
        self.sy_1 = size_y // 2
        self.sx_2 = size_x // 4
        self.sy_2 = size_y // 4
        self.sx_3 = size_x // 8
        self.sy_3 = size_y // 8
        self.sx_4 = size_x // 16
        self.sy_4 = size_y // 16

        # for localization as main frame
        self.LocalizationArch1 = nn.Sequential(
            # | ===========================> | upsampling
            #   | ====================> |
            #      | ==============> |
            nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
        )

        self.LocalizationArch2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
        )

        self.LocalizationArch3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(128),
        )

        self.LocalizationArch4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(256),
        )

        self.PosiOperator0 = nn.Upsample(
            size=[self.sy_0, self.sx_0]
        )

        self.PosiOperator1 = nn.Upsample(
            size=[self.sy_1, self.sx_1]
        )

        self.PosiOperator2 = nn.Upsample(
            size=[self.sy_2, self.sx_2]
        )

        self.PosiOperator3 = nn.Upsample(
            size=[self.sy_3, self.sx_3]
        )

        self.PosiOperator4 = nn.Upsample(
            size=[self.sy_4, self.sx_4]
        )

        self.PositArch1 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(1),
        )

        self.PositArch2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        self.PositArch3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        self.PositArch4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.end_of_line = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1),
        )
        self.MLReLU = torch.nn.LeakyReLU()
        self.FConf  = torch.nn.Sigmoid()
        self.MReLU  = torch.nn.ReLU()

    def forward(self, RA):
        # Loops
        Loop1 = self.LocalizationArch1(RA)
        Loop2 = self.LocalizationArch2(Loop1)
        Loop3 = self.LocalizationArch3(Loop2)
        Loop4 = self.LocalizationArch4(Loop3)
        # Get DeLoop Data
        Deloop1 = self.MLReLU(self.PosiOperator3(self.PositArch4(Loop4)) + Loop3)
        Deloop2 = self.MLReLU(self.PosiOperator2(self.PositArch3(Deloop1)) + Loop2)
        Deloop3 = self.MLReLU(self.PosiOperator1(self.PositArch2(Deloop2)) + Loop1)
        Deloop4 = self.end_of_line(self.PosiOperator0(self.PositArch1(Deloop3))) # self.end_of_line(self.PosiOperator0(self.PositArch1(Deloop3)) + RA)
        # Get Confidance Map
        TarConf = self.FConf(Deloop4)
        TarReLU = self.MReLU(Deloop4)

        return Loop1, Loop2, Loop3, Loop4, TarConf, TarReLU
