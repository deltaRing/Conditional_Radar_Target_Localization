import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# mirage codes of pseudo stft networks
# pseudo stft network
class PseudoSTFTNetwork(nn.Module):
    def __init__(self, ex=23, ey=64, used_frames=32, batch=32, sx=128, sy=512, depth=1, used_num=15):
        super(PseudoSTFTNetwork, self).__init__()

        # the fake stft feature
        self.batch = batch            # Batch Size
        self.sx    = sx               # input size X
        self.sy    = sy               # input size Y
        self.sz    = depth            # input size Z
        self.UsedUnits = used_num     # Find round Units

        # the fake reshaped stft feature
        self.upsample = nn.Upsample([ex, ey])
        self.downsample = nn.Upsample([sx // 4, sy // 4])
        self.dx = sx // 4
        self.dy = sy // 4
        self.epx = ex
        self.epy = ey
        self.UsedFrames = used_frames # expected input frames
        self.Lr  = 5
        self.La  = 3

        # pseudo Feature Nets ---> fx=23, fy=64, fz=128
        self.PseudoFeature = nn.Sequential(
            nn.Conv2d(used_frames, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128 * 3, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128 * 3),

            nn.Conv2d(128 * 3, 128 * 3, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128 * 3),
        )

    def ROIareaDetect(self, multi_frames, num_tar, thres=0.85):
        detect = []
        # iterate and detect
        data = []
        for ii in range(len(multi_frames)):
            if len(data) == 0:
                data = multi_frames[ii]
            else:
                data += multi_frames[ii]
        data /= len(multi_frames)
        # data = data.reshape([1, 1, 512, 128])
        # data = self.downsample(data)
        data = torch.squeeze(data)
        coords = np.argwhere(data.detach().cpu().numpy() > thres)  # 返回的是(y, x)格式

        # if coords.length

        clustering = DBSCAN(eps=5, min_samples=15).fit(coords)

        for ii in range(num_tar):
            result = np.zeros([1, 2])
            lindex = 0
            for jj in range(len(clustering.labels_)):
                if clustering.labels_[jj] == ii:
                    result += coords[jj, :]
                    lindex += 1
            det = result / lindex

            w = self.epy
            h = self.epx
            x = -1
            y = -1
            if det[0][0] - self.epy // 2 < 0:
                x = 0
            elif det[0][0] + self.epy >= self.sx:
                x = det[0][0] - self.epy - 1
            else:
                x = det[0][0] - self.epy // 2

            if det[0][1] - self.epx // 2 < 0:
                y = 0
            elif det[0][1] + self.epx >= self.sy:
                y = det[0][1] - self.epx - 1
            else:
                y = det[0][1] - self.epx // 2

            det = [int(x), int(y), int(w), int(h)]

            detect.append(det)

        return detect

    def forward(self, multi_frames, detect_conf, num_tar = 1):
        features = []
        # detect ROI area at first

        for ii in range(detect_conf.shape[0]):
            feat = []
            det  = detect_conf[ii]
            detected_area = self.ROIareaDetect(det, num_tar)
            for tt in range(len(detected_area)):
                det = detected_area[tt]
                for iii in range(detect_conf.shape[1]):
                    ft = multi_frames[ii][iii][det[0]:det[0] + det[2], det[1]: det[1] + det[3]]
                    ft = ft.reshape(1, self.epy, self.epx)
                    if len(feat) == 0:
                        feat = ft
                    else:
                        feat = torch.cat((feat, ft), dim=0)
            feat = feat.reshape(1, detect_conf.shape[1], self.epy, self.epx)
            if len(features) == 0:
                features = feat
            else:
                features = torch.cat((features, feat), dim=0)

        pseudo_features = self.PseudoFeature(features)
        return pseudo_features



class STFTNetwork(nn.Module):
    def __init__(self, batch=8, sx=256, sy=128, num_class=5, used_num=15):
        super(STFTNetwork, self).__init__()

        # Focus on Lower Feature
        # -50 -25 -10 ... and more
        # Focus on Middle Feature
        # -5   0   5 ... and more
        # Focus on Higher Feature
        # 10   25  50 ... and more

        # 512 x 187 x 15
        # STFT size
        self.num_class = num_class
        self.batch = batch
        self.sx0   = sx
        self.sy0   = sy
        self.sx1   = self.sx0 // 2
        self.sy1   = self.sy0 // 2
        self.sx2   = self.sx1 // 2
        self.sy2   = self.sy1 // 2
        self.sx3   = self.sx2 // 2
        self.sy3   = self.sy2 // 2

        ## ============================= Another Module ==================================
        #  -----------> | | ---> Get Units ----------------|
        #  -------> |     |         |                      |
        #  ----> |        |         |                      |
        #                 |        \/                      |
        #                 |       Around 9 Units---------> compare similarity
        #
        #
        #
        # The Below Networks Can be Used for multiple blocks
        self.UsedUnits = used_num
        # Single CoderNets Network Using for predict stft features
        self.CoderNets1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
        )

        self.CoderNets2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
        )

        self.CoderNets3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
        )  # coding to hyperspace

        self.CoderNets4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
        )  # coding to hyperspace

        self.U3 = nn.Upsample(size=[self.sx0, self.sy0])
        self.U2 = nn.Upsample(size=[self.sx1, self.sy1])
        self.U1 = nn.Upsample(size=[self.sx2, self.sy2])
        self.U0 = nn.Upsample(size=[self.sx3, self.sy3]) # GRAM is booming

        # Single Decoder Network Using for decoding stft features
        self.DeCoderNet0 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

        self.DeCoderNet1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        self.DeCoderNet2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.DeCoderNet3 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
        )

        self.MReLU = nn.ReLU()

        self.Reduction = nn.Sequential(
            nn.Linear(16384, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 256), # reshape-> 1 x 64 x 16
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )

        self.Reduction_Classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.num_class),
            nn.ReLU(),
        )

    def forward(self, STFT):
        hid_codes  = []
        red_codes  = []
        pred_stft  = []
        labels     = []
        # Extract STFT Features
        randstart = np.random.randint(0, 212 - 128)
        tSTFT = STFT[:, :, :, randstart:randstart+128]
        # batch_num = STFT.shape[0]
        # tSTFT = tSTFT.reshape(batch_num, 1, self.sx0, self.sy0)

        hc1 = self.CoderNets1(tSTFT)
        hc2 = self.CoderNets2(hc1)
        hc3 = self.CoderNets3(hc2)
        hc4 = self.CoderNets4(hc3)
        h0 = self.MReLU(self.DeCoderNet0(self.U0(hc4)) + hc3)
        h1 = self.MReLU(self.DeCoderNet1(self.U1(h0)) + hc2)
        h2 = self.MReLU(self.DeCoderNet2(self.U2(h1)) + hc1)
        h3 = self.MReLU(self.DeCoderNet3(self.U3(h2)))

        if len(hid_codes) == 0 or len(pred_stft) == 0:
            hid_codes = hc4
            pred_stft = h3
            labels = tSTFT
        else:
            hid_codes = torch.cat((hid_codes, hc4), dim=1)
            pred_stft = torch.cat((pred_stft, h3), dim=1)
            labels = torch.cat((labels, tSTFT), dim=1)

        batch_size = hid_codes.shape[0]
        codes = hid_codes.reshape(batch_size, -1)
        reduc = self.Reduction(codes)
        clsres = self.Reduction_Classifier(reduc)

        return hid_codes, pred_stft, labels, reduc, clsres

if __name__ == '__main__':
    stftfeature = STFTNetwork()
