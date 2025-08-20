import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

class Encoder(nn.Module):
    def __init__(self,
                 sz=32, sx=256, sy=64):
        super(Encoder, self).__init__()
        self.sx=sx
        self.sy=sy
        self.sz=sz
        self.enc_mu = nn.Sequential(
            nn.Conv2d(sz, sz, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.BatchNorm2d(sz),
        )

        self.enc_var = nn.Sequential(
            nn.Conv2d(sz, sz, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.BatchNorm2d(sz),
        )

    def forward(self, feat):
        mu = self.enc_mu(feat)
        var = self.enc_var(feat)

        return mu, var

class Decoder(nn.Module):
    def __init__(self,
                 sz=32, sx=256, sy=64,
                 dcz=1, dcx=512, dcy=128):
        super(Decoder, self).__init__()
        self.sx=sx
        self.sy=sy
        self.sz=sz
        self.dcz=dcz
        self.dcx=dcx
        self.dcy=dcy

        self.up1 = nn.Upsample([sx * 2, sy * 2])
        self.dec_loc = nn.Sequential(
            nn.Conv2d(sz, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        self.dec_feat = nn.Sequential(
            nn.Conv2d(sz, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(16, dcz, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(dcz),
        )

    def forward(self, z_loc, z_feat1, z_feat2):
        f_loc = self.dec_loc(z_loc)
        f_feat = self.dec_feat(torch.cat([z_feat1, z_feat2], dim=1))
        f1 = self.dec1(torch.cat([f_loc, f_feat], dim=1))
        f2 = self.up1(f1)
        f3 = self.dec2(f2)
        dec_res = self.dec3(f3)
        return dec_res

class CTLNetworks(nn.Module):
    def __init__(self, sx=1, sy=64, sz=16,
                 ex=512, ey=128, ez=1,
                 fx=512, fy=128, fz=16):
        super(CTLNetworks, self).__init__()
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.ex = ex
        self.ey = ey
        self.ez = ez
        self.fx = fx
        self.fy = fy
        self.fz = fz

        # upgrade the concated faetures-----
        self.up1 = nn.Upsample([64, 16])
        self.up2 = nn.Upsample([64, 16])

        self.ConcatUpgradeNet1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.ConcatUpgradeNet2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.ConcatUpgradeNet3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # downgrade the predicted features
        self.PredictUpSample1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.PredictUpSample2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.PredictUpSample3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.reshapeNet = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # reshape ---> 128 x 32 x 64
        self.p1 = nn.Upsample([64, 16])
        self.p2 = nn.Upsample([128, 32])
        self.p3 = nn.Upsample([256, 64])

        self.ReLU = torch.nn.ReLU()
        self.Sigmoid = torch.nn.Sigmoid()
        # reshape ----> 512 x 128 x 1

        self.EncFeat1 = Encoder(sz=16)
        self.EncFeat2 = Encoder(sz=16)
        self.EncLoc   = Encoder(sz=32)
        self.DecTrue  = Decoder()
        self.DecFalse = Decoder()

    # batchs x 1024
    def forward(self, feat1, feat2, feat3, feat4, current_stfts, target_num, desire_stft):
        ra_feat1 = self.PredictUpSample1(feat4)
        ra_feat2 = self.ReLU(self.p1(ra_feat1) + feat3)
        ra_feat3 = self.PredictUpSample2(ra_feat2)
        ra_feat4 = self.ReLU(self.p2(ra_feat3) + feat2) # 1 x 128 x 32
        ra_feat5 = self.PredictUpSample3(ra_feat4)
        ra_feat6 = self.ReLU(self.p3(ra_feat5) + feat1)

        feat_current_dim = torch.zeros([ra_feat6.shape[0], 16, 256, 64]).cuda() + 1e-3 * torch.randn([ra_feat6.shape[0], 16, 256, 64]).cuda()
        feat_desire_dim = torch.zeros([ra_feat6.shape[0], 16, 256, 64]).cuda() + 1e-3 * torch.randn([ra_feat6.shape[0], 16, 256, 64]).cuda()

        # the desire target single
        desire_stft_feat = desire_stft.view(len(current_stfts), 1, 32, 8)
        feat_d = self.ConcatUpgradeNet1(desire_stft_feat)
        feat_d = self.up1(feat_d)
        feat_d = self.ConcatUpgradeNet2(feat_d)
        feat_d = self.up2(feat_d)
        feat_d = self.ConcatUpgradeNet3(feat_d)

        for ii in range(len(current_stfts)):
            # probs
            # feats
            # ra
            # desire_stft feats
            for iii in range(target_num[ii]):
                current_feats = current_stfts[ii][0][iii].view(1, 1, 32, 8)
                current_locs = np.round(np.array(current_stfts[ii][1][iii]) / 2)
                feat_c = self.ConcatUpgradeNet1(current_feats)
                feat_c = self.up1(feat_c)
                feat_c = self.ConcatUpgradeNet2(feat_c)
                feat_c = self.up2(feat_c)
                feat_c = self.ConcatUpgradeNet3(feat_c)

                ix = int(current_locs[0] - 16)
                iy = int(current_locs[1] - 4)
                mx = int(current_locs[0] + 16)
                my = int(current_locs[1] + 4)

                # check codes
                if ix < 0:
                    ix = 0
                if iy < 0:
                    iy = 0
                if mx > 256:
                    ix = 255 - 32
                if my > 64:
                    iy = 63 - 8
                feat_current_dim[ii, :, ix:ix+32, iy:iy+8] = feat_c
                feat_desire_dim[ii, :, ix:ix+32, iy:iy+8]  = feat_d[ii, :, :, :]

        mu_loc, var_loc = self.EncLoc(ra_feat6)
        mu_des, var_des = self.EncFeat1(feat_desire_dim)
        mu_cur, var_cur = self.EncFeat2(feat_current_dim)

        # fit the results
        z_loc = reparameterize(mu_loc, var_loc)
        z_des = reparameterize(mu_des, var_des)
        z_cur = reparameterize(mu_cur, var_cur)

        ori_pred = self.DecTrue(z_loc, z_cur, z_cur)
        if np.random.randint(2) == 1:
            des_pred = self.DecFalse(z_loc, z_cur, z_des)
        else:
            des_pred = self.DecFalse(z_loc, z_des, z_cur)

        if torch.isnan(ori_pred).any() or torch.isnan(des_pred).any():
            print()

        return self.ReLU(ori_pred), self.Sigmoid(ori_pred), \
                self.ReLU(des_pred), self.Sigmoid(des_pred),