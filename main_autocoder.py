import os

import torch
from tqdm import tqdm
from torch.optim import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from DataLoader.dataloader import *
# from Network.Sigmese_Networks_Test import *
from Network.SimilarityNetworks import *
from Network.Loss_Functions import *
from Network.U_Net import *

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

batch_num  = 8
num_epochs = 100
threads    = 8
train_datasets   = STFTPROBDataset()
train_dataloader = DataLoader(train_datasets, batch_size=batch_num, shuffle=True, num_workers=threads)
model_save_path = "./SavedModel/"
model_name      = "u_net_model-lite.pth"
auto_coder_model_name = "auto_net_model-lite.pth"
pseudo_coder_model_name = "pseudo_net_model-lite.pth"

aFFT = 128
rFFT = 512
PositionMainLine = PositionUNet(aFFT, rFFT, aFFT, rFFT).to(device)
PseudoNegativeNet = PseudoSTFTNetwork().to(device)
STFTPositiveNet   = STFTNetwork().to(device)

# set up the optimizer
opti = Adam([{'params': STFTPositiveNet.parameters()}], lr=1e-3)

#
model_mse_diff = []
# model_kl_sim   = []
# model_mse_sim  = []
#
model_mse_diff_path = "mse_diff.npy"
# model_kl_sim_path = "model_kl_sim.npy"
# model_mse_sim_path = "model_mse_sim.npy"

if os.path.exists(model_mse_diff_path):
    model_mse_diff = np.load(model_mse_diff_path).tolist()
    # model_kl_sim   = np.load(model_kl_sim_path).tolist()
    # model_mse_sim   = np.load(model_mse_sim_path).tolist()

if __name__ == '__main__':
    if os.path.exists(model_save_path + auto_coder_model_name):
        checkpoint = torch.load(model_save_path + auto_coder_model_name)
        STFTPositiveNet.load_state_dict(checkpoint['model_stft_feature'])
        # PseudoNegativeNet.load_state_dict(checkpoint['model_pseudo_feature'])

    # if os.path.exists(model_save_path + model_name):
    #     checkpoint = torch.load(model_save_path + model_name)
    #     PositionMainLine.load_state_dict(checkpoint['model_position_mainline'])
    #     PositionMainLine = PositionMainLine.eval()

    for epoch in range(num_epochs):
        loss_mse = []
        loss_sim_mse = []
        loss_sim_kl  = []

        for _, _, sf in tqdm(train_dataloader, ncols=80, desc="Training"):
            # ra = ra.to(device)
            sf = sf.to(device)
            sf = sf.permute(0, 3, 1, 2)

            # ra_fake = []
            # ra_conf = []
            # # get feature from
            # for ii in range(ra.shape[1]):
            #     ra_i = ra[:, ii, :, :]
            #     ra_i = ra_i.reshape([batch_num, 1, rFFT, aFFT])
            #     TarConf, TarReLU = PositionMainLine(ra_i)
            #     ra_i = ra_i * TarConf # fake feature
            #     # pred = ra_i.detach().cpu().numpy()
            #     # plt.imshow(np.squeeze(pred[0][0][:][:]), cmap='hot', interpolation='nearest', aspect='auto')
            #     # plt.show(block=True)
            #
            #     if len(ra_fake) == 0:
            #         ra_fake = ra_i
            #         ra_conf = TarConf
            #     else:
            #         ra_fake = torch.cat((ra_fake, ra_i), dim=1)
            #         ra_conf = torch.cat((ra_conf, TarConf), dim=1)

            opti.zero_grad()

            # fake features
            # pseudo_features = PseudoNegativeNet(ra_fake, ra_conf)
            # real features
            hid_codes, pred_stft, labels = STFTPositiveNet(sf)
            # predictions # we will check the central one
            loss_mse_pred = mse_loss(pred_stft, labels)
            # check distribution  # we will check the central one
            # loss_kl_fake  = kl_loss(pseudo_features, hid_codes)
            # check mse
            # loss_mse_fake = mse_loss(pseudo_features, hid_codes)

            # pred1 = sf.detach().cpu().numpy()
            # pred2 = pred_stft.detach().cpu().numpy()
            # plt.imshow(np.squeeze(pred1[0][3][:][:]), cmap='hot', interpolation='nearest', aspect='auto')
            # plt.show(block=True)
            # plt.imshow(np.squeeze(pred2[0][3][:][:]), cmap='hot', interpolation='nearest', aspect='auto')
            # plt.show(block=True)

            loss_mse.append(loss_mse_pred.detach().cpu().numpy())
            # loss_sim_mse.append(loss_mse_fake.detach().cpu().numpy())
            # loss_sim_kl.append(loss_kl_fake.detach().cpu().numpy())

            loss_position_total = loss_mse_pred # + loss_mse_fake + 10 * loss_kl_fake
            loss_position_total.backward()
            opti.step()

        torch.save(
            {
                'model_stft_feature': STFTPositiveNet.state_dict(),
                # 'model_pseudo_feature': PseudoNegativeNet.state_dict()
            },
            model_save_path + auto_coder_model_name
        )

        print('Current Loss: %f', np.mean(np.array(loss_mse))) # , "Current KL ", np.mean(np.array(loss_sim_kl)), "Current Sim MSE:", np.mean(np.array(loss_sim_mse)))

        model_mse_diff.append(np.mean(np.array(loss_mse)))
        # model_kl_sim.append(np.mean(np.array(loss_sim_kl)))
        # model_mse_sim.append(np.mean(np.array(loss_sim_mse)))
        np.save(model_mse_diff_path, model_mse_diff)
        # np.save(model_kl_sim_path, model_kl_sim)
        # np.save(model_mse_sim_path, model_mse_sim)