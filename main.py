import os

import torch
from tqdm import tqdm
from torch.optim import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from DataLoader.dataloader_multi import *
# from Network.Sigmese_Networks_Test import *
from Network.Loss_Functions import *
from Network.U_Net import *

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

batch_num  = 16
num_epochs = 100
threads    = 8
train_datasets   = DatasetsName(False)
train_dataloader = DataLoader(train_datasets, batch_size=batch_num, shuffle=True, num_workers=threads)
test_datasets   = DatasetsName(True)
test_dataloader = DataLoader(test_datasets, batch_size=batch_num, shuffle=True, num_workers=threads)
model_save_path = "./SavedModel/"
model_name      = "u_net_model-lite_9.pth"

amp  = 1e3
aFFT = 128
rFFT = 512
model_loss_mse = []
model_loss_cls = []
model_loss_mse_test = []
model_loss_cls_test = []

model_loss_mse_save_path = '5-target/mse_9.npy'
model_loss_cls_save_path = '5-target/cls_9.npy'
model_loss_mse_save_path_test = '5-target/mse_9_test.npy'
model_loss_cls_save_path_test = '5-target/cls_9_test.npy'
if os.path.exists(model_loss_mse_save_path):
    model_loss_mse = np.load(model_loss_mse_save_path).tolist() # first 0.8 last 0.6666666667
    model_loss_cls = np.load(model_loss_cls_save_path).tolist()
    model_loss_mse_test = np.load(model_loss_mse_save_path_test).tolist()
    model_loss_cls_test = np.load(model_loss_cls_save_path_test).tolist()

# define the networks
PositionMainLine = PositionUNet(aFFT, rFFT, aFFT, rFFT).to(device)

# set up the optimizer
opti = Adam([{'params': PositionMainLine.parameters()}], lr=1e-3)

if __name__ == '__main__':
    if os.path.exists(model_save_path + model_name):
        checkpoint = torch.load(model_save_path + model_name)
        PositionMainLine.load_state_dict(checkpoint['model_position_mainline'])

    for epoch in range(num_epochs):
        loss_mse = []
        loss_cls = []
        loss_mse_test = []
        loss_cls_test = []

        PositionMainLine = PositionMainLine.train()

        for _, _, prob_total, ra_data, _ in tqdm(train_dataloader, ncols=80, desc="Training"):
            pr = prob_total.to(device)
            ra = ra_data.to(device)
            # sf = sf.to(device)
            pr_mse = pr * amp

            opti.zero_grad()

            feat1, feat2, feat3, feat4, TarConf, TarReLU = PositionMainLine(ra)

            # pred = pr.detach().cpu().numpy()
            # plt.imshow(np.squeeze(pred[0][0][:][:]), cmap='hot', interpolation='nearest', aspect='auto')
            # plt.show(block=True)
            #
            # pred = ra_data.detach().cpu().numpy()
            # plt.imshow(np.squeeze(pred[0][0][:][:]), cmap='hot', interpolation='nearest', aspect='auto')
            # plt.show(block=True)

            loss_cls_pred = cls_loss(TarConf, pr)
            loss_mse_pred = mse_loss(TarReLU, pr_mse)

            loss_mse.append(loss_mse_pred.detach().cpu().numpy())
            loss_cls.append(loss_cls_pred.detach().cpu().numpy())

            loss_position_total = loss_cls_pred + 1e2 * loss_mse_pred
            loss_position_total.backward()
            opti.step()

        torch.save(
            {
                'model_position_mainline': PositionMainLine.state_dict()
            },
            model_save_path + model_name
        )

        print('Current Loss: %f', np.mean(np.array(loss_mse)),
              'Current CLS: %f', np.mean(np.array(loss_cls)))


        PositionMainLine = PositionMainLine.eval()

        for _, _, prob_total, ra_data, _ in tqdm(test_dataloader, ncols=80, desc="Testing"):
            pr = prob_total.to(device)
            ra = ra_data.to(device)
            pr_mse = pr * amp

            feat1, feat2, feat3, feat4, TarConf, TarReLU = PositionMainLine(ra)
            loss_cls_pred_test = cls_loss(TarConf, pr)
            loss_mse_pred_test = mse_loss(TarReLU, pr_mse)
            loss_mse_test.append(loss_mse_pred_test.detach().cpu().numpy())
            loss_cls_test.append(loss_cls_pred_test.detach().cpu().numpy())

            # pred = pr.detach().cpu().numpy()
            # plt.imshow(np.squeeze(pred[0][0][:][:]), cmap='hot', interpolation='nearest', aspect='auto')
            # plt.show(block=True)
            #
            # pred = ra_data.detach().cpu().numpy()
            # plt.imshow(np.squeeze(pred[0][0][:][:]), cmap='hot', interpolation='nearest', aspect='auto')
            # plt.show(block=True)

        print('Current Test Loss: %f', np.mean(np.array(loss_mse_test)),
              'Current Test CLS: %f', np.mean(np.array(loss_cls_test)))

        model_loss_mse.append(np.mean(np.array(loss_mse)))
        model_loss_cls.append(np.mean(np.array(loss_cls)))
        model_loss_mse_test.append(np.mean(np.array(loss_mse_test)))
        model_loss_cls_test.append(np.mean(np.array(loss_cls_test)))
        np.save(model_loss_mse_save_path, model_loss_mse)
        np.save(model_loss_cls_save_path, model_loss_cls)
        np.save(model_loss_mse_save_path_test, model_loss_mse_test)
        np.save(model_loss_cls_save_path_test, model_loss_cls_test)