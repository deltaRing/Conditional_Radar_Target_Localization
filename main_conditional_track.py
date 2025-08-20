import os

import torch
from tqdm import tqdm
from torch.optim import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from DataLoader.dataloader_multi import *
from DataLoader.dataloader_conditional_track import *

# Networks
from Network.SigmeseNetworks import *
from Network.SimilarityNetworks import *
from Network.Loss_Functions import *
from Network.U_Net import *
from Network.conditional_target_localization import *

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

batch_num  = 8
num_epochs = 100
threads    = 8
aFFT = 128
rFFT = 512
# train_datasets   = STFTIDDataset()
train_datasets   = ConditionalDataSets(False)
train_dataloader = DataLoader(train_datasets, batch_size=batch_num, shuffle=True, num_workers=threads)
test_datasets   = ConditionalDataSets(True)
test_dataloader = DataLoader(test_datasets, batch_size=batch_num, shuffle=True, num_workers=threads)
model_save_path = "./SavedModel/"
model_name            = "u_net_model-lite_9.pth"
sigmese_coder_model_name = "sigmese_net_model-lite_9.pth"
conditional_fuse_model_name = "conditional_net_model-lite_9.pth"

# result record
model_mse_diff = []
model_sim_diff = []
model_main_mse = []
model_main_sim = []
model_mse_diff_test = []
model_sim_diff_test = []
model_main_mse_test = []
model_main_sim_test = []
#
model_mse_diff_path     = "condition_mse_sim_diff_9.npy"
model_sim_diff_path     = "condition_mse_sim_9.npy"
model_main_mse_path     = "condition_main_mse_9.npy"
model_main_sim_path     = "condition_main_sim_9.npy"
model_mse_diff_path_test = "condition_mse_sim_diff_9_test.npy"
model_sim_diff_path_test = "condition_mse_sim_9_test.npy"
model_main_mse_path_test = "condition_main_mse_9_test.npy"
model_main_sim_path_test = "condition_main_sim_9_test.npy"

if os.path.exists(model_sim_diff_path):
    model_mse_diff = np.load(model_mse_diff_path).tolist()
    model_sim_diff = np.load(model_sim_diff_path).tolist()
    model_main_mse = np.load(model_main_mse_path).tolist()
    model_main_sim = np.load(model_main_sim_path).tolist()
    model_mse_diff_test = np.load(model_mse_diff_path_test).tolist()
    model_sim_diff_test = np.load(model_sim_diff_path_test).tolist()
    model_main_mse_test = np.load(model_main_mse_path_test).tolist()
    model_main_sim_test = np.load(model_main_sim_path_test).tolist()

# CONDITIONAL TARGET TRACK MODEL
ConditionTargetDetect = CTLNetworks().to(device)
# POSITIONAL TARGET TRACK MODEL
PositionMainLine      = PositionUNet(aFFT, rFFT, aFFT, rFFT).to(device).eval()
# TRIPLET LOSS
STFTTripletLoss   = TripletLoss().to(device).eval()
# FEATURE TARGET TRACK MODEL
STFTPositiveNet       = STFTNetwork().to(device).eval()

opti = Adam([{'params': ConditionTargetDetect.parameters()}], lr=1e-4)

if __name__ == '__main__':
    if os.path.exists(model_save_path + sigmese_coder_model_name):
        checkpoint = torch.load(model_save_path + sigmese_coder_model_name)
        STFTPositiveNet.load_state_dict(checkpoint['model_stft_feature'])

    if os.path.exists(model_save_path + model_name):
        checkpoint = torch.load(model_save_path + model_name)
        PositionMainLine.load_state_dict(checkpoint['model_position_mainline'])

    if os.path.exists(model_save_path + conditional_fuse_model_name):
        checkpoint = torch.load(model_save_path + conditional_fuse_model_name)
        ConditionTargetDetect.load_state_dict(checkpoint['model_conditional_framework'])


    for epoch in range(num_epochs):
        loss_mse_predict_coder = [] # for ConditionTargetDetect
        loss_cls_predict_coder = []
        loss_mse_main_line     = [] # for PositionMainLine
        loss_cls_main_line     = []
        ConditionTargetDetect = ConditionTargetDetect.train()
        for probs, prob_amp, ra_data, past_id, past_stft, \
                label_prob, label_amp, detect_prob, detect_stft, detected_id, tar_num in tqdm(train_dataloader, ncols=80, desc="Training"):

            # get past features
            sf = past_stft.to(device)
            _, _, _, past_hid, red_cls = STFTPositiveNet(sf)

            # get current prediction
            feat1, feat2, feat3, feat4, _, _ = PositionMainLine(ra_data.to(device))

            # detected features
            lists = []

            for tt in range(len(tar_num)):
                detected_past_hid = []
                detected_probs    = []
                detected_prob_where = []
                for ttt in range(tar_num[tt]):
                    sf = detect_stft[tt, ttt, :, :].to(device)
                    sf = sf.reshape(1, 1, 256, 212)
                    _, _, _, d_hid, _ = STFTPositiveNet(sf)

                    detected_past_hid.append(d_hid)
                    dp = detect_prob[tt, ttt, :, :]
                    max_index_flat = np.argmax(dp)
                    max_index_2d = np.unravel_index(max_index_flat, dp.shape)
                    detected_prob_where.append(max_index_2d)

                lists.append([
                    detected_past_hid,
                    detected_prob_where,
                ])

            pr = label_prob.to(device)
            pra = label_amp.to(device)
            prr = probs.to(device)
            prra = prob_amp.to(device)

            opti.zero_grad()

            # get the conditional postion results
            TarReLU, TarConf, CTarReLU, CTarConf = \
                ConditionTargetDetect(feat1, feat2, feat3, feat4, lists, tar_num.to(device), past_hid)
            pr = pr.to(torch.float32)
            prr = prr.to(torch.float32)
            pra = pra.to(torch.float32)
            prra = prra.to(torch.float32)

            loss_mse_pred = mse_loss(CTarReLU, pra)
            loss_cls_pred = cls_loss(CTarConf, pr)
            loss_mainline_mse_pred = mse_loss(TarReLU, prra)
            loss_mainline_cls_pred = cls_loss(TarConf, prr)

            loss_mse_predict_coder.append(loss_mse_pred.detach().cpu().numpy())
            loss_cls_predict_coder.append(loss_cls_pred.detach().cpu().numpy())
            loss_mse_main_line.append(loss_mainline_mse_pred.detach().cpu().numpy())
            loss_cls_main_line.append(loss_mainline_cls_pred.detach().cpu().numpy())

            loss_position_total = loss_cls_pred + 1e1 * loss_mse_pred + loss_mainline_cls_pred + 1e1 * loss_mainline_mse_pred
            loss_position_total.backward()
            opti.step()

        torch.save(
            {
                'model_conditional_framework': ConditionTargetDetect.state_dict()
            },
            model_save_path + conditional_fuse_model_name
        )

        print('Current Loss: %f', np.mean(np.array(loss_mse_predict_coder)),
              'Current CLS: %f', np.mean(np.array(loss_cls_predict_coder)),
              'Current MainLine Loss: %f', np.mean(np.array(loss_mse_main_line)),
              'Current MainLine CLS: %f', np.mean(np.array(loss_cls_main_line))
              )

        loss_mse_predict_coder_test = []  # for ConditionTargetDetect
        loss_cls_predict_coder_test = []
        loss_mse_main_line_test = []  # for PositionMainLine
        loss_cls_main_line_test = []
        ConditionTargetDetect = ConditionTargetDetect.eval()

        for probs, prob_amp, ra_data, past_id, past_stft, \
                label_prob, label_amp, detect_prob, detect_stft, detected_id, tar_num in tqdm(test_dataloader,
                                                                                              ncols=80,
                                                                                              desc="Testing"):

            sf = past_stft.to(device)
            _, _, _, past_hid, red_cls = STFTPositiveNet(sf)

            # get current prediction
            feat1, feat2, feat3, feat4, _, _ = PositionMainLine(ra_data.to(device))

            # detected features
            lists = []
            for tt in range(len(tar_num)):
                detected_past_hid = []
                detected_probs = []
                detected_prob_where = []
                for ttt in range(tar_num[tt]):
                    sf = detect_stft[tt, ttt, :, :].to(device)
                    sf = sf.reshape(1, 1, 256, 212)
                    _, _, _, d_hid, _ = STFTPositiveNet(sf)

                    detected_past_hid.append(d_hid)
                    dp = detect_prob[tt, ttt, :, :]
                    max_index_flat = np.argmax(dp)
                    max_index_2d = np.unravel_index(max_index_flat, dp.shape)
                    detected_prob_where.append(max_index_2d)

                lists.append([
                    detected_past_hid,
                    detected_prob_where,
                ])

            pr = label_prob.to(device)
            pra = label_amp.to(device)
            prr = probs.to(device)
            prra = prob_amp.to(device)

            # get the conditional postion results
            TarReLU, TarConf, CTarReLU, CTarConf = \
                ConditionTargetDetect(feat1, feat2, feat3, feat4, lists, tar_num.to(device), past_hid)

            # for ii in range(8):
            #     pred1 = CTarReLU.detach().cpu().numpy()
            #     pred2 = TarReLU.detach().cpu().numpy()
            #     pred3 = prra.detach().cpu().numpy()
            #     pred4 = pra.detach().cpu().numpy()
            #     fig, axs = plt.subplots(2, 2, figsize=(8, 8))
            #     for i, ax in enumerate(axs.flat):
            #         if i == 0:
            #             ax.imshow(np.squeeze(pred1[ii][0][:][:]), cmap='jet', interpolation='nearest', aspect='auto')
            #         elif i == 1:
            #             ax.imshow(np.squeeze(pred2[ii][0][:][:]), cmap='jet', interpolation='nearest', aspect='auto')
            #         elif i == 2:
            #             ax.imshow(np.squeeze(pred3[ii][0][:][:]), cmap='jet', interpolation='nearest', aspect='auto')
            #         elif i == 3:
            #             ax.imshow(np.squeeze(pred4[ii][0][:][:]), cmap='jet', interpolation='nearest', aspect='auto')
            #     plt.tight_layout()
            #     plt.show(block=True)


            pr = pr.to(torch.float32)
            prr = prr.to(torch.float32)
            pra = pra.to(torch.float32)
            prra = prra.to(torch.float32)

            loss_mse_pred = mse_loss(CTarReLU, pra)
            loss_cls_pred = cls_loss(CTarConf, pr)
            loss_mainline_mse_pred = mse_loss(TarReLU, prra)
            loss_mainline_cls_pred = cls_loss(TarConf, prr)

            loss_mse_predict_coder_test.append(loss_mse_pred.detach().cpu().numpy())
            loss_cls_predict_coder_test.append(loss_cls_pred.detach().cpu().numpy())
            loss_mse_main_line_test.append(loss_mainline_mse_pred.detach().cpu().numpy())
            loss_cls_main_line_test.append(loss_mainline_cls_pred.detach().cpu().numpy())

        print('Current Test Loss: %f', np.mean(np.array(loss_mse_predict_coder_test)),
              'Current Test CLS: %f', np.mean(np.array(loss_cls_predict_coder_test)),
              'Current Test MainLine Loss: %f', np.mean(np.array(loss_mse_main_line_test)),
              'Current Test MainLine CLS: %f', np.mean(np.array(loss_cls_main_line_test))
              )

        model_mse_diff_test.append(np.mean(np.array(loss_mse_predict_coder_test)))
        model_sim_diff_test.append(np.mean(np.array(loss_cls_predict_coder_test)))
        model_main_mse_test.append(np.mean(np.array(loss_mse_main_line_test)))
        model_main_sim_test.append(np.mean(np.array(loss_cls_main_line_test)))

        model_mse_diff.append(np.mean(np.array(loss_mse_predict_coder)))
        model_sim_diff.append(np.mean(np.array(loss_cls_predict_coder)))
        model_main_mse.append(np.mean(np.array(loss_mse_main_line)))
        model_main_sim.append(np.mean(np.array(loss_cls_main_line)))

        np.save(model_mse_diff_path, model_mse_diff)
        np.save(model_sim_diff_path, model_sim_diff)
        np.save(model_main_mse_path, model_main_mse)
        np.save(model_main_sim_path, model_main_sim)

        np.save(model_mse_diff_path_test, model_mse_diff_test)
        np.save(model_sim_diff_path_test, model_sim_diff_test)
        np.save(model_main_mse_path_test, model_main_mse_test)
        np.save(model_main_sim_path_test, model_main_sim_test)
