import os

import torch
from tqdm import tqdm
from torch.optim import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from DataLoader.dataloader_multi import *
from Network.SigmeseNetworks import *
from Network.SimilarityNetworks import *
from Network.Loss_Functions import *
from Network.U_Net import *

def apply_tsne(data, id, n_components=2, plot=True):
    """
    对形状为 [batches, 384, 32, 11] 的数据应用 t-SNE 降维

    参数:
        data (np.ndarray): 原始数据，形状为 [batches, 384, 32, 11]
        n_components (int): 降维后的维度，通常为 2 或 3
        perplexity (float): t-SNE 的超参数之一，影响聚类效果
        random_state (int): 随机种子，保证可复现性
        plot (bool): 是否画图（仅当 n_components=2 时）

    返回:
        tsne_result (np.ndarray): 降维后的数据，形状为 [batches, n_components]
    """
    batches = data.shape[0]
    flattened_data = data.reshape(batches, -1)  # shape: [batches, 384*32*11]

    # 标准化每个特征
    tsne = TSNE(n_components=2, random_state=3)
    result = tsne.fit_transform(flattened_data)

    # 可视化（二维）
    if plot and n_components == 2:
        plt.figure(figsize=(8, 6))
        for label in np.unique(id):
            idx = id == label
            plt.scatter(result[idx, 0], result[idx, 1], label=f'Class {label}')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('t-SNE Visualization')
        plt.legend()
        plt.grid(True)
        plt.show(block=True)

    return result

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

batch_num  = 16
num_epochs = 100
threads    = 8
# train_datasets   = STFTIDDataset()
train_datasets   = DatasetsName(False)
train_dataloader = DataLoader(train_datasets, batch_size=batch_num, shuffle=True, num_workers=threads)
test_datasets   = DatasetsName(True)
test_dataloader = DataLoader(test_datasets, batch_size=batch_num, shuffle=True, num_workers=threads)
model_save_path = "./SavedModel/"
model_name            = "u_net_model-lite_9.pth"
auto_coder_model_name = "auto_net_model-lite_9.pth"
pseudo_coder_model_name = "pseudo_net_model-lite_9.pth"
sigmese_coder_model_name = "sigmese_net_model-lite_9.pth"

STFTPositiveNet   = STFTNetwork().to(device)
STFTCenterLoss    = CenterLoss().to(device)
STFTTripletLoss   = TripletLoss().to(device)

# set up the optimizer
opti = Adam([{'params': STFTPositiveNet.parameters()}, {'params': STFTCenterLoss.parameters()}], lr=5e-4)
#
model_sim_diff = []
model_cls_diff = []
model_cls_acc  = []
model_mse_diff = []
model_self_ct_diff = []

model_sim_diff_test = []
model_cls_diff_test = []
model_cls_acc_test  = []
model_mse_diff_test = []
model_self_ct_diff_test = []
#
model_mse_diff_path     = "5-target/mse_sim_diff_9.npy"
model_sim_cls_path      = "5-target/cls_sim_9.npy"
model_sim_cls_acc_path  = "5-target/cls_acc_9.npy"
model_sim_diff_path     = "5-target/mse_sim_9.npy"
model_self_ct_diff_path = "5-target/ct_sim_9.npy"

model_mse_diff_path_test     = "5-target/mse_sim_diff_9_test.npy"
model_sim_cls_path_test      = "5-target/cls_sim_9_test.npy"
model_sim_cls_acc_path_test  = "5-target/cls_acc_9_test.npy"
model_sim_diff_path_test     = "5-target/mse_sim_9_test.npy"
model_self_ct_diff_path_test = "5-target/ct_sim_9_test.npy"
#
if os.path.exists(model_sim_diff_path):
    model_mse_diff = np.load(model_mse_diff_path).tolist()
    model_cls_diff = np.load(model_sim_cls_path).tolist()
    model_cls_acc  = np.load(model_sim_cls_acc_path).tolist()
    model_sim_diff = np.load(model_sim_diff_path).tolist()
    model_self_ct_diff = np.load(model_self_ct_diff_path).tolist()

if __name__ == '__main__':
    if os.path.exists(model_save_path + sigmese_coder_model_name):
        checkpoint = torch.load(model_save_path + sigmese_coder_model_name)
        STFTPositiveNet.load_state_dict(checkpoint['model_stft_feature'])
        STFTCenterLoss.load_state_dict(checkpoint['model_center_feature'])
        STFTTripletLoss.load_state_dict(checkpoint['model_triplet_feature'])

    feats = []
    ids = []

    for epoch in range(num_epochs):
        loss_mse = []
        loss_cls = []
        acc_cls  = []
        loss_sim = []
        loss_sim_ct  = []

        loss_mset = []
        loss_clst = []
        acc_clst = []
        loss_simt = []
        loss_sim_ctt = []

        STFTPositiveNet = STFTPositiveNet.train()
        STFTCenterLoss = STFTCenterLoss.train()
        STFTTripletLoss = STFTTripletLoss.train()

        # for sf, id in tqdm(train_dataloader, ncols=80, desc="Training"):
        for _, stft_data, _, _, id in tqdm(train_dataloader, ncols=80, desc="Training"):
            # ra = ra.to(device)
            sf = stft_data.to(device)
            sf = sf.reshape(-1, 1, 256, 212)
            # sf = sf.permute(0, 3, 1, 2)
            opti.zero_grad()

            hid_codes, pred_stft, labels, reducted_code, red_cls = STFTPositiveNet(sf)
            triplet_loss = STFTTripletLoss(reducted_code, id)
            center_loss = STFTCenterLoss(reducted_code, id)
            loss_mse_pred = mse_loss(pred_stft, labels)
            loss_cls_pred = clsfi_loss(red_cls, id.to(device)-1)
            _, predicted = torch.max(red_cls, dim=1)
            correct = (predicted == id.to(device)-1).sum().item()
            accuracy = correct / id.size(0)

            # if len(feats) == 0:
            #     feats = reducted_code.detach().cpu()
            #     ids   = id.detach().cpu()
            # else:
            #     feats = torch.cat([feats, reducted_code.detach().cpu()])
            #     ids   = torch.cat([ids, id.detach().cpu()])
            #
            #     if feats.shape[0] > 2048:
            #         apply_tsne(feats, ids)

            loss_cls.append(loss_cls_pred.detach().cpu().numpy())
            acc_cls.append(accuracy)
            loss_mse.append(loss_mse_pred.detach().cpu().numpy())
            loss_sim.append(triplet_loss.detach().cpu().numpy())
            loss_sim_ct.append(center_loss.detach().cpu().numpy())

            loss_position_total = 1e2 * loss_mse_pred + triplet_loss + center_loss + loss_cls_pred
            loss_position_total.backward()
            opti.step()

        torch.save(
            {
                'model_stft_feature': STFTPositiveNet.state_dict(),
                'model_center_feature': STFTCenterLoss.state_dict(),
                'model_triplet_feature': STFTTripletLoss.state_dict(),
            },
            model_save_path + sigmese_coder_model_name
        )

        print('Current Loss: %f', np.mean(np.array(loss_mse)),
              "Current Triplet ", np.mean(np.array(loss_sim)),
              "Current Center:", np.mean(np.array(loss_sim_ct)),
              "Current Cls:", np.mean(np.array(loss_cls)),
              "Current Acc:", np.mean(np.array(acc_cls)))

        model_mse_diff.append(np.mean(np.array(loss_mse)))
        model_sim_diff.append(np.mean(np.array(loss_sim)))
        model_cls_diff.append(np.mean(np.array(loss_sim_ct)))
        model_cls_acc.append(np.mean(np.array(acc_cls)))
        model_self_ct_diff.append(np.mean(np.array(loss_cls)))
        np.save(model_mse_diff_path, model_mse_diff)
        np.save(model_sim_diff_path, model_sim_diff)
        np.save(model_sim_cls_path, model_cls_diff)
        np.save(model_sim_cls_acc_path, model_cls_acc)
        np.save(model_self_ct_diff_path, model_self_ct_diff)

        STFTPositiveNet = STFTPositiveNet.eval()
        STFTCenterLoss = STFTCenterLoss.eval()
        STFTTripletLoss = STFTTripletLoss.eval()

        for _, stft_data, _, _, id in tqdm(test_dataloader, ncols=80, desc="Testing"):
            # ra = ra.to(device)
            sf = stft_data.to(device)
            sf = sf.reshape(-1, 1, 256, 212)
            # sf = sf.permute(0, 3, 1, 2)
            hid_codes, pred_stft, labels, reducted_code, red_cls = STFTPositiveNet(sf)
            triplet_loss = STFTTripletLoss(reducted_code, id)
            center_loss = STFTCenterLoss(reducted_code, id)
            loss_mse_pred = mse_loss(pred_stft, labels)
            loss_cls_pred = clsfi_loss(red_cls, id.to(device)-1)
            _, predicted = torch.max(red_cls, dim=1)
            correct = (predicted == id.to(device)-1).sum().item()
            accuracy = correct / id.size(0)
            loss_clst.append(loss_cls_pred.detach().cpu().numpy())
            acc_clst.append(accuracy)
            loss_mset.append(loss_mse_pred.detach().cpu().numpy())
            loss_simt.append(triplet_loss.detach().cpu().numpy())
            loss_sim_ctt.append(center_loss.detach().cpu().numpy())

        print('Current Test Loss: %f', np.mean(np.array(loss_mset)),
              "Current Test Triplet ", np.mean(np.array(loss_simt)),
              "Current Test Center:", np.mean(np.array(loss_sim_ctt)),
              "Current Test Cls:", np.mean(np.array(loss_clst)),
              "Current Test Acc:", np.mean(np.array(acc_clst)))

        model_mse_diff_test.append(np.mean(np.array(loss_mset)))
        model_sim_diff_test.append(np.mean(np.array(loss_simt)))
        model_cls_diff_test.append(np.mean(np.array(loss_sim_ctt)))
        model_cls_acc_test.append(np.mean(np.array(acc_clst)))
        model_self_ct_diff_test.append(np.mean(np.array(loss_clst)))
        np.save(model_mse_diff_path_test, model_mse_diff_test)
        np.save(model_sim_diff_path_test, model_sim_diff_test)
        np.save(model_sim_cls_path_test, model_cls_diff_test)
        np.save(model_sim_cls_acc_path_test, model_cls_acc_test)
        np.save(model_self_ct_diff_path_test, model_self_ct_diff_test)