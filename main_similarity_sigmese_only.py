import os

import numpy as np
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

def gather_data(label_id, data):
    label_1 = []
    label_2 = []
    labels  = []
    for ii in range(data.shape[0]):
        for jj in range(data.shape[0]):
            if jj <= ii:
                continue
            if len(label_1) == 0:
                label_1 = data[ii][:]
                label_2 = data[jj][:]
            else:
                label_1 = torch.vstack([label_1, data[ii][:]])
                label_2 = torch.vstack([label_2, data[jj][:]])
            if label_id[ii] == label_id[jj]:
                labels.append(1)
            else:
                labels.append(0)
    return label_1, label_2, labels

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

batch_num  = 32
num_epochs = 100
threads    = 8
# train_datasets   = STFTIDDataset()
train_datasets   = DatasetsName(False)
train_dataloader = DataLoader(train_datasets, batch_size=batch_num, shuffle=True, num_workers=threads)
test_datasets   = DatasetsName(True)
test_dataloader = DataLoader(test_datasets, batch_size=batch_num, shuffle=True, num_workers=threads)
model_save_path = "./SavedModel/"
model_name            = "u_net_model-lite_9.pth"
auto_coder_model_name = "auto_net_model-lite_9_only.pth"
pseudo_coder_model_name = "pseudo_net_model-lite_9_only.pth"
sigmese_coder_model_name = "sigmese_net_model-lite_9_only.pth"

STFTPositiveNet   = STFTNetworkWithNoCLS().to(device)
EvalPositiveCode  = EvalTargets().to(device)
STFTTripletLoss   = TripletLoss().to(device)

# set up the optimizer
opti = Adam([{'params': STFTPositiveNet.parameters()}], lr=1e-3)
#
model_sim_diff = []
model_mse_diff = []
model_cls_diff = []

model_sim_diff_test = []
model_mse_diff_test = []
model_cls_diff_test = []
#
model_mse_diff_path     = "mse_sim_diff_9_only.npy"
model_sim_cls_path      = "cls_sim_9_only.npy"
model_sim_diff_path     = "mse_sim_9_only.npy"

model_mse_diff_path_test     = "mse_sim_diff_9_test_only.npy"
model_sim_diff_path_test     = "mse_sim_9_test_only.npy"
model_sim_cls_path_test      = "cls_sim_9_test_only.npy"
#
if os.path.exists(model_sim_diff_path):
    model_mse_diff = np.load(model_mse_diff_path).tolist()
    model_cls_diff = np.load(model_sim_cls_path).tolist()
    model_sim_diff = np.load(model_sim_diff_path).tolist()

if __name__ == '__main__':
    if os.path.exists(model_save_path + sigmese_coder_model_name):
        checkpoint = torch.load(model_save_path + sigmese_coder_model_name)
        STFTPositiveNet.load_state_dict(checkpoint['model_stft_feature'])
        EvalPositiveCode.load_state_dict(checkpoint['model_eval_feature'])
        STFTTripletLoss.load_state_dict(checkpoint['model_triplet_feature'])

    feats = []
    ids = []

    for epoch in range(num_epochs):
        loss_mse = []
        loss_sim = []
        loss_sim_cls  = []

        loss_mset = []
        loss_simt = []
        loss_sim_clst = []

        STFTPositiveNet = STFTPositiveNet.train()
        STFTTripletLoss = STFTTripletLoss.train()
        EvalPositiveCode = EvalPositiveCode.train()

        # for sf, id in tqdm(train_dataloader, ncols=80, desc="Training"):
        for _, stft_data, _, _, id in tqdm(train_dataloader, ncols=80, desc="Training"):
            # ra = ra.to(device)
            sf = stft_data.to(device)
            sf = sf.reshape(-1, 1, 256, 212)
            # sf = sf.permute(0, 3, 1, 2)
            opti.zero_grad()

            hid_codes, pred_stft, labels, reducted_code = STFTPositiveNet(sf)
            triplet_loss = STFTTripletLoss(reducted_code, id)
            loss_mse_pred = mse_loss(pred_stft, labels)

            # if len(feats) == 0:
            #     feats = reducted_code.detach().cpu()
            #     ids   = id.detach().cpu()
            # else:
            #     feats = torch.cat([feats, reducted_code.detach().cpu()])
            #     ids   = torch.cat([ids, id.detach().cpu()])
            #
            #     if feats.shape[0] > 2048:
            #         apply_tsne(feats, ids)

            t1, t2, label = gather_data(id, reducted_code)
            label = torch.tensor(label).to(device)
            pred  = EvalPositiveCode(t1, t2)
            clsloss   = cls_loss(pred, label)

            loss_mse.append(loss_mse_pred.detach().cpu().numpy())
            loss_sim.append(triplet_loss.detach().cpu().numpy())
            loss_sim_cls.append(clsloss.detach().cpu().numpy())

            loss_position_total = 1e2 * loss_mse_pred + triplet_loss + clsloss
            loss_position_total.backward()
            opti.step()

        torch.save(
            {
                'model_stft_feature': STFTPositiveNet.state_dict(),
                'model_triplet_feature': STFTTripletLoss.state_dict(),
                'model_eval_feature': EvalPositiveCode.state_dict()
            },
            model_save_path + sigmese_coder_model_name
        )

        print('Current Loss: %f', np.mean(np.array(loss_mse)),
              "Current Triplet %f", np.mean(np.array(loss_sim)),
              "Current Cls %f", np.mean(np.array(loss_sim_cls)))

        model_mse_diff.append(np.mean(np.array(loss_mse)))
        model_sim_diff.append(np.mean(np.array(loss_sim)))
        model_cls_diff.append(np.mean(np.array(loss_sim_cls)))
        np.save(model_mse_diff_path, model_mse_diff)
        np.save(model_sim_diff_path, model_sim_diff)
        np.save(model_sim_cls_path, model_cls_diff)

        STFTPositiveNet = STFTPositiveNet.eval()
        STFTTripletLoss = STFTTripletLoss.eval()
        EvalPositiveCode = EvalPositiveCode.eval()

        for _, stft_data, _, _, id in tqdm(test_dataloader, ncols=80, desc="Testing"):
            # ra = ra.to(device)
            sf = stft_data.to(device)
            sf = sf.reshape(-1, 1, 256, 212)
            # sf = sf.permute(0, 3, 1, 2)
            hid_codes, pred_stft, labels, reducted_code = STFTPositiveNet(sf)
            triplet_loss = STFTTripletLoss(reducted_code, id)
            loss_mse_pred = mse_loss(pred_stft, labels)

            t1, t2, label = gather_data(id, reducted_code)
            label = torch.tensor(label).to(device)
            pred = EvalPositiveCode(t1, t2)
            cls = cls_loss(pred, label)

            loss_mset.append(loss_mse_pred.detach().cpu().numpy())
            loss_simt.append(triplet_loss.detach().cpu().numpy())
            loss_sim_clst.append(cls.detach().cpu().numpy())


        print('Current Test Loss: %f', np.mean(np.array(loss_mset)),
              "Current Test Triplet ", np.mean(np.array(loss_simt)),
              "Current Cls %f", np.mean(np.array(loss_sim_clst)))

        model_mse_diff_test.append(np.mean(np.array(loss_mset)))
        model_sim_diff_test.append(np.mean(np.array(loss_simt)))
        model_cls_diff_test.append(np.mean(np.array(loss_sim_clst)))
        np.save(model_mse_diff_path_test, model_mse_diff_test)
        np.save(model_sim_diff_path_test, model_sim_diff_test)
        np.save(model_sim_cls_path_test, model_cls_diff_test)