from torch.utils.data import Dataset, DataLoader
from Utilize.LoadFiles import *
import numpy as np
import os

test_data_set_rate = 0.666666666666666666666666666667

default_cls_num = 5
default_frames  = [
                   [917, 917, 917, 917, 917, 917], # drone 1 nylon
                   [917, 917, 917, 917, 917, 917], # drone 2 carbon
                   [917, 917, 917, 917, 917, 917, 917, 917], # should be zhai
                   [917, 917, 917, 917, 917, 917, 917, 917], # should be wo
                   [917, 917, 917, 917, 917, 917, 917, 917], # should be li
                   [917], # 1 zhai 2 li
                   [917], # 1 wo 2 li
                   [917], # 1 wo 2 li
                   [917]  # 1 li 2 zhai
                ]
default_tar     = [1, 1, 1, 1, 1, 2, 2, 2, 2]
default_datalength = [6, 6, 8, 8, 8, 1, 1, 1, 1]

cls = [
    [1,1,1,1,1,1],
    [2,2,2,2,2,2,2,2,2,2],
    [3,3,3,3,3,3,3,3,3,3],
    [4,4,4,4,4,4,4,4],
    [5,5,5,5,5,5,5,5],
    [[3,5]],
    [[4,5]],
    [[4,5]],
    [[5,3]]
]

# 
def get_stft_exp_prob2(frames_length, data_length, target_num, cls_id):
    id = []
    prob_name = [] # target
    stft_name = [] # for different target
    ra_name   = [] # target
    prob_total_name = [] # total

    test_id = []
    test_prob = []
    test_stft = []
    test_ra   = []
    test_prob_total = []
    test_len = 0

    single = []

    # 先填充
    iii = 0
    for ii in range(len(frames_length)):  # 8
        for jj in range(data_length[ii]):  # [8, 10, 10, 8, 8, 1, 1, 1, 1]
            test_len += int(frames_length[ii][jj] * test_data_set_rate)
            for kk in range(frames_length[ii][jj]): # default_frames
                for tt in range(target_num[ii]):
                    if target_num[ii] == 1:
                        # 填充目标数
                        pname = './SaveData/' + str(ii + 1) + '/prob_' + str(jj + 1) + '_' + str(kk) + '.mat'
                        sname = './SaveData/' + str(ii + 1) + '/estft_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                        rname = './SaveData/' + str(ii + 1) + '/ra_' + str(jj + 1) + '_' + str(kk) + '.mat'
                        id.append(cls_id[ii][jj])
                        prob_name.append(pname)
                        stft_name.append(sname)
                        ra_name.append(rname)
                        prob_total_name.append(pname)
                        single.append(True)
                        iii+=1
                    else:
                        pname = './SaveData/' + str(ii + 1) + '/prob_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                        sname = './SaveData/' + str(ii + 1) + '/estft_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                        rname = './SaveData/' + str(ii + 1) + '/ra_' + str(jj + 1) + '_' + str(kk) + '.mat'
                        ptname = './SaveData/' + str(ii + 1) + '/prob_' + str(jj + 1) + '_' + str(kk) + '.mat'
                        id.append(cls_id[ii][jj][tt])
                        prob_name.append(pname)
                        stft_name.append(sname)
                        # if target_num[ii] == tt + 1:
                        ra_name.append(rname)
                        prob_total_name.append(ptname)
                        single.append(False)

    indices = np.random.choice(len(prob_name), size=test_len, replace=False) - 1
    if not os.path.exists('indices2.npy'):
        np.save('indices2.npy', indices)
    else:
        np.load('indices2.npy')
    test_prob = np.array(prob_name)[indices].tolist()
    prob_name = np.delete(prob_name, indices).tolist()
    test_stft = np.array(stft_name)[indices].tolist()
    stft_name = np.delete(stft_name, indices).tolist()
    test_id   = np.array(id)[indices].tolist()
    id        = np.delete(id, indices).tolist()
    test_ra   = np.array(ra_name)[indices].tolist()
    ra_name = np.delete(ra_name, indices).tolist()
    test_prob_total = np.array(prob_total_name)[indices].tolist()
    prob_total_name = np.delete(prob_total_name, indices).tolist()
    test_single  = np.array(single)[indices].tolist()
    single = np.delete(single, indices).tolist()

    return id, prob_name, stft_name, ra_name, prob_total_name, test_id, test_prob, test_stft, test_ra, test_prob_total, single, test_single

def get_stft_id_exp_prob(frames_length, data_length, target_num, cls_id):
    id        = [] # for different target
    prob_name = [] # target
    stft_name = [] # for different target
    ra_name   = [] # target
    prob_total_name = [] # total

    test_id = []
    test_prob = []
    test_stft = []
    test_ra   = []
    test_prob_total = []

    # 先填充
    iii = 0
    for ii in range(len(frames_length)): # 8
        for jj in range(data_length[ii]): # [8, 10, 10, 8, 8, 1, 1, 1, 1]
            test_len = int(frames_length[ii][jj] * test_data_set_rate)
            train_len = frames_length[ii][jj] - test_len
            for kk in range(frames_length[ii][jj]): # default_frames
                if kk < train_len:
                    for tt in range(target_num[ii]):
                        if target_num[ii] == 1:
                            # 填充目标数
                            pname = './SavedData/' + str(ii + 1) + '/prob_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                            sname = './SavedData/' + str(ii + 1) + '/estft_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                            rname = './SavedData/' + str(ii + 1) + '/ra_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                            id.append(cls_id[ii][jj])
                            prob_name.append(pname)
                            stft_name.append(sname)
                            ra_name.append(rname)
                            prob_total_name.append(pname)
                            iii+=1
                        else:
                            pname = './SavedData/' + str(ii + 1) + '/prob_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                            sname = './SavedData/' + str(ii + 1) + '/estft_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                            rname = './SavedData/' + str(ii + 1) + '/ra_' + str(jj + 1) + '_' + str(kk) + '.mat'
                            ptname = './SavedData/' + str(ii + 1) + '/prob_total_' + str(jj + 1) + '_' + str(kk) + '.mat'
                            id.append(cls_id[ii][jj][tt])
                            prob_name.append(pname)
                            stft_name.append(sname)
                            # if target_num[ii] == tt + 1:
                            ra_name.append(rname)
                            prob_total_name.append(ptname)
                else:
                    for tt in range(target_num[ii]):
                        if target_num[ii] == 1:
                            # 填充目标数
                            pname = './SavedData/' + str(ii + 1) + '/prob_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                            sname = './SavedData/' + str(ii + 1) + '/estft_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                            rname = './SavedData/' + str(ii + 1) + '/ra_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                            test_id.append(cls_id[ii][jj])
                            test_prob.append(pname)
                            test_stft.append(sname)
                            test_ra.append(rname)
                            test_prob_total.append(pname)
                            iii+=1
                        else:
                            pname = './SavedData/' + str(ii + 1) + '/prob_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                            sname = './SavedData/' + str(ii + 1) + '/estft_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                            rname = './SavedData/' + str(ii + 1) + '/ra_' + str(jj + 1) + '_' + str(kk) + '.mat'
                            ptname = './SavedData/' + str(ii + 1) + '/prob_total_' + str(jj + 1) + '_' + str(kk) + '.mat'
                            test_id.append(cls_id[ii][jj][tt])
                            test_prob.append(pname)
                            test_stft.append(sname)
                            # if target_num[ii] == tt + 1:
                            test_ra.append(rname)
                            test_prob_total.append(ptname)

    return id, prob_name, stft_name, ra_name, prob_total_name, test_id, test_prob, test_stft, test_ra, test_prob_total

class DatasetsName(Dataset):
    def __init__(self, test_enable):
        (id, prob_name, stft_name, ra_name, prob_total_name, test_id,
         test_prob, test_stft, test_ra, test_prob_total, issingle, issingle_test) =\
            get_stft_exp_prob2(default_frames, default_datalength, default_tar, cls)
        self.id = id
        self.prob_name = prob_name
        self.stft_name = stft_name
        self.prob_total_name = prob_total_name
        self.ra_name = ra_name
        self.test_id = test_id
        self.test_prob = test_prob
        self.test_stft = test_stft
        self.test_ra = test_ra
        self.test_prob_total = test_prob_total
        self.test_enable = test_enable
        self.issingle = issingle
        self.issingle_test = issingle_test

    def __len__(self):
        if self.test_enable:
            return len(self.test_prob)
        else:
            return len(self.prob_name)

    def __getitem__(self, idx):

        if self.test_enable:
            stft_data = load_mat(self.test_stft[idx], 'stft_result')
            if self.issingle_test[idx]:
                prob_data = load_mat(self.test_prob[idx], 'probs')
            else:
                prob_data = load_mat(self.test_prob[idx], 'prob')
            prob_total = load_mat(self.test_prob_total[idx], 'probs')
            ra_data = load_mat(self.test_ra[idx], 'ra')
            id = self.test_id[idx]
        else:
            stft_data  = load_mat(self.stft_name[idx], 'stft_result')
            if self.issingle[idx]:
                prob_data = load_mat(self.prob_name[idx], 'probs')
            else:
                prob_data = load_mat(self.prob_name[idx], 'prob')
            prob_total = load_mat(self.prob_total_name[idx], 'probs')
            ra_data = load_mat(self.ra_name[idx], 'ra')
            id         = self.id[idx]
        return prob_data, stft_data, prob_total, ra_data, id

