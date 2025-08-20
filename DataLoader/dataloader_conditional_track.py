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
def get_stft_id_exp_prob2(frames_length, data_length, target_num, cls_id):
    id        = [] # for different target
    prob_name = [] # target
    stft_name = [] # for different target
    ra_name   = [] # target
    prob_total_name = [] # total prob
    stft_total_name = [] # total stft
    local_total_name = []
    id_total         = []

    test_id = []
    test_prob = []
    test_stft = []
    test_ra   = []
    test_prob_total = []
    test_stft_total = []
    test_local_total = []
    test_id_total    = []
    test_num         = 0
    single = []
    record_single = []

    # 先填充
    iii = 0
    for ii in range(len(frames_length)): # 8
        for jj in range(data_length[ii]): # [8, 10, 10, 8, 8, 1, 1, 1, 1]
            test_len = int(frames_length[ii][jj] * test_data_set_rate)
            test_num += test_len
            for kk in range(frames_length[ii][jj]): # default_frames
                record_stft = []
                record_prob = []
                record_id   = []

                for tt in range(target_num[ii]):
                    if target_num[ii] == 1:
                        # 填充目标数
                        pname = './SaveData/' + str(ii + 1) + '/prob_' + str(jj + 1) + '_' + str(kk) + '.mat'
                        sname = './SaveData/' + str(ii + 1) + '/estft_' + str(jj + 1) + '_' + str(kk) + '_' + str(
                            tt + 1) + '.mat'
                        rname = './SaveData/' + str(ii + 1) + '/ra_' + str(jj + 1) + '_' + str(kk) + '.mat'
                        id.append(cls_id[ii][jj])
                        prob_name.append(pname)
                        stft_name.append(sname)
                        ra_name.append(rname)
                        prob_total_name.append(pname)
                        record_stft.append(sname)
                        record_prob.append(pname)
                        record_id.append(cls_id[ii][jj])
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
                        record_stft.append(sname)
                        record_prob.append(pname)
                        record_id.append(cls_id[ii][jj][tt])
                        single.append(False)
                for tt in range(target_num[ii]):
                    stft_total_name.append(record_stft)
                    local_total_name.append(record_prob)
                    id_total.append(record_id)
                    if target_num[ii] == 1:
                        record_single.append(True)
                    else:
                        record_single.append(False)

    indices = np.random.choice(len(prob_name), size=test_num, replace=False)
    if not os.path.exists('indices.npy'):
        np.save('indices.npy', indices)
    else:
        np.load('indices.npy')


    test_prob = np.array(prob_name)[indices].tolist()
    prob_name = np.delete(prob_name, indices).tolist()
    test_stft = np.array(stft_name)[indices].tolist()
    stft_name = np.delete(stft_name, indices).tolist()
    test_id = np.array(id)[indices].tolist()
    id = np.delete(id, indices).tolist()
    test_ra = np.array(ra_name)[indices].tolist()
    ra_name = np.delete(ra_name, indices).tolist()
    test_prob_total = np.array(prob_total_name)[indices].tolist()
    prob_total_name = np.delete(prob_total_name, indices).tolist()
    test_stft_total = np.array(stft_total_name, dtype=object)[indices].tolist()
    stft_total_name = np.delete(np.array(stft_total_name, dtype=object), indices).tolist()
    test_local_total = np.array(local_total_name, dtype=object)[indices].tolist()
    local_total_name = np.delete(np.array(local_total_name, dtype=object), indices).tolist()
    test_record_single =  np.array(record_single, dtype=object)[indices].tolist()
    record_single = np.delete(np.array(record_single, dtype=object), indices).tolist()
    test_id_total = np.array(id_total, dtype=object)[indices].tolist()
    id_total = np.delete(np.array(id_total, dtype=object), indices).tolist()
    test_single = np.array(single)[indices].tolist()
    single = np.delete(single, indices).tolist()

    return (id, prob_name, stft_name, ra_name, prob_total_name, stft_total_name, local_total_name, id_total,
            test_id, test_prob, test_stft, test_ra, test_prob_total, test_stft_total, test_local_total, test_id_total, single, test_single, record_single, test_record_single)


def get_stft_id_exp_prob(frames_length, data_length, target_num, cls_id):
    id        = [] # for different target
    prob_name = [] # target
    stft_name = [] # for different target
    ra_name   = [] # target
    prob_total_name = [] # total prob
    stft_total_name = [] # total stft
    local_total_name = []
    id_total         = []

    test_id = []
    test_prob = []
    test_stft = []
    test_ra   = []
    test_prob_total = []
    test_stft_total = []
    test_local_total = []
    test_id_total    = []
    single = []

    # 先填充
    iii = 0
    for ii in range(len(frames_length)): # 8
        for jj in range(data_length[ii]): # [8, 10, 10, 8, 8, 1, 1, 1, 1]
            test_len = int(frames_length[ii][jj] * test_data_set_rate)
            train_len = frames_length[ii][jj] - test_len
            for kk in range(frames_length[ii][jj]): # default_frames
                record_stft = []
                record_prob = []
                record_id   = []
                if kk < train_len:
                    for tt in range(target_num[ii]):
                        if target_num[ii] == 1:
                            # 填充目标数
                            pname = './SaveData/' + str(ii + 1) + '/prob_' + str(jj + 1) + '_' + str(kk) + '.mat'
                            sname = './SaveData/' + str(ii + 1) + '/estft_' + str(jj + 1) + '_' + str(kk) + '_' + str(
                                tt + 1) + '.mat'
                            rname = './SaveData/' + str(ii + 1) + '/ra_' + str(jj + 1) + '_' + str(kk) + '.mat'
                            id.append(cls_id[ii][jj])
                            prob_name.append(pname)
                            stft_name.append(sname)
                            ra_name.append(rname)
                            prob_total_name.append(pname)
                            record_stft.append(sname)
                            record_prob.append(pname)
                            record_id.append(cls_id[ii][jj])
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
                            record_stft.append(sname)
                            record_prob.append(pname)
                            record_id.append(cls_id[ii][jj][tt])
                    for tt in range(target_num[ii]):
                        stft_total_name.append(record_stft)
                        local_total_name.append(record_prob)
                        id_total.append(record_id)
                else:
                    for tt in range(target_num[ii]):
                        if target_num[ii] == 1:
                            # 填充目标数
                            pname = './SaveData/' + str(ii + 1) + '/prob_' + str(jj + 1) + '_' + str(kk) + '.mat'
                            sname = './SaveData/' + str(ii + 1) + '/estft_' + str(jj + 1) + '_' + str(kk) + '_' + str(
                                tt + 1) + '.mat'
                            rname = './SaveData/' + str(ii + 1) + '/ra_' + str(jj + 1) + '_' + str(kk) + '.mat'
                            test_id.append(cls_id[ii][jj])
                            test_prob.append(pname)
                            test_stft.append(sname)
                            test_ra.append(rname)
                            test_prob_total.append(pname)
                            record_stft.append(sname)
                            record_prob.append(pname)
                            record_id.append(cls_id[ii][jj])
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
                            record_stft.append(sname)
                            record_prob.append(pname)
                            record_id.append(cls_id[ii][jj][tt])
                    for tt in range(target_num[ii]):
                        test_stft_total.append(record_stft)
                        test_local_total.append(record_prob)
                        test_id_total.append(record_id)

    return (id, prob_name, stft_name, ra_name, prob_total_name, stft_total_name, local_total_name, id_total,
            test_id, test_prob, test_stft, test_ra, test_prob_total, test_stft_total, test_local_total, test_id_total)

class ConditionalDataSets(Dataset):
    def __init__(self, test_enable):
        (id, prob_name, stft_name, ra_name, prob_total_name, stft_total_name, local_total_name, id_total,
            test_id, test_prob, test_stft, test_ra, test_prob_total, test_stft_total, test_local_total, test_id_total, single, test_single, record_single, test_record_single) =\
            get_stft_id_exp_prob2(default_frames, default_datalength, default_tar, cls)
        self.id = id
        self.prob_name = prob_name
        self.stft_name = stft_name
        self.ra_name = ra_name
        self.prob_total_name = prob_total_name
        self.stft_total_name = stft_total_name
        self.local_total_name = local_total_name
        self.id_total = id_total

        self.test_enable = test_enable
        self.test_id = test_id
        self.test_prob = test_prob
        self.test_stft = test_stft
        self.test_ra = test_ra
        self.test_prob_total = test_prob_total
        self.test_stft_total = test_stft_total
        self.test_local_total = test_local_total
        self.test_id_total = test_id_total
        self.issingle = single
        self.test_issingle = test_single
        self.record_single = record_single
        self.test_record_single = test_record_single

        if not self.test_enable:
            self.data_length = len(self.prob_name)
        else:
            self.data_length = len(self.test_prob)
        self.tar_num     = max(default_tar)
        self.amp         = 1e3
        self.prob_same   = 0.25

    def get_random_id(self, c_id):
        random_id  = np.random.choice(self.data_length)
        if not self.test_enable:
            r_id = self.id[random_id]
        else:
            r_id = self.test_id[random_id]
        label_same = r_id == c_id
        return r_id, random_id, label_same

    def __len__(self):
        if self.test_enable:
            return len(self.test_prob)
        else:
            return len(self.prob_name)

    def __getitem__(self, idx):

        if not self.test_enable:
            if self.issingle[idx]:
                current_prob = load_mat(self.prob_name[idx], 'probs')
            else:
                current_prob = load_mat(self.prob_name[idx], 'prob')
            prob_total = load_mat(self.prob_total_name[idx], 'probs')
            ra_data = load_mat(self.ra_name[idx], 'ra')
            id = self.id[idx]
        else:
            if self.test_issingle[idx]:
                current_prob = load_mat(self.test_prob[idx], 'probs')
            else:
                current_prob = load_mat(self.test_prob[idx], 'prob')
            prob_total = load_mat(self.test_prob_total[idx], 'probs')
            ra_data = load_mat(self.test_ra[idx], 'ra')
            id = self.test_id[idx]

        get_same = np.random.random() < self.prob_same
        while True:
            past_id, random_id, is_same = self.get_random_id(id)
            if get_same == is_same:
                break
        if not self.test_enable:
            past_stft = load_mat(self.stft_name[random_id], 'stft_result')
        else:
            past_stft = load_mat(self.test_stft[random_id], 'stft_result')

        if is_same:
            label_prob = current_prob
            label_prob_amp = current_prob * self.amp
        else:
            label_prob = prob_total * 1e-3
            label_prob_amp = current_prob * 1e-3
        prob_total = prob_total
        prob_total_amp = prob_total * self.amp
        past_stft = past_stft.reshape([1, 256, 212])

        detect_prob = []
        detect_stft = []
        detect_id   = []
        if not self.test_enable:
            result_len  = len(self.stft_total_name[idx])
        else:
            result_len = len(self.test_stft_total[idx])

        if not self.test_enable:
            for ii in range(self.tar_num):
                if len(self.stft_total_name[idx]) == 1:

                    if self.record_single[idx]:
                        dp = load_mat(self.local_total_name[idx][0], 'probs')
                    else:
                        dp = load_mat(self.local_total_name[idx][0], 'prob')

                    # dp = load_mat(self.local_total_name[idx][0], 'prob')
                    ds = load_mat(self.stft_total_name[idx][0], 'stft_result')
                    detect_id.append(self.id_total[idx][0])
                    if len(detect_prob) == 0:
                        detect_prob = dp
                        ds = ds.reshape([1, 256, 212])
                        detect_stft = ds
                    else:
                        detect_prob = np.vstack([detect_prob, dp])
                        ds = ds.reshape([1, 256, 212])
                        detect_stft = np.vstack([detect_stft, ds])
                elif len(self.stft_total_name[idx]) == 2:

                    if self.record_single[idx]:
                        dp = load_mat(self.local_total_name[idx][ii], 'probs')
                    else:
                        dp = load_mat(self.local_total_name[idx][ii], 'prob')

                    ds = load_mat(self.stft_total_name[idx][ii], 'stft_result')
                    detect_id.append(self.id_total[idx][ii])
                    if len(detect_prob) == 0:
                        detect_prob = dp
                        ds = ds.reshape([1, 256, 212])
                        detect_stft = ds
                    else:
                        detect_prob = np.vstack([detect_prob, dp])
                        ds = ds.reshape([1, 256, 212])
                        detect_stft = np.vstack([detect_stft, ds])
        else:
            for ii in range(self.tar_num):
                if len(self.test_stft_total[idx]) == 1:

                    if self.test_record_single[idx]:
                        dp = load_mat(self.test_local_total[idx][0], 'probs')
                    else:
                        dp = load_mat(self.test_local_total[idx][0], 'prob')

                    # dp = load_mat(self.test_local_total[idx][0], 'prob')
                    ds = load_mat(self.test_stft_total[idx][0], 'stft_result')
                    detect_id.append(self.test_id_total[idx][0])
                    if len(detect_prob) == 0:
                        detect_prob = dp
                        ds = ds.reshape([1, 256, 212])
                        detect_stft = ds
                    else:
                        detect_prob = np.vstack([detect_prob, dp])
                        ds = ds.reshape([1, 256, 212])
                        detect_stft = np.vstack([detect_stft, ds])
                elif len(self.test_stft_total[idx]) == 2:
                    # dp = load_mat(self.test_local_total[idx][ii], 'prob')
                    if self.test_record_single[idx]:
                        dp = load_mat(self.test_local_total[idx][ii], 'probs')
                    else:
                        dp = load_mat(self.test_local_total[idx][ii], 'prob')

                    ds = load_mat(self.test_stft_total[idx][ii], 'stft_result')
                    detect_id.append(self.test_id_total[idx][ii])
                    if len(detect_prob) == 0:
                        detect_prob = dp
                        ds = ds.reshape([1, 256, 212])
                        detect_stft = ds
                    else:
                        detect_prob = np.vstack([detect_prob, dp])
                        ds = ds.reshape([1, 256, 212])
                        detect_stft = np.vstack([detect_stft, ds])

        return (prob_total, prob_total_amp, ra_data, past_id, past_stft,
                label_prob, label_prob_amp, detect_prob, detect_stft, detect_id, result_len)