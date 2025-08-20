from torch.utils.data import Dataset, DataLoader
from Utilize.LoadFiles import *
import numpy as np
import os

test_data_set_rate = 0.666666666666666666666666666667

default_cls_num = 5
default_frames  = [
                    [410, 410, 410, 410, 410, 410, 410, 410], # Ji
                    [410, 410, 410, 410, 410, 410, 410, 410], # dai
                    [410, 410, 410, 410, 410, 410, 410, 410], # zhu
                    [410, 410, 410, 410, 410, 410, 410, 410], # xiang
                    [921, 917, 917, 917], # drone 1 carbon
                    [], # drone 2 nylon
                    [404, 404, 404, 361], # multi 1
                    [221, 214, 404, 314], # multi 2
                    [361, 361, 311, 311], # multi 3
                    [311, 311, 311, 311], # multi 4
                    [], # zhe 1
                    [], # zhe 2
                    [], # zhe 3
                    [], # zhe 4
                    [], # gu
                    [], # xia
                    [], # ph4
                ]

default_alter = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    1,
    1,
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [],
    [],
    [],
    [],
    1,
    1,
    1,
    1
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
def get_stft_id_exp_prob3(frames_length, data_length, target_num, cls_id):
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
                        pname = './SavedData2/' + str(ii + 1) + '/prob_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                        sname = './SavedData2/' + str(ii + 1) + '/estft_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                        rname = './SavedData2/' + str(ii + 1) + '/ra_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1)  + '.mat'
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
                        pname = './SavedData2/' + str(ii + 1) + '/prob_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                        sname = './SavedData2/' + str(ii + 1) + '/estft_' + str(jj + 1) + '_' + str(kk) + '_' + str(tt + 1) + '.mat'
                        rname = './SavedData2/' + str(ii + 1) + '/ra_' + str(jj + 1) + '_' + str(kk) + '.mat'
                        ptname = './SavedData2/' + str(ii + 1) + '/prob_' + str(jj + 1) + '_' + str(kk) + '.mat'
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

    indices = np.random.choice(len(prob_name), size=test_num, replace=False)
    if not os.path.exists('indices2.npy'):
        np.save('indices2.npy', indices)
    else:
        np.load('indices2.npy')



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
    test_id_total = np.array(id_total, dtype=object)[indices].tolist()
    id_total = np.delete(np.array(id_total, dtype=object), indices).tolist()

    return (id, prob_name, stft_name, ra_name, prob_total_name, stft_total_name, local_total_name, id_total,
            test_id, test_prob, test_stft, test_ra, test_prob_total, test_stft_total, test_local_total, test_id_total)