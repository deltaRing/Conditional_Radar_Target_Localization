import numpy as np
from scipy.io import loadmat

def load_mat(name, data_type='prob'):
    data = loadmat(name)
    if data_type == 'prob':
        data = data['prob']
    elif data_type == 'ra':
        data = data['DoAResult']
    elif data_type == 'probs':
        if data_type in data:
            data = data["probs"]
        else:
            data = data["prob"]
    else:
        data = abs(data['stft_result']) # complex
        data = 10 * np.log10(data)
    data = data.astype(np.float32)
    if data_type != 'stft_result':
        data = np.expand_dims(data, axis=0)
    return data
