import numpy as np
from matplotlib import pyplot as plt

# using the
def stft(signal, fs=6.41e3, window_size=240, hop_size=120, window_fn=np.hanning, fftn=512):
    """
    手写 STFT 实现
    :param signal: 输入信号（1D numpy array）
    :param fs: 采样率（Hz）
    :param window_size: 窗口长度
    :param hop_size: 每次滑动的步长
    :param window_fn: 窗函数，如 np.hanning 或 np.hamming
    :return: stft_matrix (2D array), freqs, times
    """
    signal_length = len(signal)
    window = window_fn(window_size)
    num_frames = 1 + (signal_length - window_size) // hop_size

    stft_matrix = []

    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start:start + window_size] * window
        spectrum = np.fft.fftshift(np.fft.fft(frame, fftn))
        if len(stft_matrix) == 0:
            stft_matrix = np.reshape(spectrum, [1, fftn])
        else:
            sp = np.reshape(spectrum, [1, fftn])
            stft_matrix = np.vstack([stft_matrix, sp])
    # shape: (freq_bins, time_frames)

    freqs = np.fft.fftfreq(window_size, d=1/fs)
    times = np.arange(num_frames) * hop_size / fs

    return stft_matrix, freqs, times

if __name__ == '__main__':
    t = np.linspace(0, 1e-3, 10000) # 10MHz
    s0 = np.sin(2 * np.pi * t * 1e2)
    s1 = np.sin(2 * np.pi * t * 1e3)
    s2 = np.sin(2 * np.pi * t * 2.5e3)
    s3 = np.sin(2 * np.pi * t * 5e3)
    s4 = np.sin(2 * np.pi * t * 7.5e3)
    s5 = np.sin(2 * np.pi * t * 1e4)
    s6 = np.sin(2 * np.pi * t * 2.5e4)
    s7 = np.sin(2 * np.pi * t * 3e4)
    signal = np.hstack([s0, s1, s2, s3, s4, s5, s6, s7])
    stft_feature, freq, times = stft(signal, fs=1e7)
    # f, t_stft, Zxx = stft(signal, fs=1e7, nperseg=256)
    plt.imshow(abs(stft_feature), cmap='hot', aspect='auto')
    plt.show()