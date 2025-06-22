# src/feature_extraction.py

import numpy as np
import pandas as pd
import pywt
from scipy.fft import fft
from scipy.signal import hilbert
from antropy import sample_entropy, app_entropy
from scipy.stats import kurtosis, skew

def process_sequence(seq_str):
    """把 comma-separated string 转成 numpy array"""
    return np.array([float(x) for x in seq_str.split(',') if x.strip()])

def extract_stat_features(x):
    """基本时域统计特征"""
    return {
        'mean': x.mean(),
        'std': x.std(),
        'max': x.max(),
        'kurtosis': kurtosis(x),
        'skew': skew(x),
        'zcr': np.mean((x[:-1] * x[1:] < 0).astype(int))
    }

def extract_fft_features(x, fs=1000):
    """FFT 主频 & 频谱能量"""
    freqs = np.fft.fftfreq(len(x), 1/fs)
    vals  = np.abs(fft(x))
    pos   = freqs >= 0
    freqs = freqs[pos]; vals = vals[pos]
    return {
        'fft_peakfreq': freqs[np.argmax(vals)],
        'fft_energy': np.sum(vals**2)
    }

def extract_wavelet_features(x, wavelet='db4', level=3):
    """小波能量与包熵"""
    coeffs = pywt.wavedec(x, wavelet, level=level)
    energies = [np.sum(c**2) for c in coeffs]
    p = np.array(energies)/sum(energies)
    packet_ent = -np.sum(p * np.log2(p + 1e-12))
    feats = {f'wavelet_energy_L{l}': en for l, en in enumerate(energies)}
    feats['wavelet_packet_ent'] = packet_ent
    return feats

def extract_nonlinear_features(x):
    """Hjorth + Entropy + Envelope"""
    # Hjorth
    d1 = np.diff(x); d2 = np.diff(d1)
    var0, var1, var2 = x.var(), d1.var(), d2.var()
    hj_activity = var0
    hj_mobility = np.sqrt(var1/var0) if var0>0 else 0
    hj_complexity = np.sqrt((var2/var1)/(var1/var0)) if var1>0 and var0>0 else 0

    # Entropy
    samp_en = sample_entropy(x)
    app_en  = app_entropy(x)

    # Envelope
    env = np.abs(hilbert(x))
    env_feats = {'env_mean': env.mean(), 'env_std': env.std(), 'env_max': env.max()}

    return {
        'hj_activity': hj_activity,
        'hj_mobility': hj_mobility,
        'hj_complexity': hj_complexity,
        'sample_entropy': samp_en,
        'approx_entropy': app_en,
        **env_feats
    }

def featurize_row(row):
    """对一行 kinematics DataFrame 输出 dict of features"""
    feats = {}
    for col in [
        'linear_acc_x', 'linear_acc_y', 'linear_acc_z',
        'angular_vel_x','angular_vel_y','angular_vel_z',
        'angular_acc_x','angular_acc_y','angular_acc_z'
    ]:
        seq = process_sequence(row[col])
        prefix = col
        feats.update({f'{prefix}_{k}': v for k,v in extract_stat_features(seq).items()})
        feats.update({f'{prefix}_{k}': v for k,v in extract_fft_features(seq).items()})
        feats.update({f'{prefix}_{k}': v for k,v in extract_wavelet_features(seq).items()})
        feats.update({f'{prefix}_{k}': v for k,v in extract_nonlinear_features(seq).items()})
    return feats

def build_feature_matrix(kinematics_df):
    """将整个 DataFrame 转换为特征矩阵"""
    feature_dicts = kinematics_df.apply(featurize_row, axis=1).tolist()
    return pd.DataFrame(feature_dicts)

