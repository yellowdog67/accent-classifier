import librosa
import numpy as np

def extract_features(file_path):
    # Load audio (resample to 16kHz)
    y, sr = librosa.load(file_path, sr=16000)

    # Extract 40 MFCC coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Take mean and std across time
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Combine into single feature vector (80 features)
    features = np.concatenate((mfcc_mean, mfcc_std))

    return features
