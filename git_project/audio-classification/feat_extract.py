#coding= UTF-8

import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import soundfile as sf
import time
def extract_feature(file_name):
#    X, sr = sf.read(file_name, dtype='float32')
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

 #   sample_rate =  sr/2
#    X = librosa.resample(X, sr, sample_rate)

    if X.ndim > 1:
        X = X[:,0]
    X = X.T

    # short term fourier transform
    stft = np.abs(librosa.stft(X))

    # mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T,axis=0) # 13
    print('mfcc shape : {}'.format(mfccs.shape))

    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    print('chroma shape : {}'.format(chroma.shape))

    # melspectrogram
#    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
#    print('melspectrogram shape : {}'.format(mel.shape))

    # spectral contrast
#    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
#    print('contrast shape : {}'.format(contrast.shape))

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    print('tonnetz shape : {}'.format(tonnetz.shape))
    return mfccs, chroma, tonnetz
# mfcc chroma , mel ,chroma ,  tonnetz
def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,31)), np.empty(0) # 31
    files_order = []
    for label, sub_dir in enumerate(sub_dirs):
        print(sub_dir)
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            print(fn)
            files_order.append(fn)
            try:
                mfccs, chroma, tonnetz  = extract_feature(fn)
            except Exception as e:
                print("[Error] extract feature error. %s" % (e))
                continue
            ext_features = np.hstack([mfccs, chroma, tonnetz])
            features = np.vstack([features,ext_features])
            # labels = np.append(labels, fn.split('/')[1])
            labels = np.append(labels, label)
        print("extract %s features done" % (sub_dir))
    return np.array(features), np.array(labels, dtype = np.int), files_order

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

# Get features and labels
#r = os.listdir("data/")
#r.sort()
#features, labels,_ = parse_audio_files('data', r)
#np.save('feat.npy', features)
#np.save('label.npy', labels)
