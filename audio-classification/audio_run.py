import os
import ffmpeg
import imageio


from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.editor as mp
import pickle
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
import re
import collections

import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import soundfile as sf



def extract_feature(file_name):
    X, sample_rate = sf.read(file_name, dtype='float32')
    if X.ndim > 1:
        X = X[:,0]
    X = X.T

    # short term fourier transform
    stft = np.abs(librosa.stft(X))

    # mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    print('mfcc shape : {}'.format(mfccs.shape))
    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    print('chroma shape : {}'.format(chroma.shape))
    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    print('melspectrogram shape : {}'.format(mel.shape))
    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    print('contrast shape : {}'.format(contrast.shape))

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    print('tonnetz shape : {}'.format(tonnetz.shape))
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,193)), np.empty(0)
    files_order = []
    for label, sub_dir in enumerate(sub_dirs):
        print(sub_dir)
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            print(fn)
            files_order.append(fn)
            try:
                mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            except Exception as e:
                print("[Error] extract feature error. %s" % (e))
                continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
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


video_name = "./test_sk.mp4"
duration = 5
episode = ""

idx = 1
clip = VideoFileClip(video_name)

length = clip.duration
start = 0 # except begin_again is 10
end = start + duration

while end < length:

    file_name = "./data/output/" +  episode + "" + str(idx)+ ".mp4"
    wav_name = "./data/output/"+episode + "" + str(idx) + ".wav"
    # print(os.path.isfile(file_name))
    if os.path.isfile(file_name) == False :
        ffmpeg_extract_subclip(video_name, start, end, targetname=file_name)
        clip.subclip(start,end).audio.write_audiofile(wav_name)

    start = end
    end = start + duration
    if end >= length:
        end = length
    idx = idx + 1
    print(idx)

clip.close()

# Get features and labels
r = os.listdir("./data/")

features, _, filenames = parse_audio_files('./data', r)
filename = './svm/svm.sav'
with open(filename, 'br+') as my_dump_file:
    u = pickle._Unpickler(my_dump_file)
    u.encoding = 'latin1'
    loaded_model = u.load()
    print(loaded_model)
# filename = './svm/svm.sav'
# with open(filename, 'br+') as my_dump_file:
#     loaded_model = pickle.load(my_dump_file)
X = features

# Simple SVM
y_pred = loaded_model.predict_proba(X)
y_pred = y_pred.tolist()

rst = []
for i in range(len(y_pred)):
    if y_pred[i][0] > 0.7  :
        rst.append( 'song')
    elif y_pred[i][0] < 0.3:
        rst.append( 'speech')
    else:
        rst.append('none')

filenames = map(lambda x : x[x.find('output')+7:], filenames) # wav path -> wav file name
id_label = dict(zip(filenames,rst))

IDs = list(id_label.keys())

print(IDs)
IDs.sort(key=lambda f: int(f.split('.')[0]))
print(IDs)

labels = []
for i in range(len(IDs)):
    labels.append(id_label.get(IDs[i]))

zip(IDs, labels)

start_point = []
end_point = []

# singing duration
# y_pred continue 5 count -> singing point
count = 0
reverse_count  = 0
bb = labels
for i, predict in enumerate(bb):
    if bb[i:(i+4)] == ['song','song','song','song']:
        start_point.append(i)
    if bb[i:(i+4)] == ['speech', 'speech', 'speech', 'speech']:
        end_point.append(i)


# divide and merge mp4
clip = VideoFileClip(video_name)
ffmpeg_extract_subclip(video_name, 5.0*(np.min(start_point)+1), 5.0*np.max(end_point), targetname = 'extract_test0.mp4')
clip.close()
# dlib search code
# Berkeley code
# is park?