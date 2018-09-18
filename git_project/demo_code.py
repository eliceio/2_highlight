# coding: utf-8 & python3
import sys
sys.path.insert(0, './audio-classification/')
import os
import feat_extract
from moviepy.video.io.VideoFileClip import VideoFileClip
import pickle
import numpy as np
import re


from moviepy.tools import subprocess_call
from moviepy.config import get_setting
from det_dlib import determine_dlib

#### syncronize & remove black_frame
def ffmpeg_extract_subclip_mod(filename, t1, t2, targetname=None):
    """ makes a new video file playing video file ``filename`` between
        the times ``t1`` and ``t2``. """
    name,ext = os.path.splitext(filename)
    
    if not targetname:
        T1, T2 = [int(1000*t) for t in [t1, t2]]
        targetname = name+ "%sSUB%d_%d.%s"(name, T1, T2, ext)
    
    cmd = [get_setting("FFMPEG_BINARY"),
      "-i", filename,
      "-ss", "%0.2f"%t1,
      "-t", "%0.2f"%(t2-t1),
      "-y",targetname]
    
    subprocess_call(cmd)
        

        
video_name = "../python_DL/practice/data/20min.mp4"

## sec
duration = 5

episode = ""

idx = 1
clip = VideoFileClip(video_name)

length = clip.duration
start = 
end = start + duration

while end < length:

    wav_name = "./audio-classification/output/test/"+episode + "_" + str(idx) + ".wav"

    if os.path.isfile(wav_name) == False :
        clip.subclip(start,end).audio.write_audiofile(wav_name)
        
    start = end
    end = start + duration
    if end >= length:
        end = length
    idx = idx + 1
clip.close()

sys.path.insert(0, './audio-classification/')
r = os.listdir("./audio-classification/output")

path_feature = './audio-classification/saved_file/features.npy'
path_names = './audio-classification/saved_file/filenames.npy'

if os.path.isfile(path_feature) == False :
    features, _, filenames = feat_extract.parse_audio_files('./audio-classification/output', r)
    np.save(path_feature,features)
    np.save(path_names,filenames)
else:
    features = np.load(path_feature)
    filenames = np.load(path_names)
    
#### load svm_file
filename = './audio-classification/svm.sav'
loaded_model = pickle.load(open(filename, 'rb'))
X = features

y_pred = loaded_model.predict_proba(X)
y_pred = y_pred.tolist()

rst = []
for i in range(len(y_pred)): 
    if y_pred[i][0] > 0.7  :
        rst.append( 'song')
    elif y_pred[i][0] < 0.5:
        rst.append( 'speech')
    else:
        rst.append('none')
        
#### sorting 
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split('\.', string_)]

## will modify "('./audio-classification/data')+37" path
filename_2 = map(lambda x : x[x.find('./audio-classification/data')+37:], filenames) # wav path -> wav file name 
id_label = dict(zip(filename_2,rst))
IDs = id_label.keys()

labels = []
for i in sorted(id_label, key=natural_key):
#     print(i)
#     print(id_label[i])
    labels.append(id_label[i])

# print(list(zip(IDs, labels)))

song_point = []
speech_point = []

## singing duration
## y_pred continue 5 count -> singing point
count = 0
reverse_count  = 0
## for i in np.arange(1, len(labels), 4):
for i in range(len(labels)):
    print(labels[i] , i+1)
    if labels[i:(i+2)] == ['song','speech']:
        if  'song' not in labels[(i+2):(i+6)]:
            speech_point.append(i)

    if labels[i:(i+2)] == ['song', 'none']:
        if  'song' not in labels[(i+2):(i+6)]:
            speech_point.append(i)

    if labels[i:(i+2)] == ['speech', 'song']:
        if  'speech' not in labels[(i+2):(i+6)]:
            song_point.append(i)
            
    if labels[i:(i+3)] == ['none', 'song' , 'song']:
        if  (labels[i-1] != 'song') and (labels[(i+3):(i+6)] == ['song', 'song', 'song']):
            song_point.append(i)
            
            
i = 0
while (len(song_point) != 0) :
    
    
#### devide & save videos
    speech_point = list(filter( lambda x : x > np.min(song_point), speech_point))
#     print("start = ",np.min(song_point), " end = ",np.min(speech_point))

## using dlib, get specific person
    result = determine_dlib(video_name, np.min(song_point), np.min(speech_point), "../python_DL/practice/AI_project/SO_RAN.JPG", duration)
    if result == True:
        print("saved File")
        clip = VideoFileClip(video_name)
        i += 1
        if len(speech_point) == 0: 
#             print(i)
            ffmpeg_extract_subclip_mod(video_name, duration*np.min(song_point)  , duration*len(labels)+5, targetname = 'extract_test_{}.mp4'.format(i))
        else:
#             print(i)
            ffmpeg_extract_subclip_mod(video_name, duration*np.min(song_point)-5  , duration*np.min(speech_point)+5, targetname = 'extract_test_{}.mp4'.format(i))
        clip.close()
        
    else:
        print("False File")
    song_point = list(filter(lambda x : x > np.min(speech_point), song_point))