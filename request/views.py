from django.shortcuts import render
import cv2
from .models import *
from .r_functions import *
import face_recognition
import os
from  highlight.settings import *
from io import BytesIO
import time
import zipfile
from django.http import HttpResponse
import sys
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
import subprocess as sp

from moviepy.tools import subprocess_call
from moviepy.config import get_setting

PIPE = -1
STDOUT = -2
DEVNULL = -3

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


def sys_write_flush22(s):
    """ Writes and flushes without delay a text in the console """
    # Reason for not using `print` is that in some consoles "print"
    # commands get delayed, while stdout.flush are instantaneous,
    # so this method is better at providing feedback.
    # See https://github.com/Zulko/moviepy/pull/485
    sys.stdout.write(s)
    sys.stdout.flush()

def verbose_print(verbose, s):
    """ Only prints s (with sys_write_flush) if verbose is True."""
    if verbose:
        sys_write_flush22(s)

def subprocess_call22(cmd, verbose=True, errorprint=True):
    """ Executes the given subprocess command."""

    verbose_print(verbose, "\n[MoviePy] Running:\n>>> "+ " ".join(cmd))

    popen_params = {"stdout": DEVNULL,
                    "stderr": sp.PIPE,
                    "stdin": DEVNULL}

    if os.name == "nt":
        popen_params["creationflags"] = 0x08000000

    proc = sp.Popen(cmd, **popen_params)

    out, err = proc.communicate() # proc.wait()
    proc.stderr.close()

    if proc.returncode:
        verbose_print(errorprint, "\n[MoviePy] This command returned an error !")
        # raise IOError(err.decode('utf8'))
        raise IOError(err.decode('utf8'))
    else:
        verbose_print(verbose, "\n... command successful.\n")

    del proc

def ffmpeg_extract_subclip22(filename, t1, t2, targetname=None):
    """ Makes a new video file playing video file ``filename`` between
        the times ``t1`` and ``t2``. """
    name, ext = os.path.splitext(filename)
    print("test40 = ")
    if not targetname:
        T1, T2 = [int(1000 * t) for t in [t1, t2]]
        print("test41 = ")
        targetname = "%sSUB%d_%d.%s" % (name, T1, T2, ext)
        print("test42 = ")
    cmd = [get_setting("FFMPEG_BINARY"), "-y",
           "-i", filename,
           "-ss", "%0.2f" % t1,
           "-t", "%0.2f" % (t2 - t1),
           "-vcodec", "copy", "-acodec", "copy", targetname]
    print("test43 = ")
    subprocess_call22(cmd)
    print("test44 = ")

def sound_ex(high):
    video_name = BASE_DIR + high.file_in.url
    print("bd = ",BASE_DIR)
    duration = 5
    episode = ""

    idx = 1
    clip = VideoFileClip(video_name)
    print("clip = ",clip)
    length = clip.duration
    start = 0  # except begin_again is 10
    end = start + duration

    while end < length:

        file_name = BASE_DIR + "/request/data/output/" + episode + "" + str(idx) + ".mp4"
        wav_name = BASE_DIR + "/request/data/output/" + episode + "" + str(idx) + ".wav"
        # print(os.path.isfile(file_name))
        print("test1 = ",file_name)
        print("test2 = ", os.path.isfile(file_name))
        if os.path.isfile(file_name) == False:
            print("test3 = ",video_name)
            ffmpeg_extract_subclip22(video_name, start, end, targetname=file_name)
            print("test4 = ")
            clip.subclip(start, end).audio.write_audiofile(wav_name)
            print("test5 = ")

        start = end
        end = start + duration
        if end >= length:
            end = length
        idx = idx + 1
        print(idx)

    clip.close()

    # Get features and labels
    r = os.listdir(BASE_DIR + "/request/data/")

    features, _, filenames = parse_audio_files(BASE_DIR + '/request/data', r)
    filename = BASE_DIR +'/request/svm/svm.sav'
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
        if y_pred[i][0] > 0.7:
            rst.append('song')
        elif y_pred[i][0] < 0.3:
            rst.append('speech')
        else:
            rst.append('none')

    filenames = map(lambda x: x[x.find('output') + 7:], filenames)  # wav path -> wav file name
    id_label = dict(zip(filenames, rst))

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
    reverse_count = 0
    bb = labels
    for i, predict in enumerate(bb):
        if bb[i:(i + 4)] == ['song', 'song', 'song', 'song']:
            start_point.append(i)
        if bb[i:(i + 4)] == ['speech', 'speech', 'speech', 'speech']:
            end_point.append(i)

    # divide and merge mp4
    clip = VideoFileClip(video_name)
    t_name = BASE_DIR + high.file_out.url

    ffmpeg_extract_subclip(video_name, 5.0 * (np.min(start_point) + 1), 5.0 * np.max(end_point),
                           targetname=t_name)
    clip.close()
    high.status = 1
    high.save()
    # dlib search code
    # Berkeley code
    # is park?


# Create your views here.
M_path = os.path.join(BASE_DIR, 'media')
file_path_base = M_path + str(os.sep)

def index(request):
    if request.method == "POST":
        if ('file' in request.POST):
            high =None
            for f in request.FILES.getlist('files_video'):
                print(f.name)
                high = Highlight(file_in = f, file_out = f, fname = f.name)
                high.save()
                for i in request.FILES.getlist('files_face'):
                    f_img = Face_img(img = i, fname = i.name)
                    f_img.save()
                    high.f_img.add(f_img)
                    high.save()
                # Open the input movie file
                print("file url",high.file_in.url)
                print("BASE_DIR url", BASE_DIR)
                """
                input_movie = cv2.VideoCapture(BASE_DIR + high.file_in.url)
                length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
                w = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(w, h)
                # Create an output movie file (make sure resolution/frame rate matches input video!)
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')

                output_movie = cv2.VideoWriter(BASE_DIR + high.file_out.url, fourcc, 29.97, (w, h))
                f_img_list = high.f_img.all().order_by('pk')
                known_faces = []
                known_faces_name = []
                for face_img in f_img_list:
                    image = face_recognition.load_image_file(BASE_DIR + face_img.img.url)
                    if (len(face_recognition.face_encodings(image))>0):
                        face_encoding = face_recognition.face_encodings(image)[0]
                    else:
                        return render(request, 'request/message.html', {'msg': "error!"})
                    known_faces.append(face_encoding)
                    known_faces_name.append(face_img.fname)

                print(len(f_img_list))
                # Initialize some variables
                face_locations = []
                face_encodings = []
                face_names = []
                frame_number = 0

                while True:
                    # Grab a single frame of video
                    ret, frame = input_movie.read()
                    frame_number += 1

                    # Quit when the input video file ends
                    if not ret:
                        break

                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    rgb_frame = frame[:, :, ::-1]

                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                    face_names = []
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

                        # If you had more than 2 faces, you could make this logic a lot prettier
                        # but I kept it simple for the demo
                        name = None
                        i=0
                        print("len match = ", len(match))
                        print("len match = ", match)
                        if len(match)>0:
                            for m in match:
                               print("i = ",i)
                               print("m = ",m)
                               if m:
                                   name = known_faces_name[i]
                               i = i+1
                            face_names.append(name)
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        if not name:
                            continue

                        # Draw a box around the face
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                        # Draw a label with a name below the face
                        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

                    # Write the resulting image to the output video file
                    print("Writing frame {} / {}".format(frame_number, length))
                    output_movie.write(frame)
                high.status = 1
                high.save()
            return render(request, 'request/layout_request.html',{'msg': "complete"} )
            """
            sound_ex(high)
            return render(request, 'request/layout_request.html', {'msg': "complete"})
        elif ('download' in request.POST):
            print("download")
            high_list = Highlight.objects.filter(status=1)
            print("len = ", len(high_list))
            in_memory = BytesIO()
            now = time.localtime()
            fname = "%04d%02d%02d-%02d%02d%02d.zip" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
            zfile = zipfile.ZipFile(in_memory, 'w', zipfile.ZIP_DEFLATED)
            for high in high_list:
                print("high.file_out.name = ", high.file_out.name)
                print("high.fname = ", high.fname)
                zfile.write(file_path_base + high.file_out.name, str(high.pk)+"_out_"+high.fname )
                high.delete()
            zfile.close()
            response = HttpResponse()
            response['content_type'] = 'application/zip'
            response['Content-Disposition'] = 'attachment;filename=' + fname

            in_memory.seek(0)
            response.write(in_memory.read())
            return response
        else:
            print("...")
    else:
        print("Get")
        return render(request, 'request/layout_request.html',)