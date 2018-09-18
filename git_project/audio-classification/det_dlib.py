import face_recognition
import cv2
import sys, os
# import numpy as np
sys.path.append(os.pardir)

def determine_dlib(file_name, start, end, picture_path, duration, fps = 30, frame_duration_suc = 10, frame_duration_fail = 30):
    
    ## using dlib
    # codec 
    # This is a demo of running face recognition on a video file and saving the results to a new video file.
    #
    # PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
    # OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
    # specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

    # Open the input movie file
    input_movie = cv2.VideoCapture(file_name)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT)) #frames

#     w = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH));
#     h = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT));
 
#     # Create an output movie file (make sure resolution/frame rate matches input video!)
#     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#     output_movie = cv2.VideoWriter('./sample1_1111.mp4', fourcc, 29.97, (w, h))

    al_image = face_recognition.load_image_file(picture_path)
    al_face_encoding = face_recognition.face_encodings(al_image)[0]

#     you_image = face_recognition.load_image_file("./you.jpg")
#     you_face_encoding = face_recognition.face_encodings(you_image)[0]

    known_faces = [
    #    lmm_face_encoding,
        al_face_encoding,
    #    hy_face_encoding,
#         you_face_encoding,

    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    frame_number = 0
    idx = 0
    count = 0
    
    count_fail = 0
    pre_count = 100
    frame_seq = start*fps*duration
    input_movie.set(1, frame_seq) # 1, start_frame

    while True:
        # Grab a single frame of video
        # idx == 0 or 1
        if idx != 2:
            ret, frame = input_movie.read()
            frame_number += 1
        # idx == 2
        else:
            for i in range(frame_duration_fail):
                if i != 0:
                    ret, frame = input_movie.read()
                    frame_number += 1

                idx = 1
            continue
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
            
            if pre_count == count:
                count_fail += 1
            else:
                pre_count = count
                
            if match[0]:
                name = "detect"
                count += 1
                      
            print(count)
            if count > 5:
                print("True = ", count)
                
                return True

            elif  count_fail > 15: # half of song
                print("False = ", frame_number)
                return False
            face_names.append(name)

        if face_locations == []:
            idx = 2
            continue
        else:
            for i in range(frame_duration_suc):
                if i != 0:
                    ret, frame = input_movie.read()
                    frame_number += 1
                      
    # All done!
    input_movie.release()
#     output_movie.release()
    cv2.destroyAllWindows()
