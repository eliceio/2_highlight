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
    print("determine_dlib")
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

    frame_seq = start*fps*duration
    input_movie.set(1, frame_seq) # 1, start_frame

    while True:
        # Grab a single frame of video
        # idx == 0 or 1
        if idx != 2:
            ret, frame = input_movie.read()
            frame_number += 1
#             print(frame_number,"=11111111111111111111111")
        # idx == 2
        else:
            for i in range(frame_duration_fail):
                if i != 0:
                    ret, frame = input_movie.read()
                    frame_number += 1
#                 print("Writing frame {} / {}".format(frame_number, length))
#                 output_movie.write(frame)
                idx = 1
#             print(frame_number,"=2222222222222222222222")
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
            
            if match[0]:
                name = "detect"
                count += 1
                
#             elif match[1]:
#                 name = 'YOU'
#             print(name)
            
            print(count)
            if count > 5:
                print("True = ", count)
                
                return True
            elif  frame_number > ((end - start)*fps*duration) / 10: # half of song
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
                      
                # Label the results
#                 for (top, right, bottom, left), name in zip(face_locations, face_names):
#                     if not name:
#                         continue
#                     idx = 1
#                     # Draw a box around the face

#                     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#                     # Draw a label with a name below the face
#                     cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
#                     font = cv2.FONT_HERSHEY_DUPLEX
#                     cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

                # Write the resulting image to the output video file
#                 print("Writing frame {} / {}".format(frame_number, length))
#                 output_movie.write(frame)
#             print("+10", idx)


    # All done!
    input_movie.release()
#     output_movie.release()
    cv2.destroyAllWindows()
