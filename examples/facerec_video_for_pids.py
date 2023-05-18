import face_recognition
import cv2
import numpy as np

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
input_movie = cv2.VideoCapture("xiaoy.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (640, 360))
width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second=input_movie.get(cv2.CAP_PROP_FPS)
if frames_per_second is not None:
    output_movie = cv2.VideoWriter('xiaoy_fr4.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frames_per_second, (width, height))

# Load some sample pictures and learn how to recognize them.
'''
xd_image = face_recognition.load_image_file("xiaoding.jpg")
xd_face_encoding = face_recognition.face_encodings(xd_image)[0]

xy_image = face_recognition.load_image_file("xiaoyang.jpg")
xy_face_encoding = face_recognition.face_encodings(xy_image)[0]

dy_image = face_recognition.load_image_file("dayang.jpg")
dy_face_encoding = face_recognition.face_encodings(dy_image)[0]
'''

known_faces = [
    face_recognition.face_encodings(face_recognition.load_image_file("xiaoding.jpg"))[0],
    face_recognition.face_encodings(face_recognition.load_image_file("xiaoding2.jpg"))[0],
    face_recognition.face_encodings(face_recognition.load_image_file("xiaoyang.jpg"))[0],
    face_recognition.face_encodings(face_recognition.load_image_file("xiaoyang2.jpg"))[0],
    face_recognition.face_encodings(face_recognition.load_image_file("dayang.jpg"))[0],
    face_recognition.face_encodings(face_recognition.load_image_file("dayang2.jpg"))[0]
]
known_names = [
    "Xiaoding",
    "Xiaoding",
    "Xiaoyang",
    "Xiaoyang",
    "Dayang",
    "Dayang"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1
    '''
    if frame_number % 3 != 0:
        continue
    if frame_number > 1000:
        break
    '''
    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_frame = frame[:, :, ::-1]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    dists_list = []
    if len(face_encodings) == 0:
        continue
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        # match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
        dists = face_recognition.face_distance(known_faces, face_encoding)
        dists_list.append(dists) # ndets x nfaces
    dists_mat = np.array(dists_list).transpose() # nfaces x ndets
    
    merge_dists = {}
    for i, name in enumerate(known_names): # merge comparisons for the same name with multiple samples
        if name not in merge_dists:
            merge_dists[name] = dists_mat[i]
        else:
            merge_dists[name] = np.concatenate((merge_dists[name], dists_mat[i])) # n_unique_faces x (ndets*nsamples)

    ndets = len(face_encodings)
    face_names = [None] * ndets # track all detections
    face_dists = [1] * ndets
    for name, m_dists in merge_dists.items(): # enumerate all names and their merged distance list
        if np.min(m_dists) <= 0.5: # check all detections for current name, if any match (<=0.5) exists
            mdet_id = np.argmin(m_dists) # get the best detection id
            dist = m_dists[mdet_id] # get the distance for the best detection
            det_id = mdet_id % ndets # one name may have multiple samples
            if not face_names[det_id] or face_dists[det_id] > dist: # if hit firstly or with smaller distance, update
                face_names[det_id] = name
                face_dists[det_id] = dist

    # Label the results
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

# All done!
input_movie.release()
cv2.destroyAllWindows()
