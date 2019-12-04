import pickle
import sys
import cv2
import argparse
import numpy as np
import time
from keras.models import load_model
from util import (get_ordinal_score, make_vector, get_webcam, get_image, label_img, video_cv, transform_image)
import freenect

sys.path.append('/usr/local/python')

from openpose import pyopenpose as op

# Flags
parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, default='../openpose/models/')
parser.add_argument('--target_video', type=str, default='./test.mp4')
<<<<<<< HEAD
parser.add_argument('--model', type=str, default='nano')
parser.add_argument('--skeleton_video', type=str, default='./target_skeleton_real_test.mp4')
parser.add_argument('--target_vector', type=str, default='./complete_target_vector_map_test.txt')
parser.add_argument('--cam_width', type=int, default=1920) #1920 original
parser.add_argument('--cam_height', type=int, default=1080) #1080
=======
parser.add_argument('--skeleton_video', type=str, default='./target_skeleton_real_test.mp4')
parser.add_argument('--target_vector', type=str, default='./complete_target_vector_map_test.txt')
parser.add_argument('--net_resolution', type=str, default='112x112')  #used to be 176x176
parser.add_argument('--cam_width', type=int, default=1920) #1920 original
parser.add_argument('--cam_height', type=int, default=1080) #1080 
>>>>>>> 77365638b9709824d6489d14a071ea8a024d3248
parser.add_argument('--number_people_max', type=int, default=1)

args = parser.parse_args()

if args.model == 'test':
     parser.add_argument('--net_resolution', type=str, default='16x16')
if args.model == 'laptop':
     parser.add_argument('--net_resolution', type=str, default='112x112')
if args.model == 'desktop':
     parser.add_argument('--net_resolution', type=str, default='800x800')
if args.model != 'test' and args.model != 'latop' and args.model != 'desktop':
    parser.add_argument('--net_resolution', type=str, default='64x64')
args = parser.parse_args()

# Custom openpose params
params = dict()
params['face'] = False
params['face_net_resolution'] = '16x16'
params['disable_blending'] = True
params['model_folder'] = args.model_folder
params['net_resolution'] = args.net_resolution
params['number_people_max'] = args.number_people_max
params['display'] = 1
params['disable_multi_thread'] = True
params['model_pose'] = 'BODY_25'

# Start openpose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


# Start streams
#webcam = get_webcam(args.cam_width, args.cam_height))
#webcam = cv2.VideoCapture(0)
#target = cv2.VideoCapture(args.target_video)
skeleton = cv2.VideoCapture(args.skeleton_video)

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter(
#    'test_output.mp4',
#    fourcc,
#    30,
#    (args.cam_width, args.cam_height)  #args.cam_width, args.cam_height
#)

# Setup framerate params
frames = 0
framerate = 0
start = time.time()
#time.sleep(2)  # delay to wait for detection ####################TAKE NOTE MIGHT NEED
model = load_model('ComparatorNet.h5',compile=False)


with open(args.target_vector, "rb") as fp:
     b = pickle.load(fp)

while True:
    for vector in b:
        frames += 1

        # Get images
    #    webcam_img = get_image(video_cv(freenect.sync_get_video()[0]), args.cam_width, args.cam_height)
        webcam_img = transform_image(video_cv(freenect.sync_get_video()[0]), args.cam_width, args.cam_height)
    #    target_img = get_image(target, args.cam_width, args.cam_height)
       # webcam_img = video_cv(freenect.sync_get_video()[0])
        skeleton_img = get_image(skeleton, args.cam_width, args.cam_height)
        if webcam_img is None:
            continue

        # Label images
        webcam_datum = label_img(opWrapper, webcam_img)

        # Check if OpenPose managed to label
        ordinal_score = ('', 0.0, (0, 0, 0))
        if type(webcam_datum.poseKeypoints) == np.ndarray and \
           webcam_datum.poseKeypoints.shape == (1, 25, 3):

           # Scale, transform, normalize, reshape, predict
           coords_vec = make_vector(webcam_datum.poseKeypoints)
           input_vec = np.concatenate([coords_vec, vector]).flatten()
           similarity_score = model.predict(input_vec.reshape((1, -1)))
           ordinal_score = get_ordinal_score(similarity_score)


        # Concatenate webcam and target video
    #    screen_out = np.concatenate((webcam_datum.cvOutputData,
    #                                 skeleton_img),
    #                                 axis=1)
        # Capture frame-by-frame
        screen_out = skeleton_img
     #   print(skeleton_img.shape)
        # Add overlay to show results
        overlay = screen_out.copy()
        cv2.rectangle(overlay, (0, 0), (args.cam_width, args.cam_height),  # previously args.cam_width // 2 now args.cam_width
                      ordinal_score[2], -1)
        screen_out = cv2.addWeighted(overlay, ordinal_score[1],     # Creates the different feedback colours
                                     screen_out,
                                     1 - ordinal_score[1], 0,
                                     screen_out)

       # Add overlay to show ideal body **
    #    overlay = target_img
        overlay = webcam_datum.cvOutputData
    #    overlay = webcam_img
    #    print(overlay.shape)
    #    overlay = target_datum.cvOutputData
        screen_out = cv2.addWeighted(overlay,1 - ordinal_score[1],
                                     screen_out,
                                     ordinal_score[1], 0,
                                     screen_out)

        # Draw a vertical white line with thickness of 10 px
       # cv2.line(screen_out, (args.cam_width // 2, 0),
       #          (args.cam_width // 2, args.cam_height),
       #          (255, 255, 255), 10)

        # Display comment
        cv2.rectangle(screen_out, (10, 30), (600, 120), (255, 255, 255), 3)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(screen_out, ' ' + ordinal_score[0], (10, 100), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

        # Record Video
    #    out.write(screen_out)


        # Display img
        cv2.imshow("Webcam and Target Image", screen_out)


        # Check for quit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        # Print frame rate
        if time.time() - start >= 1:
            framerate = frames
            print('Framerate: ', framerate)
            frames = 0
            start = time.time()

    # Clean up
    skeleton.release()
    #webcam.release()
    #target.release()
    #out.release()
    cv2.destroyAllWindows()
