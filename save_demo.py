import sys
import cv2
import argparse
import numpy as np
import time
from keras.models import load_model
from util import (get_ordinal_score, make_vector, get_webcam, get_image, label_img)

sys.path.append('/usr/local/python')

from openpose import pyopenpose as op

import time
start = time.process_time()
# Flags
parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, default='../openpose/models/')
parser.add_argument('--target_video', type=str, default='./test.mp4')
parser.add_argument('--net_resolution', type=str, default='96x96')  #used to be 176x176
parser.add_argument('--cam_width', type=int, default=1920) #1920 original
parser.add_argument('--cam_height', type=int, default=1080)
parser.add_argument('--number_people_max', type=int, default=1)

args = parser.parse_args()

# Custom openpose params
params = dict()
params['face'] = False
params['face_net_resolution'] = '160x160'
params['disable_blending'] = True
params['model_folder'] = args.model_folder
params['net_resolution'] = args.net_resolution
params['number_people_max'] = args.number_people_max
params['display'] = 0
params['disable_multi_thread'] = False
params['model_pose'] = 'BODY_25'

print(time.process_time() - start)
start = time.process_time()


# Start openpose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

print("open prose time",time.process_time() - start)
start = time.process_time()

# Start streams
webcam = get_webcam(args.cam_width, args.cam_height)
target = cv2.VideoCapture(args.target_video)
display_vid = cv2.VideoCapture('./foreground.mp4')

print("start stream",time.process_time() - start)
start = time.process_time()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(
    'test_output.mp4',
    fourcc,
    30,
    (args.cam_width, args.cam_height)  #args.cam_width, args.cam_height
)

print("videowriter stream",time.process_time() - start)
start = time.process_time()

# Setup framerate params
frames = 0
framerate = 0
start = time.time()
time.sleep(2)  # delay to wait for detection
model = load_model('ComparatorNet.h5',compile=False)

print("load model",time.process_time() - start)
start = time.process_time()


while True:
    frames += 1

    print("start loop",time.process_time() - start)
    start = time.process_time()

    # Get images
    webcam_img = get_image(webcam, args.cam_width, args.cam_height)
    target_img = get_image(target, args.cam_width, args.cam_height)

    if webcam_img is None or target_img is None:
        continue
    
    print("get image",time.process_time() - start)
    start = time.process_time()
    
    # Label images
    webcam_datum = label_img(opWrapper, webcam_img)
    target_datum = label_img(opWrapper, target_img)

    print("label image",time.process_time() - start)
    start = time.process_time()


    # Check if OpenPose managed to label
    ordinal_score = ('', 0.0, (0, 0, 0))
    if type(webcam_datum.poseKeypoints) == np.ndarray and \
       webcam_datum.poseKeypoints.shape == (1, 25, 3):

        if type(target_datum.poseKeypoints) == np.ndarray or \
             target_datum.poseKeypoints.shape == (1, 25, 3):
            print("start ETL",time.process_time() - start)
            start = time.process_time()

            # Scale, transform, normalize, reshape, predict
            coords_vec = make_vector(webcam_datum.poseKeypoints)
            if target_datum.poseKeypoints.shape:
                 target_coords_vec = make_vector(target_datum.poseKeypoints)
                 print(target_coords_vec.shape)
                 input_vec = np.concatenate([coords_vec, target_coords_vec]).flatten()
                 similarity_score = model.predict(input_vec.reshape((1, -1)))
                 ordinal_score = get_ordinal_score(similarity_score)
                 print("END ETL", time.process_time() - start)
                 start = time.process_time()
    print("end of loop", time.process_time() - start)
    start = time.process_time()


    # Concatenate webcam and target video
#    screen_out = np.concatenate((webcam_datum.cvOutputData,
#                                 target_datum.cvOutputData),
#                                axis=1)

    screen_out = webcam_datum.cvOutputData
    print("output screen", time.process_time() - start)
    # Add overlay to show results
    overlay = screen_out.copy()
    cv2.rectangle(overlay, (0, 0), (args.cam_width, args.cam_height),  # previously args.cam_width // 2 now args.cam_width
                  ordinal_score[2], -1)
    screen_out = cv2.addWeighted(overlay, ordinal_score[1],
                                 screen_out,
                                 1 - ordinal_score[1], 0,
                                 screen_out)

    print("output screen 1", time.process_time() - start)
    start = time.process_time()
   # Add overlay to show ideal body **
#    overlay = target_img
    overlay = target_img
#    overlay = target_datum.cvOutputData
    screen_out = cv2.addWeighted(overlay, ordinal_score[1],   # Creates the different feedback colours
                                 screen_out,
                                 1 - ordinal_score[1], 0,
                                 screen_out)

    print("output screen 2", time.process_time() - start)
    start = time.process_time()
    # Draw a vertical white line with thickness of 10 px
   # cv2.line(screen_out, (args.cam_width // 2, 0),
   #          (args.cam_width // 2, args.cam_height),
   #          (255, 255, 255), 10)

    # Display comment
    cv2.rectangle(screen_out, (10, 30), (600, 120), (255, 255, 255), 3)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(screen_out, ' ' + ordinal_score[0], (10, 100), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    print("output screen 3", time.process_time() - start)
    start = time.process_time()
    # Record Video
    out.write(screen_out)
    print("record video screen", time.process_time() - start)
    start = time.process_time()


    # Display img
    cv2.imshow("Webcam and Target Image", screen_out)

    print("display screen", time.process_time() - start)
    start = time.process_time()

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
webcam.release()
target.release()
out.release()
cv2.destroyAllWindows()
