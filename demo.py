
from subprocess import call
import signal
import pickle
from os import system
from playsound import playsound
from PyQt5.QtWidgets import QMainWindow, QApplication
import sys
from PyQt5.QtGui import QIcon, QImage, QPalette, QBrush,QColor,QFont
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import QSize, QTimer
from PyQt5 import QtCore
import numpy as np
import pandas as pd
from scoreGUI import App
import sys
import cv2
import argparse
import numpy as np
import time
from keras.models import load_model
from util import (final_transform, crop_image,get_ordinal_score, make_vector, get_webcam, get_image, label_img, video_cv, transform_image)
import freenect

sys.path.append('/usr/local/python')

from openpose import pyopenpose as op

#Allows for the closing of PyQt
signal.signal(signal.SIGINT,signal.SIG_DFL)

#Back straighten 1000 - 1100
#keep back straight 1550 - 1600

#align heels 1900 - 2000

#elbow straighten 2213

#good form 2468 - 2619 



# Flags
parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, default='../openpose/models/')
parser.add_argument('--sound_folder', type=str, default='./sounds/')
parser.add_argument('--target_video', type=str, default='./test.mp4')
parser.add_argument('--skeleton_video', type=str, default='./anshiqa_yoga_debut/yoga_vid_new_1.mp4')
#parser.add_argument('--skeleton_video', type=str, default='target_skeleton_real_test.mp4')
parser.add_argument('--target_vector', type=str, default='./complete_target_vector_map_test.txt')
parser.add_argument('--model', type=str, default='nano')
parser.add_argument('--cam_width', type=int, default=1920) #1920 original
parser.add_argument('--cam_height', type=int, default=1080) #1080 
parser.add_argument('--number_people_max', type=int, default=1)
args = parser.parse_args()

#Specifically for certain models
if args.model == 'laptop':
   parser.add_argument('--net_resolution', type=str, default='112x112')  #used to be 176x176
if args.model == 'desktop':
   parser.add_argument('--net_resolution', type=str, default='528x528')  #used to be 176x176
if not args.model == 'laptop' and not args.model == 'desktop':
   parser.add_argument('--net_resolution', type=str, default='656x368')  #used to be 656x368
args = parser.parse_args()

# Custom openpose params
params = dict()
params['face'] = False
params['hand'] = False
#params['body'] = 2
#params['output_resolution'] = '1920x1080'
#params['identification'] = True
params['fullscreen'] = True
#params['face_net_resolution'] = '16x16'
params['disable_blending'] = True
params['model_folder'] = args.model_folder
params['net_resolution'] = args.net_resolution
params['number_people_max'] = args.number_people_max
#params['scale_number'] = 4
#params['scale_gap'] = 0.25
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

counter = 0
counter_len = 5
lst = []
average_score = 0
current_score = 0
played_voice = 0 
#sys.stdout.flush()
good_form_counter = 0
pretty_good = 0

while True:
    for vector in b:
        counter += 1
        if counter == 286 or counter == 631 or counter == 1101 or counter == 1601 or counter == 2001: 
             played_voice = 0 
        print(counter)
        frames += 1

        # Get images
        # video_cv(freenect.sync_get_video()[0] gets single frame of the kinect 
    #    webcam_img = get_image(video_cv(freenect.sync_get_video()[0]), args.cam_width, args.cam_height)
        webcam_img = transform_image(video_cv(freenect.sync_get_video()[0]), args.cam_width, args.cam_height)
#        webcam_img = transform_image(video_cv(freenect.sync_get_video()[0]), args.cam_height, args.cam_width)
 #    target_img = get_image(target, args.cam_width, args.cam_height)
#        webcam_img = video_cv(freenect.sync_get_video()[0])
        skeleton_img = get_image(skeleton, args.cam_width, args.cam_height)
#        print(skeleton_img.shape)
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
     
           # records previous similiarities
           lst.append(similarity_score)
             
           print('recordede sim score', counter, similarity_score)
           
           # Use modulus to find the frames in multiples of 5 and then find an average of the last 5 frames
           if counter % counter_len == 0 and counter > 0:
                newlst = lst[counter-counter_len:counter]
                average_score = sum(newlst)/counter_len
                print("aavg",average_score)
                ordinal_score = get_ordinal_score(average_score)
                if counter >= 2468 and counter <= 2619:
                     good_score = get_ordinal_score(average_score)
                print(ordinal_score)
                if  ordinal_score[0] == 'Perfect!':
                     current_score += 10 
                if  ordinal_score[0] == 'Nice try!':
                     current_score += 5
                if  ordinal_score[0] == 'Try harder!!':
                     current_score += 1
                print("curreeent scoreee",current_score,ordinal_score) 
           if counter >= counter_len:
                  ordinal_score = get_ordinal_score(average_score)
                  print("average",average_score)


     # Previously this was used to overlay the vector of the user and the skeleton
         
        # Concatenate webcam and target video
    #    screen_out = np.concatenate((webcam_datum.cvOutputData,
    #                                 skeleton_img),
    #                                 axis=1)
        # Capture frame-by-frame
#        screen_out = skeleton_img
        # Show the screen of the user skeleton
        screen_out = webcam_datum.cvOutputData
        # Add overlay to show results
        overlay = screen_out.copy()
        cv2.rectangle(overlay, (0, 0), (args.cam_width, args.cam_height),  # previously args.cam_width // 2 now args.cam_width
                      ordinal_score[2], -1)
        screen_out = cv2.addWeighted(overlay, ordinal_score[1],
                                     screen_out,
                                     1 - ordinal_score[1], 0,
                                     screen_out)

       # Add overlay to show ideal body **
#        print(screen_out.shape)
#        overlay = webcam_datum.cvOutputData 
        overlay = skeleton_img
 #       overlay = cv2.Canny(skeleton_img,150,200) 
    #    overlay = webcam_img
#        print(overlay.shape)
    #    overlay = target_datum.cvOutputData

        # Select opacity of the user skeleton and ideal skeleton (Currently 50% and 50%)
        screen_out = cv2.addWeighted(overlay, 0.5,
                                      screen_out,
                                      0.5, 0,
                                      screen_out)


 #       overlay = cv2.Canny(skeleton_img,150,200)
        # Draw a vertical white line with thickness of 10 px
       # cv2.line(screen_out, (args.cam_width // 2, 0),
       #          (args.cam_width // 2, args.cam_height),
       #          (255, 255, 255), 10)

        screen_out = final_transform(screen_out,args.cam_width, args.cam_height)
     
       # Display comment
        cv2.rectangle(screen_out, (60, 30), (1000, 120), (255, 255, 255), 3)
        font = cv2.FONT_HERSHEY_TRIPLEX
#        if not ordinal_score[0]:
#             ordinal_score = 'Try Harder!!'
#             print(ordinal_score[0])
     
#Back straighten 1000 - 1100
#keep back straight 1550 - 1600

#align heels 1900 - 2000

#elbow straighten 2213

#good form 2468 - 2619

        # good form counter
          
        if counter >= 2468 and counter <= 2619:
             if ordinal_score[0] == 'Perfect!':
                  good_form_counter += 1
             else:
                  good_form_counter = 0
        if good_form_counter == 10:
             pretty_good = 1 
        print('voice_played',played_voice)
     # Voice Correction Protocol (VCP) 
        if played_voice == 0 and ordinal_score[0] != 'Perfect!' and counter >= 235 and counter <= 285:
              system('nohup python3 ~/PycharmProjects/yogamirror/sounds/straighten_arm.py &')
              played_voice += 1
        if played_voice == 0 and ordinal_score[0] != 'Perfect!' and counter >= 557 and counter <= 630:
              system('nohup python3 ~/PycharmProjects/yogamirror/sounds/straighten_arm.py &')
              played_voice += 1
        if played_voice == 0 and ordinal_score[0] != 'Perfect!' and counter >= 1000 and counter <= 1100:
              system('nohup python3 ~/PycharmProjects/yogamirror/sounds/back_strt.py &')
              played_voice += 1
        if played_voice == 0 and ordinal_score[0] != 'Perfect!' and counter >= 1550 and counter <= 1600:
              system('nohup python3 ~/PycharmProjects/yogamirror/sounds/back.py &')
              played_voice += 1
        if played_voice == 0 and ordinal_score[0] != 'Perfect!' and counter >= 1900 and counter <= 2000:
              system('nohup python3 ~/PycharmProjects/yogamirror/sounds/align_heels.py &')
              played_voice += 1
        if played_voice == 0 and pretty_good == 1 and counter >= 2468 and counter <= 2619:
              system('nohup python3 ~/PycharmProjects/yogamirror/sounds/good_form.py &')
              played_voice += 1


        highscore = round(current_score/2)
     #   highscore = current_score
        #print(highscore)
        #print('current', current_score)
        #print('counter', counter) 
        #print(current_score*(counter-current_score))
      #  if highscore % 5 != 0: 
      #          highscore += (5 - highscore % 5)
        cv2.putText(screen_out, '  ' + ordinal_score[0], (10, 100), font, 2, (255, 255, 255), 4, cv2.LINE_AA)
        if ordinal_score[0]:
             cv2.putText(screen_out, '                  Score: ' + str(highscore), (10,100), font, 2,(255,255,255),4,cv2.LINE_AA) #34 41, 155

        # Record Video
    #    out.write(screen_out)
#        screen_out = crop_image(screen_out, 1920, 1080)

       # screen_out = final_transform(screen_out,args.cam_width, args.cam_height)
        # Display img
        cv2.namedWindow('Nyoga')
        cv2.moveWindow('Nyoga',1930,-1930)
        cv2.imshow('Nyoga', screen_out)
#        cv2.namedWindow("Webcam and Target Image",cv2.WND_PROP_FULLSCREEN)
 #       cv2.setWindowProperty("Webcam and Target Image",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Check for quit
        key = cv2.waitKey(1)
          

         #Manually turns off after a certain number 
        if key == ord('q') or counter == 2776: #2776
            highscore = round(current_score/2)
            if highscore % 5 != 0: 
                highscore += (5 - highscore % 5)
            print("Highscoreeeeeeeeeeeeeeeee",highscore)
            # Opens leaderboard
            app = QApplication(sys.argv)
            ex = App(curr_score=highscore)
            ex.show()
            sys.exit(app.exec_())
            ex.quit()
            print("hi")
            break

        # Print frame rate
        if time.time() - start >= 1:
            framerate = frames
            print('Framerate: ', framerate)
            frames = 0
            start = time.time()

    # Clean up
    skeleton.release()
   # webcam.release()
    #target.release()
    #out.release()
    cv2.destroyAllWindows()
