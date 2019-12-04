
from util import (get_ordinal_score, make_vector, get_webcam, get_image, label_img)
import argparse
import cv2
from openpose import pyopenpose as op
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--target_video', type=str, default='./anshiqa_yoga_debut/real_test.mp4') #'./test.mp4'
parser.add_argument('--model_folder', type=str, default='../openpose/models/')
parser.add_argument('--video_output', type=str, default='target_skeleton_real_test.mp4')
parser.add_argument('--target_vector', type=str, default='./complete_target_vector_map_test.txt')
parser.add_argument('--net_resolution', type=str, default='800x800')  #used to be 176x176
parser.add_argument('--cam_width', type=int, default=1920) #1920 original
parser.add_argument('--cam_height', type=int, default=1080)
parser.add_argument('--number_people_max', type=int, default=1)
args = parser.parse_args()

target = cv2.VideoCapture(args.target_video)

# Custom openpose params
params = dict()
params['face'] = False
#params['face_net_resolution'] = '160x160'
params['disable_blending'] = False
params['model_folder'] = args.model_folder
params['net_resolution'] = args.net_resolution
params['number_people_max'] = args.number_people_max
params['display'] = 0
params['disable_multi_thread'] = False
params['model_pose'] = 'BODY_25'

# Start openpose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(
    args.video_output, #'target_skeleton.mp4'
    fourcc,
    10,
    (args.cam_width, args.cam_height)  #args.cam_width, args.cam_height
)

complete_target_vector_map = []

while True:

     target_img = get_image(target, args.cam_width, args.cam_height)
     if target_img is None:
        continue
     target_datum = label_img(opWrapper, target_img)
     ordinal_score = ('', 0.0, (0, 0, 0))
     if type(target_datum.poseKeypoints) == np.ndarray or \
             target_datum.poseKeypoints.shape == (1, 25, 3):
          if target_datum.poseKeypoints.shape:
                 target_coords_vec = make_vector(target_datum.poseKeypoints)
                 complete_target_vector_map.append(target_coords_vec) 

     screen_out = target_datum.cvOutputData
#     cv2.rectangle(screen_out, (10, 30), (600, 120), (255, 255, 255), 3)
#     font = cv2.FONT_HERSHEY_DUPLEX
     out.write(screen_out)
     cv2.imshow("Webcam and Target Image", screen_out)



     key = cv2.waitKey(1)
     if key == ord('q'):
         print(np.shape(complete_target_vector_map))
         with open(args.target_vector, 'wb') as fp:
              pickle.dump(complete_target_vector_map, fp) 
         break


# Clean up
target.release()
out.release()
cv2.destroyAllWindows()



