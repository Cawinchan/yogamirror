from __future__ import print_function
import cv2
import argparse
import numpy

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='yoga.mp4')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()
capture = cv2.VideoCapture(args.input)  #cv.samples.findFileOrKeep(args.input)

myvideo = cv2.VideoWriter("tester.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 30 \
                          , (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                          False)

if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    fgMask = backSub.apply(frame)
    
    
    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    print(fgMask.shape)
    myvideo.write(fgMask)
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)
    
    
    
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
