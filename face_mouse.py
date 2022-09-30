"""
Runs the program without any GUI window
"""
# Importing packages
from matplotlib import testing
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from pyautogui import size
import time
import dlib
import cv2
import mouse
import threading
import math
import argparse
import os
import queue
import sounddevice as sd
import vosk
import sys
import gtts  
import pygame
from gtts import gTTS  
from playsound import playsound 
from multiprocessing import Process
#----------------------------------///////----------------------------------
from imutils import face_utils
from utils import *

import pyautogui as pag
import imutils



#-------------------------------------------------------------------------------------------------------

# Initializing indexes for the features to track as an Ordered Dictionary
# FACIAL_LANDMARKS_IDXS = OrderedDict([
#     ("right_eye", (36, 42)),
#     ("left_eye", (42, 48)),
#     ("nose", (27, 36)),
# ])


# def shape_arr_func(shape, dtype="int"):
#     """
#     Function to convert shape of facial landmark to a 2-tuple numpy array
#     """
#     # Initializing list of coordinates
#     coords = np.zeros((68, 2), dtype=dtype)
#     # Looping over the 68 facial landmarks and converting them
#     # to a 2-tuple of (x, y) coordinates
#     for i in range(0, 68):
#         coords[i] = (shape.part(i).x, shape.part(i).y)
#     # Returning the list of (x, y) coordinates
#     return coords


# def mvmt_func(x):
#     """
#     Function to calculate the move value as fractional power of displacement.
#     This helps to reduce noise in the motion
#     """
#     if x > 1.:
#         return math.pow(x, float(3) / 2)
#     elif x < -1.:
#         return -math.pow(abs(x), float(3) / 2)
#     elif 0. < x < 1.:
#         return 1
#     elif -1. < x < 0.:
#         return -1
#     else:
#         return 0


# def ear_func(eye):
#     """
#     Function to calculate the Eye Aspect Ratio.
#     """
#     # Finding the euclidean distance between two groups of vertical eye landmarks [(x, y) coords]
#     v1 = dist.euclidean(eye[1], eye[5])
#     v2 = dist.euclidean(eye[2], eye[4])
#     # Finding the euclidean distance between the horizontal eye landmarks [(x, y) coords]
#     h = dist.euclidean(eye[0], eye[3])
#     # Finding the Eye Aspect Ratio (E.A.R)
#     ear = (v1 + v2) / (2.0 * h)
#     # Returning the Eye Aspect Ratio (E.A.R)
#     return ear


# # Defining a constant to indicate a blink when the EAR gets less than the threshold
# # Next two constants to specify the number of frames blink has to be sustained
# EYE_AR_THRESH = 0.20
# EYE_AR_CONSEC_FRAMES_MIN = 1
# EYE_AR_CONSEC_FRAMES_MAX = 5

# # Initializing Frame COUNTER and the TOTAL number of blinks in a go


# COUNTER = 0
# TOTAL = 0

# # Initializing Mouse Down Toggle
# isMouseDown = False

# # Initializing Dlib's face detector (HOG-based) and creating the facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Taking the indexes of left eye, right eye and nose
# (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
# (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
# (nStart, nEnd) = FACIAL_LANDMARKS_IDXS["nose"]

# # Initializing the Video Capture from source
# vs = cv2.VideoCapture(0)
# # 1 sec pause to load the VideoStream before running the predictor
# time.sleep(1.0)


# def left_click_func():
#     """
#     Function to handle left clicks via blinking
#     """

#     global isMouseDown
#     global TOTAL
#     # Performs a mouse up event if blink is observed after mouse down event
#     if isMouseDown and TOTAL != 0:
#         mouse.release(button='left')
#         isMouseDown = False

#     else:
#         # Single Click
#         if TOTAL == 1:
#             mouse.click(button='left')
#         # Double Click
#         elif TOTAL == 2:
#             mouse.double_click(button='left')
#         # Mouse Down (to drag / scroll)
#         elif TOTAL == 3:
#             mouse.press(button='left')
#             isMouseDown = True
#     # Resetting the TOTAL number of blinks counted in a go
#     TOTAL = 0


# def right_click_func():
#     """
#     Function to perform right click triggered by blinking
#     """
#     global TOTAL
#     mouse.click(button='right')
#     TOTAL = 0


# # Factor to amplify the cursor movement by.
# sclFact = 4
# firstRun = True

# # Declaring variables to hold the displacement
# # of tracked feature in x and y direction respectively
# global xC
# global yC

# # Setting the initial location for the cursor to the middle of screen
# mouse.move(size()[0] // 2, size()[1] // 2)


# def track_nose(nose):
#     """
#     Function to track the tip of the nose and move the cursor accordingly
#     """
#     global xC
#     global yC
#     global firstRun
#     # Finding the position of tip of nose
#     cx = nose[3][0]
#     cy = nose[3][1]
#     if firstRun:
#         xC = cx
#         yC = cy
#         firstRun = False
#     else:
#         # Calculating distance moved
#         xC = cx - xC
#         yC = cy - yC
#         # Moving the cursor by appropriate value according to calculation
#         mouse.move(mvmt_func(-xC) * sclFact, mvmt_func(yC) * sclFact, absolute=False, duration=0)
#         # Resetting the current position of cursor
#         xC = cx
#         yC = cy
 

# # Looping over video frames

class Both():
    global testingr 
    testingr = 0
    def video_inp(self):
        
        MOUTH_AR_THRESH = 0.3
        MOUTH_AR_CONSECUTIVE_FRAMES = 5
        EYE_AR_THRESH = 0.20
        EYE_AR_CONSECUTIVE_FRAMES = 5
        WINK_AR_DIFF_THRESH = 0.001
        WINK_AR_CLOSE_THRESH = 0.4
        WINK_CONSECUTIVE_FRAMES = 10
        MOUTH_COUNTER = 0
        EYE_COUNTER = 0
        WINK_COUNTER = 0
        INPUT_MODE = False
        EYE_CLICK = False
        LEFT_WINK = False
        RIGHT_WINK = False
        #SCROLL_MODE = False
        ANCHOR_POINT = (0, 0)
        WHITE_COLOR = (255, 255, 255)
        YELLOW_COLOR = (0, 255, 255)
        RED_COLOR = (0, 0, 255)
        GREEN_COLOR = (0, 255, 0)
        BLUE_COLOR = (255, 0, 0)
        BLACK_COLOR = (0, 0, 0)

        shape_predictor = "model/shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_predictor)
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        vid = cv2.VideoCapture(0)
        resolution_w = 1366
        resolution_h = 768
        cam_w = 640
        cam_h = 480
        unit_w = resolution_w / cam_w
        unit_h = resolution_h / cam_h

        while True:
            _, frame = vid.read()
            frame = cv2.flip(frame, 1)
            frame = imutils.resize(frame, width=cam_w, height=cam_h)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            if len(rects) > 0:
                rect = rects[0]
            else:
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                continue
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            mouth = shape[mStart:mEnd]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            nose = shape[nStart:nEnd]
            temp = leftEye
            leftEye = rightEye
            rightEye = temp
            mar = mouth_aspect_ratio(mouth)
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            diff_ear = np.abs(leftEAR - rightEAR)

            nose_point = (nose[3, 0], nose[3, 1])
            mouthHull = cv2.convexHull(mouth)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
            cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
            cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)

            for (x, y) in np.concatenate((mouth, leftEye, rightEye), axis=0):
                cv2.circle(frame, (x, y), 2, GREEN_COLOR, -1)
            if diff_ear > WINK_AR_DIFF_THRESH:

                if leftEAR < rightEAR:
                    if leftEAR < EYE_AR_THRESH:
                        WINK_COUNTER += 1

                        if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                            # pag.click(button='left')

                            WINK_COUNTER = 0

                elif leftEAR > rightEAR:
                    if rightEAR < EYE_AR_THRESH:
                        WINK_COUNTER += 1

                        if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                            # pag.click(button='right')

                            WINK_COUNTER = 0
                else:
                    WINK_COUNTER = 0
            else:
                if ear <= EYE_AR_THRESH:
                    EYE_COUNTER += 1

                    if EYE_COUNTER > EYE_AR_CONSECUTIVE_FRAMES:
                    # SCROLL_MODE = not SCROLL_MODE
                        EYE_COUNTER = 0

                else:
                    EYE_COUNTER = 0
                    WINK_COUNTER = 0

            if mar > MOUTH_AR_THRESH:
                MOUTH_COUNTER += 1

                if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES:
                    INPUT_MODE = not INPUT_MODE
                    MOUTH_COUNTER = 0
                    ANCHOR_POINT = nose_point

            else:
                MOUTH_COUNTER = 0

            if INPUT_MODE:
                cv2.putText(frame, "Lets get started", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
                x, y = ANCHOR_POINT
                nx, ny = nose_point
                w, h = 40, 20
                multiple = 1
                cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), GREEN_COLOR, 2)
                cv2.line(frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)

                dir = direction(nose_point, ANCHOR_POINT, w, h)
                cv2.putText(frame, dir.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
                drag = 30
                if dir == 'right':
                    pag.moveRel(drag, 0)
                elif dir == 'left':
                    pag.moveRel(-drag, 0)
                elif dir == 'up':
                    # if SCROLL_MODE:
                    #     pag.scroll(40)
                    #else
                    pag.moveRel(0, -drag)
                elif dir == 'down':
                    # if SCROLL_MODE:
                    #     pag.scroll(-40)
                    # else:
                    pag.moveRel(0, drag)

            # if SCROLL_MODE:
            #     cv2.putText(frame, 'SCROLL MODE IS ON!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        cv2.destroyAllWindows()
        vid.release()
    def audio_input(self):

        language ='en'
        speech = gTTS(text = "Welcome to drishti, For operations you can say left click , right click , double click. ", lang = language, slow = False)
        speech.save("text.mp3")
        # os.system("start text.mp3")
        def music(music):
            pygame.mixer.init()
            pygame.mixer.music.load(music)
            pygame.mixer.music.play()
        music("text.mp3")
        q = queue.Queue()

        def int_or_str(text):
            """Helper function for argument parsing."""
            try:
                return int(text)
            except ValueError:
                return text

        def callback(indata, frames, time, status):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status, file=sys.stderr)
            q.put(bytes(indata))

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            '-l', '--list-devices', action='store_true',
            help='show list of audio devices and exit')
        args, remaining = parser.parse_known_args()
        if args.list_devices:
            print(sd.query_devices())
            parser.exit(0)
        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            parents=[parser])
        parser.add_argument(
            '-f', '--filename', type=str, metavar='FILENAME',
            help='audio file to store recording to')
        parser.add_argument(
            '-m', '--model', type=str, metavar='MODEL_PATH',
            help='Path to the model')
        parser.add_argument(
            '-d', '--device', type=int_or_str,
            help='input device (numeric ID or substring)')
        parser.add_argument(
            '-r', '--samplerate', type=int, help='sampling rate')
        args = parser.parse_args(remaining)

        try:
            if args.model is None:
                args.model = "model"
            if not os.path.exists(args.model):
                print ("Please download a model for your language from https://alphacephei.com/vosk/models")
                print ("and unpack as 'model' in the current folder.")
                parser.exit(0)
            if args.samplerate is None:
                device_info = sd.query_devices(args.device, 'input')
                # soundfile expects an int, sounddevice provides a float:
                args.samplerate = int(device_info['default_samplerate'])

            model = vosk.Model(args.model)

            if args.filename:
                dump_fn = open(args.filename, "wb")
            else:
                dump_fn = None

            with sd.RawInputStream(samplerate=args.samplerate, blocksize = 8000, device=args.device, dtype='int16',
                                    channels=1, callback=callback):
                    print('#' * 80)
                    print('Press Ctrl+C to stop the recording')
                    print('#' * 80)
     
                    rec = vosk.KaldiRecognizer(model, args.samplerate)
                    while True:
                        data = q.get()
                        if rec.AcceptWaveform(data):
                            g = rec.Result()[14:-3]
        
                            if ((str(g))=="left click"):
                                
                                mouse.click(button='left')
                                
                   
                            elif((str(g))=="right click"):
                                mouse.click(button='right')

                            elif((str(g))=="double click"):
                                mouse.double_click(button='left')

                   

                               
                                    
                                # print(testingr)
                            elif((str(g))=="stop mouse"):
                                sys.exit()
                       
                            
                     
                        if dump_fn is not None:
                            dump_fn.write(data)

        except KeyboardInterrupt:
            print('\nDone')
            parser.exit(0)
        except Exception as e:
            parser.exit(type(e).__name__ + ': ' + str(e))
if __name__ == '__main__':
    drishti = Both()
    
    Process(target=drishti.audio_input).start()
    Process(target=drishti.video_inp).start()
    


cv2.destroyAllWindows()
