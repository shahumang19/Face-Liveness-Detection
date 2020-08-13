from os.path import join
from PIL import Image
from FaceEyeDetection import FaceEyeDetectionDlib
import numpy as np
import keras
import cv2
import os, datetime as dt

IMG_SIZE = (24, 24)

def load_face_eye_models(fpath, lpath, rpath):
    face_detector = cv2.CascadeClassifier(fpath)
    left_eye_detector = cv2.CascadeClassifier(lpath)
    right_eye_detector = cv2.CascadeClassifier(rpath)
    return face_detector, left_eye_detector, right_eye_detector


def load_blink_model(wpath="model2.h5"):
    model = keras.models.load_model(wpath)
    return model

def predict_eye(img, model):
    """[summary]

    Arguments:
        img {[type]} -- [description]
        model {[type]} -- [description]

    Returns:
        [type] -- [description]
    """    
    eye_input = img.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255
    prediction = model.predict(eye_input)
    # print(np.argmax(prediction))
    index = np.argmax(prediction)
    # print(prediction)
    if prediction[0][index] > 0.99:
        return str(index)
    else:
        return ""
        

def isBlinking(history, maxFrames):
    """ @history: A string containing the history of eyes status 
         where a '1' means that the eyes were closed and '0' open.
        @maxFrames: The maximal number of successive frames where an eye is closed """
    for i in range(maxFrames):
        # if i == 0:
        #     pattern = '0' + '11'*(i+1) + '0'
        # else:
        #     pattern = '0' + '1'*(i+1) + '0'
        pattern = '0' + '1'*(i+1) + '0'
        if pattern in history:
            return True
    return False