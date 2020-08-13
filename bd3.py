from os.path import join
from PIL import Image
from FaceEyeDetection import FaceEyeDetectionDlib
import numpy as np
import keras
import cv2
import os, datetime as dt

IMG_SIZE = (24, 24)


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

    # if prediction < 0.1:
    #     prediction = 0 # 0 - eyes close
    # elif prediction > 0.9:
    #     prediction = 1 # 1 - eyes open
    # else:
    #     prediction = -1 # -1 - mid
    # return prediction

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

if __name__ == "__main__":
    hist = ""
    blink_count = 0

    count = 0
    fd = FaceEyeDetectionDlib(join("models","shape_predictor_68_face_landmarks.dat"))
    model = load_blink_model( os.path.join("models","eb1.h5"))
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture("blink-detection\\11.MP4")


    try:
        while True:
            ret, frame = cap.read()
            if ret:
                # frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects, faces = fd.detect_faces(gray)
                
                if len(faces)>0:
                    x,y,w,h = faces[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    face = frame[y:y+h,x:x+w]
                    gray_face = gray[y:y+h,x:x+w]

                    # left_face = face[y:y+h, x+int(w/2):x+w]
                    # left_face_gray = gray_face[y:y+h, x+int(w/2):x+w]

                    # right_face = face[y:y+h, x:x+int(w/2)]
                    # right_face_gray = gray_face[y:y+h, x:x+int(w/2)]

                    # Detect eyes
                    face_landmarks = fd.detect_eyes(gray, [rects[0]])
                    left_eye_pos = face_landmarks[0][1]
                    right_eye_pos = face_landmarks[0][2]

                    eye_status = ''
                    # prediction = ''

                    if len(left_eye_pos) > 0:
                        ex, ey, ew, eh = left_eye_pos
                        color = (0,255,0)
                        eye_img = cv2.resize(gray[ey:ey+eh,ex:ex+ew], dsize=IMG_SIZE)
                        prediction = predict_eye(eye_img, model)
                        if prediction != '':
                            eye_status= prediction
                        cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),color,2)

                    if len(right_eye_pos) > 0:
                        ex, ey, ew, eh = right_eye_pos
                        color = (0,255,0)
                        if eye_status != '1':
                            eye_img = cv2.resize(gray[ey:ey+eh,ex:ex+ew], dsize=IMG_SIZE)
                            prediction = predict_eye(eye_img, model)
                            if prediction != '':
                                eye_status= prediction
                        cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),color,2)

                    hist += eye_status
                    # hist += prediction
                    
                    if isBlinking(hist, 4):
                        # print(dt.datetime.now(),"Blinked")
                        blink_count += 1
                        # print(hist)
                        hist = ""

                cv2.putText(frame, f"Blinks :  {blink_count}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
                cv2.imshow("eye blink detection",frame)
                if cv2.waitKey(1) == 13:
                    break
            else:
                break

    except Exception as e:
        print(f"ERROR : {e}")
    finally:
        print(f"Blink : {blink_count}")
        cap.release()
        cv2.destroyAllWindows()
