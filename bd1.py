from PIL import Image
import numpy as np
import keras
import cv2
import os, datetime as dt


IMG_SIZE = 24

def load_face_eye_models(fpath, lpath, rpath):
    face_detector = cv2.CascadeClassifier(fpath)
    left_eye_detector = cv2.CascadeClassifier(lpath)
    right_eye_detector = cv2.CascadeClassifier(rpath)
    return face_detector, left_eye_detector, right_eye_detector



def load_blink_model(jpath= 'model.json', wpath="model.h5"):
    """Loads eye open-close classification model

        Keyword Arguments:
            jpath {str} -- json file path (default: {'model.json'})
            wpath {str} -- h5 weights file path (default: {"model.h5"})

        Returns:
            keras.model -- returns keras model file
    """  
    json_file = open(jpath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(wpath)
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model

def predict_eye(img, model):
    """[summary]

    Arguments:
        img {[type]} -- [description]
        model {[type]} -- [description]

    Returns:
        [type] -- [description]
    """    
	# img = Image.fromarray(img, 'RGB').convert('L')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE)).astype('float32')
    img /= 255
    img = img.reshape(1,IMG_SIZE,IMG_SIZE,1)
    prediction = model.predict(img)
    if prediction < 0.1:
        prediction = 0 # 0 - eyes close
    elif prediction > 0.9:
        prediction = 1 # 1 - eyes open
    else:
        prediction = -1 # -1 - mid
    return prediction

def isBlinking(history, maxFrames):
    """ @history: A string containing the history of eyes status 
         where a '1' means that the eyes were closed and '0' open.
        @maxFrames: The maximal number of successive frames where an eye is closed """
    for i in range(maxFrames):
        pattern = '1' + '0'*(i+1) + '1'
        if pattern in history:
            return True
    return False

if __name__ == "__main__":
    hist = ""
    blink_count = 0
    fpath = 'models\\haarcascade_frontalface_alt.xml'
    lpath = 'models\\haarcascade_lefteye_2splits.xml'
    rpath ='models\\haarcascade_righteye_2splits.xml'
    face_detector, left_eye_detector, right_eye_detector = load_face_eye_models(fpath, lpath, rpath)
    model = load_blink_model('models\\model.json', "models\\model.h5")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                # frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(50, 50),flags=cv2.CASCADE_SCALE_IMAGE)
                
                if len(faces)>0:
                    x,y,w,h = faces[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    face = frame[y:y+h,x:x+w]
                    gray_face = gray[y:y+h,x:x+w]

                    left_face = face[y:y+h, x+int(w/2):x+w]
                    left_face_gray = gray_face[y:y+h, x+int(w/2):x+w]

                    right_face = face[y:y+h, x:x+int(w/2)]
                    right_face_gray = gray_face[y:y+h, x:x+int(w/2)]

                    # Detect the left eye
                    left_eye_pos = left_eye_detector.detectMultiScale(gray_face,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)

                    # Detect the right eye
                    right_eye_pos = right_eye_detector.detectMultiScale(gray_face,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
                    print(left_eye_pos, right_eye_pos)

                    eye_status = '1'
                    if len(left_eye_pos) > 0:
                        ex,ey,ew,eh = left_eye_pos[0]
                        color = (0,255,0)
                        prediction = predict_eye(face[ey:ey+eh,ex:ex+ew], model)
                        if prediction == 0:
                            eye_status='0'
                        cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),color,2)

                    if len(right_eye_pos) > 0:
                        ex,ey,ew,eh = right_eye_pos[0]
                        color = (0,255,0)
                        prediction = predict_eye(face[ey:ey+eh,ex:ex+ew], model)
                        if prediction == 0:
                            eye_status='0'
                        cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),color,2)
                    hist += eye_status

                    # for (ex,ey,ew,eh) in left_eye_pos:
                    #     color = (0,255,0)
                    #     prediction = predict_eye(face[ey:ey+eh,ex:ex+ew], model)
                    #     if prediction == 0:
                    #         eye_status='0'
                    #     cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),color,2)

                    # for (ex,ey,ew,eh) in right_eye_pos:
                    #     color = (0,255,0)
                    #     prediction = predict_eye(face[ey:ey+eh,ex:ex+ew], model)
                    #     if prediction == 0:
                    #         eye_status='0'
                    #     cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),color,2)
                    # hist += eye_status
                    
                    if isBlinking(hist, 2):
                        print(dt.datetime.now(),"Blinked")
                        blink_count += 1
                        print(hist)
                        hist = ""



                    # if len(left_eye_pos)>0 and len(right_eye_pos)>0:
                    #     for hface, ex,ey,ew,eh in zip( (left_face, right_face) ,(left_eye_pos[0],right_eye_pos[0])):
                    #         color = (0,255,0)
                    #         prediction = predict_eye(hface[ey:ey+eh,ex:ex+ew], model)
                    #         if prediction >= 0:
                    #             # if prediction == 1 : print(dt.datetime.now(), prediction)
                    #             hist += str(prediction)
                    #         cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),color,2)

                cv2.putText(frame, f"Blinks :  {blink_count}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
                cv2.imshow("eye blink detection",frame)
                if cv2.waitKey(1) == 13:
                    break
            else:
                break

    except Exception as e:
        print(f"ERROR : {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

