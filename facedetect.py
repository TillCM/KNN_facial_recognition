#https://medium.com/analytics-vidhya/face-detection-and-recognition-using-opencv-and-knn-from-scratch-dcba9b0fd07d
from cv2 import cv2 
import numpy as np
import time;
import npwriter


#gather user name:
name = input("Enter your name: ")
#time.sleep(30)
cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
f_list = []

while True: 
    ret, frame = cap.read() 
    #show webcam capture (frame)
    cv2.imshow('frame', frame)
    #convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #use the haarscascade classifier to convert the face to coordinates
    faces = classifier.detectMultiScale(gray, 1.5, 5) 
    #select face closes to webcam:
    faces = sorted(faces, key = lambda x: x[2]*x[3], 
         reverse = True)
    # use only the detected face:
    faces = faces[:1]
    if len(faces) == 1:  
     face = faces[0] 
     #storing the coordinates of the face in a hyper-plane
     x, y, w, h = face
     #show only the face captured ( double chin and all)
     im_face = frame[y:y + h, x:x + w]
     cv2.imshow("face", im_face)

   
    if not ret:
        continue 
    cv2.imshow("full", frame) 

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
       break

    elif key & 0xFF == ord('c'):
        if len(faces) == 1: 
            gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY) 
            gray_face = cv2.resize(gray_face, (100, 100)) 
            f_list.append(gray_face.reshape(-1)) 
            print(len(f_list), type(gray_face), gray_face.shape)
            
        else: 
            print("face not found")
            if len(f_list) == 10: 
                break

npwriter.write(name, np.array(f_list))
   
cap.release()
cv2.destroyAllWindows()
    