import cv2 
import numpy as np
import time;
import npwriter
name = input("Enter your name: ")

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
f_list = []
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    cv2.imshow("full", frame) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5)
    faces = sorted(faces, key = lambda x: x[2]*x[3], 
         reverse = True)
    faces = faces[:1]
    if len(faces) == 1:  
     face = faces[0]
     x, y, w, h = face
     #show only the face captured ( double chin and all)
     im_face = frame[y:y + h, x:x + w]
     cv2.imshow("face", im_face)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
       break

    elif key & 0xFF == ord('c'):
        print("hello")
       
        gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY) 
        gray_face = cv2.resize(gray_face, (100, 100)) 
        print(len(f_list), type(gray_face), gray_face.shape)
        f_list.append(gray_face.reshape(-1)) 
        print(len(f_list), type(gray_face), gray_face.shape)
        print("saving file ")   
        npwriter.write(name, np.array(f_list))
        print("file saved")
        break
            
        # else: 
        #     print("face not found")
        # if len(f_list) == 10: 
        #     break
npwriter.write(name, np.array(f_list))