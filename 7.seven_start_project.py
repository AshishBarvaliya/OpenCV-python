import cv2 
import numpy as np

cap = cv2.VideoCapture(0)
haar_face =  cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
haar_eye =  cv2.CascadeClassifier('haarcascade_eye.xml')
#haar_smile =  cv2.CascadeClassifier('haarcascade_smile.xml')

while True:
    boo,frame =cap.read()
    GRAY = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face= haar_face.detectMultiScale(GRAY, 1.3, 5)

    for(x,y,w,h) in face:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
        roi_gray= GRAY[y:y+h, x:x+w]
        roi_frame= frame[y:y+h, x:x+w]
        eye= haar_eye.detectMultiScale(roi_frame)
        #smile= haar_smile.detectMultiScale(roi_frame)
        for (ex,ey,ew,eh) in eye:
            cv2.rectangle(roi_frame, (ex,ey), (ex+ew,ey+eh),(255,0,0),2)
        #for (sx,sy,sw,sh) in smile:
         #   cv2.rectangle(roi_frame, (sx,sy), (sx+sw,sy+sh),(0,0,255),2)
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()        
