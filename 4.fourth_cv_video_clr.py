import cv2 
import numpy as np

handc = cv2.CascadeClassifier('myhaar.xml')

cap = cv2.VideoCapture(0)

while True:
    boo,frame =cap.read()
    frame=cv2.flip(frame,1)

    hvs = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # hvs hue setulation  value
    #lower_red = np.array([150,150,50])
    #higher_red = np.array([180,255,150])
    lower_red = np.array([0,30,60])
    higher_red = np.array([20,150,255])
    
    mask= cv2.inRange(hvs, lower_red, higher_red)
    res=cv2.bitwise_and(frame, frame, mask = mask)
    
    kernal =np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernal)
    closing = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernal)
    
    hands = handc.detectMultiScale(closing)
    for (x,y,w,h) in hands:
        cv2.rectangle(closing, (x,y), (x+w,y+h), (255,0,0),2)   
    
    cv2.imshow("frame",frame)
    cv2.imshow("res",res)
    cv2.imshow("opening",opening)
    cv2.imshow("closing",closing)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cv2.destroyAllWindows()
cv2.release()    