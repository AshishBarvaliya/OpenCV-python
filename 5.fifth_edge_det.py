import cv2 
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    boo,frame =cap.read()
    laplacian= cv2.Laplacian(frame,cv2.CV_64F)
    
    cv2.imshow("frame",frame)
    cv2.imshow("lap",laplacian)
    
    k = cv2.waitKey(5) & 0xff
    if k ==27:
        break
    
cv2.destroyAllWindows()
cv2.release()    