import cv2
import numpy as np

img = cv2.imread('bookpage.jpg')
re,threshold1 = cv2.threshold(img,12,255, cv2.THRESH_BINARY)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
re2,threshold2 = cv2.threshold(gray,12,255, cv2.THRESH_BINARY)
threshold3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,155,1)
re4,threshold4 = cv2.threshold(gray,125,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("nor",threshold1)
cv2.imshow("gray1",threshold2)
cv2.imshow("gaussf",threshold3)
cv2.imshow("last",threshold4)

cv2.waitKey(0)
cv2.destroyAllWindows()