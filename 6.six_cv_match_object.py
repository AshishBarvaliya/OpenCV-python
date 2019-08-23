import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("ghano.JPG")
mask = np.zeros(img.shape[:2], np.uint8)

bgmodel = np.zeros((1,65),np.float64)
fgmodel = np.zeros((1,65),np.float64)

rect =(50,50,300,500)

cv2.grabCut(img, mask, rect, bgmodel, fgmodel, 5, cv2.GC_INIT_WITH_RECT)
mask2= np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[::np.newaxis]
plt.imshow(img)
plt.colorbar()
plt.show()