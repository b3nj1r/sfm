import matplotlib.pyplot as plt
import numpy as np
import cv2

limg = cv2.imread('../stereo/left.png')
rimg = cv2.imread('../stereo/right.png')


limg = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)

disparity = stereo.compute(limg,rimg)
disp = plt.figure(1)
plt.imshow(disparity)
left = plt.figure(2)
plt.imshow(limg)
right = plt.figure(3)
plt.imshow(rimg)
plt.show()

