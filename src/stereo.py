import matplotlib.pyplot as plt
import numpy as np
import cv2

limg = cv2.imread('../stereo/left.png')
rimg = cv2.imread('../stereo/right.png')


limg = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)


# disparity settings
block_size = 5
min_disp = 32
num_disp = 112-min_disp

stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = block_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 12,
    P1 = 8*3*block_size**2,
    P2 = 32*3*block_size**2,
)

disparity = stereo.compute(limg,rimg)
disp = plt.figure(1)
plt.imshow(disparity)
left = plt.figure(2)
plt.imshow(limg)
right = plt.figure(3)
plt.imshow(rimg)
plt.show()

