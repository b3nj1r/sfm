import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

def identify(img):

    """ keypoint identification using opencv's ORB detector and descriptor extractor
    img, str
    """
    img = cv2.imread(img)

    # create orb
    orb = cv2.ORB_create()

    # solve keypoints
    kp = orb.detect(img,None)

    # return computed descriptors and image
    kp, des = orb.compute(img,kp)
    return img, kp, des

def match():

    """ detect matching keypoints
    """

    pass

def main():

    # identify keypoints on given image
    img, kp, des = identify(sys.argv[1])

    # display keypoints
    draw = cv2.drawKeypoints(img,kp,None,color=(1,255,1))
    plt.imshow(draw)
    plt.show()

if __name__ == '__main__':
    main()
