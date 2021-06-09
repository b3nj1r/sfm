import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

def identify(img):

    """ keypoint identification using opencv's ORB detector and descriptor extractor
    img,    str
    """
    img = cv2.imread(img)

    # create orb
    orb = cv2.ORB_create()

    # solve keypoints
    kp = orb.detect(img,None)

    # return computed descriptors and image
    kp, des = orb.compute(img,kp)
    return img, kp, des

def match(kp1,des1,kp2,des2,epsilon):

    """ detect matching keypoints
    kp1,    list
    des1,   ndarray
    kp2,    list
    des2,   ndarray
    """

    # initialize FLANN matcher
    FLANN_INDEX_KDTREE = 1
    indx_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    srch_params = dict(checks   =50)
    flann = cv2.FlannBasedMatcher(indx_params,srch_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # solve for usable matches
    gd=[]
    for m,n in matches:
        if m.distance < epsilon*n.distance:
            gd.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in gd] ).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in gd] ).reshape(-1,1,2)

    return pts1, pts2


def main():

    # identify keypoints on given image
    img1, kp1, des1 = identify(sys.argv[1])
    img2, kp2, des2 = identify(sys.argv[2])

    # display keypoints
    plt.imshow(draw)
    plt.show()

if __name__ == '__main__':
    main()
