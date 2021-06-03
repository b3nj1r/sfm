import numpy as np
import glob
import cv2
import sys


def calib(images):
    pass

def main():
    images = str(sys.argv[1])

    img = cv2.imread(images)
    cv2.imshow('img',img)
    cv2.waitKey(0)

if __name__ =="__main__":
    main()
    for i, arg in enumerate(sys.argv):
        print(arg)
