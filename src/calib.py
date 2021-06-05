import numpy as np
import pickle
import glob
import cv2
import sys


def calc(arr):

    """ camera matrix calculation

    arr, str

    """

    # initialize object points
    objp = np.zeros((6*9,3),np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # obj and img point arrays
    objpts = [] # 3d real world space
    imgpts = [] # 2d image space


    # read image and convert to grayscale
    img = cv2.imread(arr)
    gry = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # solve for corners
    ret, corners = cv2.findChessboardCorners(gry,(9,6),None)

    if ret:
        # update point arrays
        objpts.append(objp)
        imgpts.append(corners)

    # img width and height
    h,w = img.shape[:2]

    # camera matrix, distortion coefficients, radial and translational vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, gry.shape[::-1], None, None)
    newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist, (w,h), 1, (w,h))
    return ret, mtx, newmtx, dist, rvecs, tvecs


def rectify(arr, mtx, newmtx, dist):

    """ image rectification

    arr, str
    mtx, ndarray
    newmtx, ndarray
    dist, ndarray

    """

    img = cv2.imread(arr)
    h,w = img.shape[:2]

    xmap, ymap = cv2.initUndistortRectifyMap(mtx,dist,None,newmtx,(w,h),5)
    return cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)


def write(ret,mtx,newmtx,dist,rvecs,tvecs, file):

    """ write camera matrix data to file

    ret, float
    mtx, ndarray
    newmtx, ndarray
    dist, ndarray
    file, str

    """
    data = {}
    data['ret'] = ret
    data['mtx'] = mtx
    data['newmtx'] = mewmtx
    data['dist'] = dist
    pickle.dump(data,open(str(file),'rb'))


def read(file):

    """ read camera matrix data from file

    file, str

    """

    return pickle.load(str(file),'rb')


def main():
    ret, mtx, newmtx, dist, rvecs, tvecs = calc(str(sys.argv[1]))

    cv2.imwrite('../calib/result.jpg',rectify(str(sys.argv[2]), mtx, newmtx, dist))

if __name__ =="__main__":
   main()
