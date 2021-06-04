import numpy as np
import glob
import cv2
import sys


def calib(images):
    pass


def main():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # initialize object points
    objp = np.zeros((6*9,3),np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    print(objp)

    # obj and img point arrays
    objpts = [] # 3d real world space
    imgpts = [] # 2d image space

    images = glob.glob(str(sys.argv[1]))

    for f in images:
        # read image and convert to grayscale
        img = cv2.imread(f)
        gry = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # solve for corners
        ret, corners = cv2.findChessboardCorners(gry,(9,6),None)

        if ret:
            # update point arrays
            objpts.append(objp)
            imgpts.append(corners)

        # camera matrix, distortion coefficients, radial and translational vectors
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, gry.shape[::-1], None, None)

        # undistortion
        h,w = img.shape[:2]
        newmtx,roi = cv2.getOptimalNewCameraMatrix(mtx,dist, (w,h), 1, (w,h))

        xmap, ymap = cv2.initUndistortRectifyMap(mtx,dist,None,newmtx,(w,h),5)
        dst = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)

        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite('../calib/result.jpg',dst)

if __name__ =="__main__":
    main()
    for i, arg in enumerate(sys.argv):
        print(arg)
