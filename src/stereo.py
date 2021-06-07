import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write(file,pts,clrs):
    pts = pts.reshape(-1,3)
    clrs = clrs.reshape(-1,3)
    pts = np.hstack([pts,clrs])

    # write file
    with open(file,'wb') as f:
        f.write((ply_header % dict(vert_num=len(pts))).encode('utf-8'))
        np.savetxt(f,pts,fmt='%f %f %f %d %d %d')

def main():
    limg = cv2.imread(str(sys.argv[1]),1)
    rimg = cv2.imread(str(sys.argv[2]),1)


    #limg = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
    #rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)

    # disparity settings
    block_size = 10
    min_disp = 16
    num_disp = 256-min_disp

    stereo = cv2.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = block_size,
        uniquenessRatio = 15,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 12,
        P1 = 8*3*block_size**2,
        P2 = 32*3*block_size**2,
    )

    disparity = stereo.compute(limg,rimg).astype(np.float32) / 16.0
    disparity = (disparity-min_disp)/num_disp

    h,w = limg.shape[:2]
    f = 0.8*w

    #TODO: REMOVE PLACEHOLDER Q MATRIX WITH STEREO CALIBRATED Q MATRIX

    Q = np.float32([[1, 0, 0, -0.5*h],
               [0, -1, 0, -0.5*w],
               [0, 0, f * 0.5, 0],
               [0, 0, 0, 1]])

    pts = cv2.reprojectImageTo3D(disparity, Q)
    clrs = cv2.cvtColor(limg,cv2.COLOR_BGR2RGB)
    mask = disparity > disparity.min()
    outpts = pts[mask]
    outclrs = clrs[mask]
    file = str(sys.argv[3])

    write(file,outpts,outclrs)
    print('%s saved' % file)
    disp = plt.figure(1)
    plt.imshow(disparity)
    plt.show()

if __name__=='__main__':
    main()

