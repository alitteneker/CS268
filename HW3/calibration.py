import cv2, numpy as np

def calibrate(filenames):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = []
    for filename in filenames:
        # Find the chess board corners. If found, add object points, image points (after refining them)
        img = cv2.imread(filename)
        if img != None:
            print "Loaded " + repr(filename)
        else:
            print "Unable to load image " + repr(filename)
            continue
        images.append(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print "Loaded all images and calbulated calibration"
    for i, img in enumerate(images):
        img = images[i]
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        cv2.imwrite( 'calibrated/out_' + str(i) + '.png', dst[ y : y+h, x : x+w ])
        print "Outputted calibrated image: 'calibrated/out_" + str(i) + ".png'"

filenames = []
for i in xrange(1, 15):
    filenames.append( 'chessboard/left' + str(i).zfill(2) + '.jpg' )
calibrate(filenames)
