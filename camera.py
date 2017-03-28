
import cv2
import numpy as np

class CameraImage(object):

    def adjust_normal_hist(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # create a CLAHE object
        clip = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # equalize the histogram of the Y channel
        hsv[:, :, 2] = clip.apply(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def adjust_gamma(self, img, gamma=1.):
        # adjusts the brightness of an image
        # img : source image
        # gamma : brightness correction factor, gamma < 1 => darker image
        # returns : gamma corrected image

        # convert to HSV to adjust gamma by V
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        # http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
        invGamma = 1.0 / np.absolute(gamma)
        table = (np.array([((i / 255.0) ** invGamma) * 255
                           for i in np.arange(0, 256)]).astype("uint8"))

        # apply gamma correction using the lookup table
        img[:, :, 2] = cv2.LUT(img[:, :, 2], table)
        #img = cv2.LUT(img, table)

        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def auto_adjust_gamma(self, img, gamma_thresh=(0.5, 2), gamma_gain=1., roi_area=None):
        # Auto adjust an image gamma based off the mean and median in an image
        # img : supplied image
        # gamma_thresh : values for maximum auto gamma correction, (min, max)
        # gamma_gain : gain factor to make images brighter or darker
        # roi_area : a region of interest that can be used to get the correction
        # values supplied with two coordinates as (x1, y1, x2, y2)
        # returns : adjusted image

        if roi_area != None:
            mean = np.median(img[roi_area[3]:roi_area[1],
                           roi_area[0]:roi_area[2]])
            median = np.median(img) #[roi_area[3]:roi_area[1],
            #                   roi_area[0]:roi_area[2]])
            print(mean, "    median : ", median)
        else:
            width, height, _ = img.shape
            mean = np.mean(img[0:height, 0:width])
            median = np.median(img[0:height, 0:width])

        gamma = (mean / median) * gamma_gain
        if gamma > gamma_thresh[1]:
            gamma = gamma_thresh[1]  # clip gamma to maximum value
        elif gamma < gamma_thresh[0]:
            gamma = gamma_thresh[0]  # clip gamma to minimum value
        print(gamma)
        return self.adjust_gamma(img, gamma)

    def adjust_channel(self, channel, clip=1, ksize=3):
        # create a CLAHE object
        clip = cv2.createCLAHE(clipLimit=clip, tileGridSize=(ksize, ksize))
        # equalize the histogram of the Y channel
        return clip.apply(channel)

    def combine_binary(self, arg, *argv):
        # combines multiple binary vectors together to create one binary vector
        # arg : first binary vector
        # *argv : other binary vectors to be be combined
        # return : binary vector same shape as arg

        # Combine the two binary thresholds
        combined = arg
        for arg_vect in argv:
            combined[(combined == 1) | (arg_vect == 1)] = 1

        return combined

class CameraCalibration(object):
    def get_chessboard_corners(self, img, chess_count):
        # Finds the chessboard corners of the supplied size
        # img : input image
        # chess_count : grid size of the chessboard used for calibration
        # return : chessboard corners, ret, corners

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # get the checkerboard corners
        return cv2.findChessboardCorners(gray, (chess_count[0], chess_count[1]), None)

    def getCamera_calibration(self, img, chess_count=(3, 3)):
        # Finds all the corners in the chessboard pattern
        # img : input image
        # chess_count : grid size of the chessboard used for calibration
        # returns: ret, mtx, dist, rvecs, tvecs

        # get the checkerboard corners
        ret, corners = self.getChessboardCorners(img, chess_count=chess_count)

        if ret == True:
            # found the chessboard, continue with calibrations
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((chess_count[0] * chess_count[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:chess_count[0], 0:chess_count[1]].T.reshape(-1, 2)

            # Arrays to store object points and image points from all the images.
            objpoints = []  # 3d point in real world space
            imgpoints = []  # 2d points in image plane.
            objpoints.append(objp)

            # termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                               img.shape[::-1], None, None)
        return ret, mtx, dist, rvecs, tvecs

    def undistort_image(self, img, old_mtx, new_mtx, dist):
        # undistors an image with the calibrated camera properties
        # img : input image
        # old_mxt : pre calibration cameraMatrix
        # new_mxt : post calibration cameraMatrix
        # dist : input vector of distortion coefficients
        # return : an undistorted image of img

        return cv2.undistort(img, old_mtx, dist, None, new_mtx)

    def calibration_camera(self, img, chess_count=(3, 3)):
        # finds the calibration parameters for a camera
        # img : input image
        # chess_count : grid size of the chessboard used for calibration
        # return : camera matrix, distortion coefficients, rotation and translation vector

        ret, mtx, dist, rvecs, tvecs = self.getCameraCalibration(img, chess_count=chess_count)
        width, height, _ = img.shape
        # cv2.drawChessboardCorners(img, (check_dim[0], check_dim[1]), corners, ret)

        return cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))