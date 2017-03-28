

import cv2
import numpy as np
import matplotlib.pyplot as plt

class Lane():
    fig = plt.figure()
    plt.ion()

    def __init__(self, focal_point=None, roi_height=None, source_pts=None):
        # initalises common variables in the class
        # focal_point : location of the focal point of the lane. Can be the
        #               vanishing point of the image
        # roi_height : height where the lane region of interest is at most
        #              considered
        # source_pts : bottom start points of the lane roi

        if focal_point is None:
            self.focal_point = [0,0]
        else:
            self.focal_point = focal_point

        if roi_height is None:
            self.roi_height = 0.
        else:
            self.roi_height = roi_height

        if source_pts is None:
            self.source_pts = [[0, 0], [0, 0]]
        else:
            self.source_pts = source_pts

        self.roi_pts = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]])
        self.left_fit = None
        self.right_fit = None

    def lane_roi(self, img_shape, roi_height=None, focal_point=None, source_pts=None):
        # defines a lanes region of interest
        # img_shape : shape of the input image
        # roi_height : the pixel height of the higest point of interest
        # focal_point : location of the focal focal_point. If None, will use the center of the image
        # source_pts : location of the two bottom corner points
        # return : coordinates of the region of interest of a lane

        if focal_point is None:
            focal_point = self.focal_point

        if roi_height is None:
            roi_height = self.roi_height

        h = img_shape[0] # image height

        # top of the roi is a factor of the height from the bottom of the roi
        # to the focal point.
        # ratio = (1 - fph/h) -> focal point position compared to the height
        # inv_fp = (1 - fph/h)*h -> inverse focal position
        # h_top = (ratio * (1 - roi_height)) * inv_fp
        # h_top is the y position of the height with respect to the focal
        fph = self.focal_point[1]  # height of focal point
        fp_ratio = (1 - fph / h)
        h_top = h * fp_ratio**2 * (1 - roi_height)

        if source_pts is None:
            # create the source points as the two bottom corners of the image
            source_pts = self.source_pts

        m_left = (focal_point[1] - source_pts[0][1]) / (focal_point[0] - source_pts[0][0])
        b_left = focal_point[1] - (m_left * focal_point[0])
        x_left = (h_top - b_left) // m_left

        m_right = (focal_point[1] - source_pts[1][1]) / (focal_point[0] - source_pts[1][0])
        b_right = focal_point[1] - (m_right * focal_point[0])
        x_right = (h_top - b_right) // m_right

        self.roi_pts = np.float32([source_pts[0], [x_left, h_top], [x_right, h_top], source_pts[1]])
        return self.roi_pts

    def draw_lane_roi(self, img, roi_pts=None, focal_point=None, color=(255, 255, 255)):
        # draws the region of interest onto the supplied image
        # img : source image
        # roi_pts : coordinate points of the region of interest
        # focal_point : location of the focal focal_point
        # return : the supplied image with the roi drawn on

        if focal_point is None:
            focal_point = self.focal_point
        if roi_pts is None:
            roi_pts = self.roi_pts

        image = img.copy()
        pts = np.int32(roi_pts)
        pts = pts.reshape((-1, 1, 2))
        cv2.circle(image, (focal_point[0], focal_point[1]), 5, color, 2)
        cv2.polylines(image, [pts], True, color, 2)

        return image

    def warp_image(self, img, roi_pts=None, location_pts=None, padding=(0,0)):
        # img : image to be transformed into the new perspective
        # roi_pts : location points from the original image to be transformed.
        #           Points must be in a clock wise order.
        # location_pts : the final location points in the image where the
        #           old_pts will be located. If None supplied, the new points
        #           will be the four corners off the supplied image in a
        #           clockwise order, starting at point (0,0).
        # offset : adds padding onto the roi points so the warped image is
        #          larger than the roi. Supplied as (width, height) padding
        # returns : the warped perspective image with the supplied points

        if roi_pts is None:
            roi_pts = self.roi_pts

        h, w = img.shape[:2]
        if location_pts is None:
            location_pts = np.float32([[padding[0], h-padding[1]], # bot-left
                                       [padding[0], padding[1]], # top-left
                                       [w-padding[0], padding[1]], # top-right
                                       [w-padding[0], h-padding[1]]]) # bot-right

        # calculate the perspective transform matrix between the old and new points
        M = cv2.getPerspectiveTransform(roi_pts, location_pts)
        # Warp the image to the new perspective
        return cv2.warpPerspective(img, M, (w, h))

    def mask_roi(self, img, roi_pts=None, outside_mask=True):
        # create a masked image showing only the area of the roi_pts
        # img : source image to be masked
        # roi_pts : region for masking
        # outside_mask : True if masking area outside roi, False if masking roi
        # return : masked image

        if roi_pts is None:
            roi_pts = self.roi_pts

        pts = np.int32(roi_pts)
        pts = [pts.reshape((-1, 1, 2))]

        # create a blank image to create a threshold
        """mask = np.ones_like(img)
        ignore_mask_color = (0, 0, 0)  # *channel_count
        # create a polygon that is white
        m = cv2.fillPoly(mask, pts, ignore_mask_color)"""

        mask = np.zeros_like(img)
        ignore_mask_color = (255, 255, 255)  # *channel_count
        # create a polygon that is white
        m = cv2.fillPoly(mask, pts, ignore_mask_color)

        # return the applyed mask
        if outside_mask == False:
            m = cv2.bitwise_not(m)
            return cv2.bitwise_and(img, m)
        else:
            return cv2.bitwise_and(img, mask)

    def combine_images(self, img_one, img_two, img_one_weight=0.8, img_two_weight=1.):
        # combines two images into one for display purposes
        # img_one : image one
        # img_two : image two
        # img_one_weight : transparency weight of image one
        # img_two_weight : transparency weight of image two
        # return : combined image
        return cv2.addWeighted(img_one, img_one_weight, img_two, img_two_weight, 0)

    def gauss(self, x, mu, sigma, A):
        # creates a gaussian distribution from the data
        # x : input data
        # mu : mean data point
        # sigma : variance from the mean
        # return : Gaussian distribution
        return A * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)

    def bimodal(self, x, mu1, sigma1, A1, mu2, sigma2, A2):
        return self.gauss(x, mu1, sigma1, A1) + self.gauss(x, mu2, sigma2, A2)

    def plot_histogram(self, data):
        # plot a real time histogram of the supplied data
        # data : data to plot
        plt.clf()
        plt.plot(data)
        plt.pause(0.00001)

    def histogram(self, data):
        # calculates the histogram of data
        # data : data to be transformed into a histogram
        # returns : a vector of the histogram data
        return np.sum(data, axis=0)

    def histogram_peaks(self, data, plot_hist=False):
        hist = self.histogram(data)

        if plot_hist == True:
            self.plot_histogram(hist)

        midpoint = np.int(hist.shape[0] // 2)
        leftx_base = np.argmax(hist[:midpoint])
        rightx_base = np.argmax(hist[midpoint:]) + midpoint

        return leftx_base, rightx_base

    def plot_best_fit(self, img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, margin=100):

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((img, img, img)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = self.combine_images(out_img, window_img, img_one_weight=1, img_two_weight=0.3)

        cv2.imshow('result', result) # visulise the output of the function

    def find_lane_lines(self, img, line_windows=10, plot_line=False, draw_square=False):

        out_img = img.copy()

        # Set height of windows
        window_height = np.int(img.shape[0] / line_windows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx, rightx = self.histogram_peaks(img)
        leftx_current = leftx
        rightx_current = rightx
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(line_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            if draw_square == True:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 255, 255), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 255, 255), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        if plot_line==True:
            # plot the line of best fit onto the image
            self.plot_best_fit(out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds)

        return self.left_fit, self.right_fit

    def lane_lines(self, img, plot_line=False):

        # Does the program know where the lane lines are?
        #image = self.left_fit
        if self.left_fit is None or self.right_fit is None:
            # Don't know where the lane lines are, so go and find them
            self.find_lane_lines(img)
        else:
            nonzero = img.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            margin = 100
            left_lane_inds = (
            (nonzerox > (self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] - margin)) & (
            nonzerox < (self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] + margin)))
            right_lane_inds = (
            (nonzerox > (self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] - margin)) & (
            nonzerox < (self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] + margin)))

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # Fit a second order polynomial to each
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)

            if plot_line == True:
                # plot the line of best fit onto the image
                self.plot_best_fit(img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds)

        return self.left_fit, self.right_fit

    def set_roi_points(self, roi_pts):
        # set the region of interest for the class
        # roi_pts : region of interest points
        self.roi_pts = roi_pts

    def get_roi_points(self):
        # gets the current the region of interest points for the class
        # return : roi_pts
        return self.roi_pts

    def set_focal_point(self, focal_point):
        # set the focal_point for the class
        # focal_point : the new focal point
        self.focal_point = focal_point

    def get_focal_point(self):
        # gets the current focal point for the class
        # return : focal_point
        return self.focal_point

    def set_roi_height(self, height):
        # set the roi_height for the class
        # height : the new roi_height
        self.roi_height = height

    def get_roi_height(self):
        # gets the current roi_height for the class
        # return : roi_height
        return self.roi_height