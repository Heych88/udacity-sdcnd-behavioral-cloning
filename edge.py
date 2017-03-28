import numpy as np
import cv2

def sobel(img_channel, orient='x', sobel_kernel=3):

    if orient == 'x':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, sobel_kernel)

    return sobel

def binary_array(array, thresh, value=1):
    # turns an array into a binary array when between a threshold
    # array : numpy array to be converted to binary
    # thresh : threshold values between which a change in binary is stored.
    #          Threshold is inclusive
    # value : output value when between the supplied threshold
    # return : Binary array version of the supplied array

    # Is activation (1) between the threshold values (band-pass) or is it
    # outside the threshold values (band-stop)
    if value == 0:
        # Create a binary array the same size of as the input array
        # band-stop binary array
        binary = np.ones_like(array)
    else:
        # band-pass binary array
        binary = np.zeros_like(array)
        value = 1

    binary[(array >= thresh[0]) & (array <= thresh[1])] = value
    return binary

def rescale_to_8bit(array, max_value=255):
    # rescales input array to an uint8 array
    # array : array to be rescaled
    # max_value : maximum value in the returned array
    # return : 8-bit array
    return np.uint8(max_value * array / np.max(array))

def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):

    # get the edges in the horizontal direction
    sobelx = np.absolute(sobel(image, orient='x', sobel_kernel=sobel_kernel))
    # get the edges in the vertical direction
    sobely = np.absolute(sobel(image, orient='y', sobel_kernel=sobel_kernel))

    # Calculate the edge magnitudes
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    return binary_array(mag, thresh)

def blur_gaussian(channel, ksize=3):
    # blurs the channel data using gaussian
    # channel : 2D array of data to be blured
    # ksize : size of the kernel
    # return : blured data
    return cv2.GaussianBlur(channel, (ksize, ksize), 0)
