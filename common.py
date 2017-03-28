import edge
from driveline import Lane
import cv2
import numpy as np

def image_preprocess(img):
    height, width = img.shape[:2]
    # crop the skyline from the camera image
    image = img[60:height, 0:width]
    h, w = image.shape[:2]

    # create a focal point on the image for masking over the vehicles bonnet
    focal_point = [w // 2, 0]
    lane = Lane(focal_point=focal_point, roi_height=25/100, source_pts=[[20, h], [w-20, h]])
    src = lane.lane_roi((h, w))

    # Convert to HSV color space and separate the V channel
    # hls for Sobel edge detection
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    sxbinary = hls[:, :, 1] # use on the luminance channel data for edges

    # get the median of the road area of the channel for the adaptive threshold
    area = sxbinary[h//3:h*3//4, w//4:w*3//4]
    median = np.median(area)
    # blur the image to help find edges
    sxbinary = edge.blur_gaussian(sxbinary, ksize=3)
    low_thresh = median * 1.25
    if low_thresh > 150:
        low_thresh = 150

    # find the edges in the channel data using sobel magnitudes
    sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(low_thresh, 255))
    # remove the car edges from the image
    sxbinary = lane.mask_roi(sxbinary, src, outside_mask=False)
    sxbinary = edge.rescale_to_8bit(sxbinary)

    s_channel = hls[:, :, 2] # use only the saturation channel data
    # get the median of the road area of the channel for the adaptive threshold
    area = s_channel[h//3:h*3//4, w//4:w*3//4]
    median = np.median(area)
    s_binary = edge.binary_array(s_channel, (median*.55, median*1.75), value=1)
    s_binary = lane.mask_roi(s_binary, src, outside_mask=False) # remove the car area
    s_binary = edge.rescale_to_8bit(s_binary)

    # mask the red channel data of the image
    masked_img = lane.mask_roi(image[:,:,2], src, outside_mask=False)

    # Stack each channel into one array
    return np.dstack((masked_img, sxbinary, s_binary)).astype(np.uint8)
