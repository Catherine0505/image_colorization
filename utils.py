import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.color as color
import os
import matplotlib.pyplot as plt

def auto_contrast_lab(image):
    """
    Automatically increases contrast on the given image. Performs luminance
    histogram equalization on LAB images.
    :param image: numpy 3D-array containing R, G and B values for each pixel.
    """
    # Transform RGB representation to LAB representation.
    image_lab = color.rgb2lab(image)

    # Compute the luminance histogram, and calculate its cumulative distribution
    # function.
    l_hist, _ = np.histogram(image_lab[:, :, 0], bins = 101, range = (0, 100))
    l_cumsum = np.cumsum(l_hist) / (image.shape[0] * image.shape[1])

    # Perform histogram equalization. 
    result = color.lab2rgb(
        np.dstack([l_cumsum[np.floor(image_lab[:, :, 0]).astype(np.int)] * 100,
        image_lab[:, :, 1], image_lab[:, :, 2]]))

    return result

def auto_contrast_rgb(image):
    """
    Automatically increases contrast on the given image. Performs RGB histogram
    equalization.
    :param image: numpy 3D-array containing R, G and B values for each pixel.
    """
    # Compute R, G and B histograms respectively. And find the cumulative
    # distribution for each channel.
    histogram_r, _ = np.histogram(image[:, :, 0], bins = 256, range = (0, 1))
    cum_hist_r = np.cumsum(histogram_r) / (image.shape[0] * image.shape[1])
    histogram_g, _ = np.histogram(image[:, :, 1], bins = 256, range = (0, 1))
    cum_hist_g = np.cumsum(histogram_g) / (image.shape[0] * image.shape[1])
    histogram_b, _ = np.histogram(image[:, :, 2], bins = 256, range = (0, 1))
    cum_hist_b = np.cumsum(histogram_b) / (image.shape[0] * image.shape[1])

    # Perform histogram equalization.
    return np.dstack([cum_hist_r[np.floor(image[:, :, 0] * 255).astype(np.int)],
        cum_hist_g[np.floor(image[:, :, 1] * 255).astype(np.int)],
        cum_hist_b[np.floor(image[:, :, 2] * 255).astype(np.int)]])

def auto_cropping(image):
    """
    Automatically crops out uni-color or bi-color boundaries.
    :param image: numpy 3D-array containing R, G and B values for each pixel.
    """
    # Find the pixels that have at least one very dark layer. Such pixels are
    # considered candidates in the boundary.
    mask = image < 0.15
    border = np.sum(mask, axis = 2) >= 1

    # Scan through each row and column. If the proportion of candidate boundary
    # pixels are more than 0.8, regard this row/ column as boundary and crop it
    # out.
    vertical_thresh = np.floor(image.shape[1] * 0.8).astype(np.int)
    crop_vertical = image[np.sum(border, axis = 1) < vertical_thresh]
    horizontal_thresh = np.floor(image.shape[0] * 0.8).astype(np.int)
    result = crop_vertical[:, np.sum(border, axis = 0) < horizontal_thresh]
    return result
