"""
CS 194-26 Project 1 Part 1.
Author: Catherine Gai
"""

import numpy as np
import skimage as sk
import skimage.io as skio
import os
import utils as utils
import cv2

def single_scale_multi(displacement_window, g, r, b, padding_height,
    padding_width):
    """
    Finds the best displacement given a displacement window, based on cropped r,
    g and b layers. Used during image pyramid.
    :param displacement_window: a tuple of two iterables specifying the range
        of pixels to move around g and r layer.
    :param g: the g layer of image.
    :param r: the r layer of image.
    :param b: the b layer of image.
    :param padding_height: the size of top and bottom padding when cropping the
        image
    :param padding_width: the size of left and right padding when cropping the
        image
    """
    dict_bg = {}
    dict_br = {}
    b_slice = b[padding_height:-padding_height, padding_width:-padding_width]

    # Iterates over the specified candidate displacements both horizontally and
    # vertically to find the best displacement value.
    # For each displacement, take only a crop of the r, g and b layer.
    for i in displacement_window[0]:
        for j in displacement_window[1]:
            g_slice = g[padding_height+i: -padding_height+i,
                padding_width+j: -padding_width+j]
            r_slice = r[padding_height+i: -padding_height+i,
                padding_width+j: -padding_width+j]
            l2_bg = ncc(b_slice, g_slice)
            l2_br = ncc(b_slice, r_slice)
            dict_bg[l2_bg.item()] = (i, j)
            dict_br[l2_br.item()] = (i, j)

    # Finds the displacement value both horizontally and vertically with minimum
    # l2 difference.
    g_shift = dict_bg.get(max(dict_bg.keys()))
    r_shift = dict_br.get(max(dict_br.keys()))
    return ((-g_shift[0], -g_shift[1]), (-r_shift[0], -r_shift[1]))

def single_scale(displacement_window, g, r, b, padding_height,
    padding_width):
    """
    Finds the best displacement given a displacement window, based on cropped r,
    g and b layers. Used during naive exhaustive search.
    :param displacement_window: a tuple of two iterables specifying the range
        of pixels to move around g and r layer.
    :param g: the g layer of image.
    :param r: the r layer of image.
    :param b: the b layer of image.
    :param padding_height: the size of top and bottom padding when cropping the
        image
    :param padding_width: the size of left and right padding when cropping the
        image
    """
    dict_bg = {}
    dict_br = {}
    b_slice = b[padding_height:-padding_height, padding_width:-padding_width]

    # Iterates over the specified candidate displacements both horizontally and
    # vertically to find the best displacement value.
    # For each displacement, take only a crop of the r, g and b layer.
    for i in displacement_window[0]:
        for j in displacement_window[1]:
            g_slice = g[padding_height+i: -padding_height+i,
                padding_width+j: -padding_width+j]
            r_slice = r[padding_height+i: -padding_height+i,
                padding_width+j: -padding_width+j]
            l2_bg = l2_norm(b_slice, g_slice)
            l2_br = l2_norm(b_slice, r_slice)
            dict_bg[l2_bg.item()] = (i, j)
            dict_br[l2_br.item()] = (i, j)

    # Finds the displacement value both horizontally and vertically with minimum
    # l2 difference.
    g_shift = dict_bg.get(min(dict_bg.keys()))
    r_shift = dict_br.get(min(dict_br.keys()))
    return ((-g_shift[0], -g_shift[1]), (-r_shift[0], -r_shift[1]))

def l2_norm(image1, image2):
    """
    Finds the l2 norm difference between image 1 and image 2.
    :param image1: input image one.
    :param image2: input image two.
    """
    return np.linalg.norm((image1 - image2).reshape(-1, 1), axis = 0)

def ncc(image1, image2):
    """
    Finds the normalized cross correlation value between image 1 and image 2.
    :param image1: input image one.
    :param image2: input image two.
    """
    image1_flatten = image1.flatten()
    image2_flatten = image2.flatten()
    image1_mean = np.mean(image1_flatten)
    image2_mean = np.mean(image2_flatten)
    image1_normalize = (image1_flatten - image1_mean) / \
        np.linalg.norm(image1_flatten)
    image2_normalize = (image2_flatten - image2_mean) / \
        np.linalg.norm(image2_flatten)
    return image1_normalize @ image2_normalize

def generate(imname):
    """
    Generates the synthesized image.
    :param imname: a list of strings containing the name of images to be
        synthesized.
    """
    for name in imname:
        im = skio.imread('data/' + name)
        im = sk.img_as_float(im)

        # Find the height and width of each layer.
        height = np.floor(im.shape[0] / 3.0).astype(np.int)
        width = im.shape[1]
        window_height = 15
        window_len = 15

        # Extract r, g and b layer.
        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        # Set the size of cropped layers.
        padding_width = np.floor(width / 5.0).astype(np.int)
        padding_height = np.floor(height / 5.0).astype(np.int)

        g_shift, r_shift = single_scale((range(-window_height, window_height),
            range(-window_len, window_len)), g, r, b, padding_height,
            padding_width)
        print(name, g_shift, r_shift)
        g_roll = np.roll(g, shift = g_shift, axis = (0, 1))
        r_roll = np.roll(r, shift = r_shift, axis = (0, 1))

        # After displacing the layers, r, g and b layers may not be perfectly
        # aligned. Crop out the area with only one or two layers as much as
        # possible.
        top = max([g_shift[0], r_shift[0], 0])
        bottom = min([g_shift[0], r_shift[0], 0])
        left = max([g_shift[1], r_shift[1], 0])
        right = min([g_shift[1], r_shift[1], 0])

        b_crop = b[top: height + bottom, left: width + right]
        g_crop = g_roll[top: height + bottom, left: width + right]
        r_crop = r_roll[top: height + bottom, left: width + right]

        # Stack and show aligned images. 
        im_out = np.dstack([r_crop, g_crop, b_crop])
        im_out_ubyte = sk.img_as_ubyte(im_out)
        skio.imshow(im_out_ubyte)
        skio.show()
        skio.imsave('normal_results_part1/' + name, im_out_ubyte)

        # Automatic contrast with LAB images. Save generated results to folder.
        im_out_contrast_lab = utils.auto_contrast_lab(im_out)
        im_out_contrast_lab = sk.img_as_ubyte(im_out_contrast_lab)
        skio.imshow(im_out_contrast_lab)
        skio.show()
        skio.imsave('lab_contrast_results/' + name, im_out_contrast_lab)

        # Automatic contrast with RGB images. Save generated results to folder.
        im_out_contrast_rgb = utils.auto_contrast_rgb(im_out)
        im_out_contrast_rgb = sk.img_as_ubyte(im_out_contrast_rgb)
        skio.imshow(im_out_contrast_rgb)
        skio.show()
        skio.imsave('rgb_contrast_results/' + name, im_out_contrast_rgb)

        # Automatic cropping with LAB-contrasted images.
        # Save generated results to folder.
        im_crop = utils.auto_cropping(im_out_contrast_lab)
        skio.imshow(im_crop)
        skio.show()
        skio.imsave('crop_results/' + name, im_crop)

def main():
    """
    Runs exhaustive search on JPEG low-resolution images.
    """
    if not os.path.exists('normal_results_part1'):
        os.mkdir('normal_results_part1')
    if not os.path.exists('lab_contrast_results'):
        os.mkdir('lab_contrast_results')
    if not os.path.exists('rgb_contrast_results'):
        os.mkdir('rgb_contrast_results')
    if not os.path.exists('crop_results'):
        os.mkdir('crop_results')

    imname = ['cathedral.jpeg', 'monastery.jpeg', 'tobolsk.jpeg']
    generate(imname)
