import numpy as np
import skimage as sk
import skimage.io as skio
import os
import single_scale as ss
from skimage.transform import resize
import utils as utils

def multi_scale(b, g, r, num_steps):
    """
    Finds the best displacement given a displacement window, based on cropped r,
    g and b layers.
    :param b: numpy 2D-array containing values on the b layer.
    :param g: numpy 2D-array containing values on the g layer.
    :param r: numpy 2D-array containing values on the r layer.
    :num_steps: the number of steps of the image pyramid (i.e., number of layers
        from coarsest to finest)
    """
    r_displacements = []
    g_displacements = []

    # Find the size of coarsest image, and resize R, G, B channels to this size.
    coarse_height = np.floor(b.shape[0] / (2 ** num_steps)).astype(np.int)
    coarse_width = np.floor(b.shape[1] / (2 ** num_steps)).astype(np.int)
    b_crop = resize(b, (coarse_height, coarse_width), anti_aliasing = True)
    r_crop = resize(r, (coarse_height, coarse_width), anti_aliasing = True)
    g_crop = resize(g, (coarse_height, coarse_width), anti_aliasing = True)

    g_shift = (0, 0)
    r_shift = (0, 0)

    for i in range(num_steps):
        height = b_crop.shape[0]
        width = b_crop.shape[1]
        # During each iteration, the size of R, G and B layers are doubled to
        # go from coarsest to finest.
        b_crop = resize(b_crop, (height * 2, width * 2), anti_aliasing = True)
        r_crop = resize(r_crop, (height * 2, width * 2), anti_aliasing = True)
        g_crop = resize(g_crop, (height * 2, width * 2), anti_aliasing = True)

        # Pick the middle region of the current image to avoid side effects of
        # boundaries. Calculate best displacement level on the current
        # resolution.
        height = b_crop.shape[0]
        width = b_crop.shape[1]
        padding_width = np.floor(width / 5.0).astype(np.int)
        padding_height = np.floor(height / 5.0).astype(np.int)
        displacement_window = (range(-20, 20), range(-20, 20))
        g_shift, r_shift = ss.single_scale_multi(displacement_window, g_crop,
            r_crop, b_crop, padding_height, padding_width)

        # Save displacement level of the current resolution for future
        # calculation.
        r_displacements.append(r_shift)
        g_displacements.append(g_shift)

        # Align images based on calculated displacement, and crop out unaligned
        # boundaries.
        g_crop = np.roll(g_crop, shift = g_shift, axis = (0, 1))
        r_crop = np.roll(r_crop, shift = r_shift, axis = (0, 1))


        top = max([g_shift[0], r_shift[0], 0])
        bottom = min([g_shift[0], r_shift[0], 0])
        left = max([g_shift[1], r_shift[1], 0])
        right = min([g_shift[1], r_shift[1], 0])

        b_crop = b_crop[top: height + bottom, left: width + right]
        g_crop = g_crop[top: height + bottom, left: width + right]
        r_crop = r_crop[top: height + bottom, left: width + right]

    # Calculate the final displacement based on displacements on each resolution
    # level.
    r_shift_scaled_height = [r_displacements[i][0] * (2 ** (num_steps - i - 1)) for i in range(num_steps)]
    r_shift_scaled_width = [r_displacements[i][1] * (2 ** (num_steps - i - 1)) for i in range(num_steps)]
    g_shift_scaled_height = [g_displacements[i][0] * (2 ** (num_steps - i - 1)) for i in range(num_steps)]
    g_shift_scaled_width = [g_displacements[i][1] * (2 ** (num_steps - i - 1)) for i in range(num_steps)]

    r_shift_final = (sum(r_shift_scaled_height), sum(r_shift_scaled_width))
    g_shift_final = (sum(g_shift_scaled_height), sum(g_shift_scaled_width))
    print(r_shift_final, g_shift_final)

    return(r_shift_final, g_shift_final)

def generate(imname):
    """
    Generates the synthesized image.
    :param imname: a list of strings containing the name of images to be
        synthesized.
    """
    for name in imname:
        path = 'data/' + name + ".tiff"
        im = skio.imread(path)
        im = sk.img_as_float(im)

        # Find the height and width of each layer.
        height = np.floor(im.shape[0] / 3.0).astype(np.int)
        width = im.shape[1]

        # Extract r, g and b layer.
        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        print(name)
        r_shift, g_shift = multi_scale(b, g, r, num_steps = 4)

        # After displacing the layers, r, g and b layers may not be perfectly
        # aligned. Crop out the area with only one or two layers as much as
        # possible.
        top = max([g_shift[0], r_shift[0], 0])
        bottom = min([g_shift[0], r_shift[0], 0])
        left = max([g_shift[1], r_shift[1], 0])
        right = min([g_shift[1], r_shift[1], 0])

        g_roll = np.roll(g, shift = g_shift, axis = (0, 1))
        r_roll = np.roll(r, shift = r_shift, axis = (0, 1))

        b_crop_final = b[top: height + bottom, left: width + right]
        g_crop_final = g_roll[top: height + bottom, left: width + right]
        r_crop_final = r_roll[top: height + bottom, left: width + right]

        # Stack and show aligned images.
        im_out = np.dstack([r_crop_final, g_crop_final, b_crop_final])
        skio.imshow(im_out)
        skio.show()
        skio.imsave('normal_results/' + name + ".jpeg", im_out, quality = 100)

        # Automatic contrast with LAB images. Save generated results to folder.
        im_out_contrast_lab = utils.auto_contrast_lab(im_out)
        skio.imshow(im_out_contrast_lab)
        skio.show()
        skio.imsave('lab_contrast_results/' + name + ".jpeg",
            im_out_contrast_lab, quality = 100)

        # Automatic contrast with RGB images. Save generated results to folder.
        im_out_contrast_rgb = utils.auto_contrast_rgb(im_out)
        skio.imshow(im_out_contrast_rgb)
        skio.show()
        skio.imsave('rgb_contrast_results/' + name + ".jpeg",
            im_out_contrast_rgb, quality = 100)

        # Automatic cropping with LAB-contrasted images.
        # Save generated results to folder.
        im_crop = utils.auto_cropping(im_out_contrast_lab)
        skio.imshow(im_crop)
        skio.show()
        skio.imsave('crop_results/' + name + ".jpeg", im_crop, quality = 100)

def main():
    """
    Runs image pyramid method on TIFF high-resolution images.
    """
    if not os.path.exists('normal_results'):
        os.mkdir('normal_results')
    if not os.path.exists('lab_contrast_results'):
        os.mkdir('lab_contrast_results')
    if not os.path.exists('rgb_contrast_results'):
        os.mkdir('rgb_contrast_results')
    if not os.path.exists('crop_results'):
        os.mkdir('crop_results')

    imname = ['church', 'emir', 'harvesters', 'icon', 'lady',
        'melons', 'onion_church', 'self_portrait',
        'three_generations', 'train', 'workshop']
    generate(imname)

def extra():
    """
    Runs image pyramid method on extra TIFF high-resolution images.
    """
    if not os.path.exists('normal_results'):
        os.mkdir('normal_results')
    if not os.path.exists('lab_contrast_results'):
        os.mkdir('lab_contrast_results')
    if not os.path.exists('rgb_contrast_results'):
        os.mkdir('rgb_contrast_results')
    if not os.path.exists('crop_results'):
        os.mkdir('crop_results')
    imname = ['three_brothers', 'sailors']
    generate(imname)
