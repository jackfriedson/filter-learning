import numpy as np
import os
import shutil
import tensorflow as tf

from functools import partial
from glob import glob
from numpy import random

from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.exposure import adjust_gamma, adjust_sigmoid
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.util import img_as_float, img_as_ubyte
from sklearn.utils import shuffle

IMG_FOLDER = '../../images/'
TEST_IMG_PATH = '../../transformed_imgs/'
MIN_HEIGHT = 100
MIN_WIDTH = 100

def create_filter_imgs(transform_fns, img_height=MIN_HEIGHT, img_width=MIN_WIDTH, max_examples=200):
    imgpaths = glob(IMG_FOLDER + '*.jpg')
    originals = [read_or_delete_img(path) for path in imgpaths[:max_examples]]
    values = []

    imgs = originals
    for i, fn in enumerate(transform_fns):
        transforms, new_values = fn(imgs)
        imgs = transforms
        values.append(new_values)

    imsize = partial(resize_img, height=img_height, width=img_width)
    imgpairs = zip(imsize(transforms), imsize(originals))

    n_to_save = 10
    save_images(originals[:n_to_save], transforms[:n_to_save], 'filter')

    return imgpairs, values

def create_img_data(transform_fn, preprocess_fn=None, img_height=MIN_HEIGHT,
                    img_width=MIN_WIDTH, data_batch=100, max_examples=None, **config):
    """Create data for the algorithm to learn from.

    Reads images from the image folder, then transforms them using the provided function.
    This transformations is what the algorithm will learn to mimic. Both original
    and transformed images are then resized and preprocessed as specified in the config.
    The pixelwise difference between each original and transformed image is computed,
    and these differences with their corresponding transformation value are returned.

    Arguments:
        transform_fn {function} -- A function that takes a list of images as input and transforms
                                   them in some way
        **config {dictionary} -- Configuration settings

    Returns:
        (pixelwise_difference, transform_value) for each image
    """
    imgpaths = glob(IMG_FOLDER + '*.jpg')
    imgpaths = imgpaths[:max_examples]

    n_imgs = len(imgpaths)
    n_batches = int(np.ceil(n_imgs / float(data_batch))) if data_batch is not None else 1

    originals, transforms, values, diffs = [], [], [], []

    # TODO: Abstract batch functionality
    for i in range(n_batches):
        batch_start = i * data_batch
        batch_end = (i + 1) * data_batch
        batch_imgs = [read_or_delete_img(path) for path in imgpaths[batch_start:batch_end]]
        originals.extend(batch_imgs)

        batch_transform_imgs, batch_values = transform_fn(batch_imgs)
        transforms.extend(batch_transform_imgs)

        imsize = partial(resize_img, height=img_height, width=img_width)
        imgpairs = zip(imsize(batch_transform_imgs), imsize(batch_imgs))

        if preprocess_fn is not None:
            imgpairs = [(preprocess_fn(t), preprocess_fn(o)) for t, o in imgpairs]

        batch_values = np.array(batch_values, np.float32)
        batch_diffs = [(t - o).flatten() for t, o in imgpairs]

        values.extend(batch_values)
        diffs.extend(batch_diffs)

        print 'Created {}/{} examples'.format(len(diffs), n_imgs)

    # Save a handful of images
    n_to_save = 10
    save_images(originals[:n_to_save], transforms[:n_to_save], config.get('name'))

    return diffs, values

def read_or_delete_img(path):
    try:
        img = img_as_ubyte(imread(path))
        raise_if_invalid(img)
        return img
    except (ValueError, IOError):
        print 'Deleting file: {}'.format(path)
        os.remove(path)

def raise_if_invalid(img):
    if not (img.ndim < 4 \
        and img.shape[0] > MIN_HEIGHT \
        and img.shape[1] > MIN_WIDTH):
        raise ValueError

def resize_img(imgset, height=MIN_HEIGHT, width=MIN_WIDTH):
    return [resize(img, (height, width)) for img in imgset]

def save_images(originals, transforms, name):
    directory = '{}{}/'.format(TEST_IMG_PATH, name)

    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.makedirs(directory)

    for i in range(len(originals)):
        imsave('{}original_{}.jpg'.format(directory, i), originals[i])
        imsave('{}transformed_{}.jpg'.format(directory, i), transforms[i])


#####################################
#       Transform Functions         #
#####################################

def adjust_hsv(imgset, idx):
    values = random.normal(loc=0.0, scale=0.33, size=len(imgset))
    np.clip(values, -1.0, 0.0)

    def adjust_one(img, value):
        hsv = rgb2hsv(img)
        sat = hsv[:,:,idx]
        sat += value
        np.clip(sat, 0., 1., hsv[:,:,idx])
        return hsv2rgb(hsv)

    result_imgs = [adjust_one(img, values[i]) for i, img in enumerate(imgset)]
    return result_imgs, values

def adjust_hue(imgset):
    return adjust_hsv(imgset, 0)

def adjust_saturation(imgset):
    return adjust_hsv(imgset, 1)

def adjust_brightness(imgset):
    return adjust_hsv(imgset, 2)

def adjust_contrast(imgset):
    values = random.normal(loc=0.5, scale=0.152, size=len(imgset))
    np.clip(values, 0.0, 1.0)
    result_imgs = [adjust_sigmoid(img, cutoff=values[i]) for i, img in enumerate(imgset)]
    return result_imgs, values


#####################################
#       Preprocessing Functions     #
#####################################

def hue_only(img):
    return rgb2hsv(img)[:,:,0]

def saturation_only(img):
    return rgb2hsv(img)[:,:,1]

def value_only(img):
    return rgb2hsv(img)[:,:,2]
