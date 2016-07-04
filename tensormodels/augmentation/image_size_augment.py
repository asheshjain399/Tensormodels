"""
Author: Ashesh Jain [asheshjain399@gmail.com]
MIT License
"""

import cv2
from numpy.random import RandomState
rand = RandomState(123456789)


def random_crop(img, label, crop_size):
    """
    Returns a random crop of the image
    TODO: Make this method more generic to modify label
    Args:
        img: input image
        label: input label
        crop_size: list [crop_width,crop_height]
    """

    ndim = img.ndim
    img_shape = img.shape
    h = img_shape[0]
    w = img_shape[1]

    crop_width = crop_size[0]
    crop_height = crop_size[1]
    wmin = rand.randint(0, w - crop_width)
    hmin = rand.randint(0, h - crop_height)
    wmax = wmin + crop_width
    hmax = hmin + crop_height

    if ndim > 2:
        return img[hmin:hmax, wmin:wmax, :], label
    else:
        return img[hmin:hmax, wmin:wmax], label


def resize(img, label, new_size, interpolation=cv2.INTER_CUBIC):
    """
    Resizes the image
    TODO: Make this method more generic to modify the label
    Args:
        img: input image
        new_size: new image size [new_width,new_height]
        interpolation: Kind of interpolation to use
    """

    return cv2.resize(img, new_size, interpolation=interpolation), label


def random_flip(img, label):
    """
    Randomly flips the image
    TODO: Make this method more generic to modify the label
    Args:
        img: input image
    """

    if rand.rand() < 0.5:
        return cv2.flip(img, 1), label
    else:
        return img, label