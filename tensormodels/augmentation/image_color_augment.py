"""
Author: Ashesh Jain [asheshjain399@gmail.com]
MIT License
"""

import cv2
import numpy as np
from numpy.random import RandomState
rand = RandomState(123456789)

def random_saturation(img, label, lower=0.5, upper=1.5):
    """
    Multiplies saturation with a constant and clips the value between [0,1.0]
    Args:
        img: input image in float32
        label: returns label unchanged
        lower: lower val for sampling
        upper: upper val for sampling
    """
    alpha = lower + (upper - lower) * rand.rand()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # saturation should always be within [0,1.0]
    hsv[:, :, 1] = np.clip(alpha * hsv[:, :, 1], 0.0, 1.0)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), label

def random_brightness(img, label, delta=32.0):
    """
    Adds a constant delta value to the image
    Args:
        img: input image in float32
    """
    return (img + rand.randint(-delta, delta)), label

def random_contrast(img, label, lower=0.5, upper=1.5):
    """
    Multiplies each channel with a constant
    Args:
        img: input image in float32
    """

    alpha = lower + (upper - lower) * rand.rand()
    ndim = img.ndim
    if ndim == 3:
        channels = img.shape[-1]
        for i in range(channels):
            channel_mean = np.mean(img[:, :, i])
            img[:, :, i] = (img[:, :, i] - channel_mean) * alpha + channel_mean
    elif ndim == 2:
        channel_mean = np.mean(img)
        img = (img - channel_mean) * alpha + channel_mean
    return img, label

def random_hue(img, label, max_delta=10):
    """
    Rotates the hue channel
    Args:
        img: input image in float32
        max_delta: Max number of degrees to rotate the hue channel
    """
    # Rotates the hue channel by delta degrees
    delta = -max_delta + 2.0 * max_delta * rand.rand()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hchannel = hsv[:, :, 0]
    hchannel = delta + hchannel

    # hue should always be within [0,360]
    idx = np.where(hchannel > 360)
    hchannel[idx] = hchannel[idx] - 360
    idx = np.where(hchannel < 0)
    hchannel[idx] = hchannel[idx] + 360

    hsv[:, :, 0] = hchannel
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), label