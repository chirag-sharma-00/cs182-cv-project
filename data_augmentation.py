from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import pandas as pd


def flip_x(img):
    #returns copy of img that is flipped horizontally
    flipped = np.fliplr(img)
    return flipped


def flip_y(img):
    #returns copy of img that is flipped vertically
    flipped = np.fliplr(img)
    return flipped


def down_translation(img, y):
    #returns copy of img that is translated downwards by y pixels and
    #fills in gaps with edge pixels in that row from original image
    translated = np.zeros(img.shape)
    for i in range(img.shape[0]):
        if (i >= y and i < img.shape[0] + y):
            translated[i, :, :] = img[i - y, :, :]
        elif (y > 0):
            translated[i, :, :] = img[0, :, :]
        else:
            translated[i, :, :] = img[img.shape[0] - 1, :, :]
    return translated.astype('uint8')


def right_translation(img, x):
    #returns copy of img that is translated right by x pixels and
    #fills in gaps with edge pixelx in that row from original image
    translated = np.zeros(img.shape)
    for j in range(img.shape[1]):
        if (j >= x and j < img.shape[1] + x):
            translated[:, j, :] = img[:, j - x, :]
        elif (x > 0):
            translated[:, j, :] = img[:, 0, :]
        else:
            translated[:, j, :] = img[:, img.shape[1] - 1, :]
    return translated.astype('uint8')


def gaussian_noise(img, mean, std):
    #returns copy of img with gaussian noise with the provided
    #mean and std added to each pixel
    noise = np.random.normal(mean, std, size=img.shape)
    noisy = img + noise
    noisy = np.minimum(noisy, 255 * np.ones(noisy.shape))
    noisy = np.maximum(noisy, np.zeros(noisy.shape))
    return noisy.astype('uint8')


def rotation(img, theta):
    #returns copy of img, rotated anticlockwise by theta degrees
    img_ob = Image.fromarray(img, 'RGB')
    rotated = img_ob.rotate(theta)
    return np.array(rotated)


def augmented_data(img):
    #returns list of:
    # - original image
    # - image flipped horizontally
    # - image flipped vertically
    # - 5 random horizontal translations of image by x in [-15, 15]
    # - 5 random vertical translations of image by y in [-15, 15]
    # - 5 noise-augmentions of image by random mean in [-100, 100]
    #   and random std in [0, 10]
    # - 7 random rotations of image by theta in [0, 180]
    augmented = [img]

    augmented.append(flip_x(img))

    augmented.append(flip_y(img))

    for x in np.random.randint(-15, 16, 5):
        augmented.append(right_translation(img, x))

    for y in np.random.randint(-15, 16, 5):
        augmented.append(down_translation(img, y))

    for i in range(5):
        mean = np.random.uniform(-100, 100)
        std = np.random.uniform(0, 10)
        augmented.append(gaussian_noise(img, mean, std))

    for theta in np.random.uniform(0, 180, 7):
        augmented.append(rotation(img, theta))

    return augmented

def augmented_data_from_path(img_path):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    return augmented_data(img)