# -*- coding: utf-8 -*-

import PIL
from PIL import Image
import numpy as np
import os, sys
from tools import *
from keras.preprocessing import image, sequence


#inspired from https://www.kaggle.com/gauss256/preprocess-images

def norm_image(img):
    """
    Normalize PIL image

    Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
    """
    img_y, img_b, img_r = img.convert('YCbCr').split()

    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

    img_nrm = img_ybr.convert('RGB')

    return img_nrm

def resize_image(img, size):
    """
    Resize PIL image

    Resizes image to be square with sidelength size. Pads with black if needed.
    """
    # Resize
    n_x, n_y = img.size
    if n_y > n_x:
        n_y_new = size
        n_x_new = int(size * n_x / n_y + 0.5)
    else:
        n_x_new = size
        n_y_new = int(size * n_y / n_x + 0.5)

    img_res = img.resize((n_x_new, n_y_new), resample=PIL.Image.BICUBIC)

    # Pad the borders to create a square image
    img_pad = Image.new('RGB', (size, size), (128, 128, 128))
    ulc = ((size - n_x_new) // 2, (size - n_y_new) // 2)
    img_pad.paste(img_res, ulc)

    return img_pad



def preprocess_imgs(img_folder, img_size, limit = 100):
    i = 0
    processed_img = []
    for img in os.listdir(img_folder):
        if i >= limit:
            break
        im = Image.open(os.path.join(img_folder,img))
        reduced_img = resize_image(im, img_size)
        norm_img = norm_image(reduced_img)
        processed_img.append(np.array(norm_img))
        i+=1
        progress(i, limit)
    print "\n"
    return processed_img


def get_batches(img_folder, img_size, limit = 100):
    gen=image.ImageDataGenerator()
    return gen.flow_from_directory(img_folder, target_size=(img_size,img_size),
        class_mode='categorical', shuffle=True, batch_size=8).next()
