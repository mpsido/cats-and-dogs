# -*- coding: utf-8 -*-

#preprocessing images
#i need all the images to become a numpy array of the same size

#see convertImg.py


import PIL
from PIL import Image
import numpy as np
import os, sys
import random
import pickle #store and load variables
# import shelve #store and load variables


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

import sys


def progress(count, total, status=''):
# As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 

def preprocess_imgs(img_size, limit = 100):
    i = 0
    resized_img = []
    for img in os.listdir("train/cats"):
        if i >= limit:
            break
        im = Image.open("train/cats/"+img)
        reduced_img = resize_image(im, img_size)
        resized_img.append(np.array(reduced_img))
        i+=1
        progress(i, limit)
    print "\n"
    return resized_img



# im = Image.open("train/cats/cat.115.jpg")
# im.show()

def store(obj, target):
    file = open(target, 'wb')
    pickle.Pickler(file).dump(obj)
    # pickle.dump(obj, f)
    file.close()

def restore(source):
    f = open(source, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


    
# def store(obj, target):
#     my_shelf = shelve.open(target,'n') # 'n' for new
#     my_shelf[target] = obj
#     my_shelf.close()


# def restore(source):
#     my_shelf = shelve.open(source)
#     data = my_shelf[source]
#     my_shelf.close()
#     return data



if __name__ == "__main__":
    limit = 1000#0
    if False :
        preprocessed_imgs = preprocess_imgs(128, limit = limit)
        store(preprocessed_imgs, 'preprocessed_imgs.pckl')
    else:
        preprocessed_imgs = restore('preprocessed_imgs.pckl')

    print random.randint(0,limit - 1)
    print preprocessed_imgs[random.randint(0,limit - 1)].shape
    # print random.randint(0,limit - 1)
    # preprocessed_imgs[random.randint(0,limit - 1)].show()
    # print random.randint(0,limit - 1)
    # preprocessed_imgs[random.randint(0,limit - 1)].show()
