# -*- coding: utf-8 -*-

#preprocessing images
#i need all the images to become a numpy array of the same size

#see convertImg.py

import os, sys
import numpy as np
from tools import *
from image_processing import *
import random
from optparse import OptionParser

from keras.models import Sequential

from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Adam




def ConvBlock(model, layers, filters):
    """
        Adds a specified number of ZeroPadding and Covolution layers
        to the model, and a MaxPooling layer at the very end.

        Args:
            layers (int):   The number of zero padded convolution layers
                            to be added to the model.
            filters (int):  The number of convolution filters to be 
                            created for each layer.
    """
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(filters, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


def FCBlock(model):
    """
        Adds a fully connected layer of 4096 neurons to the model with a
        Dropout of 0.5

        Args:   None
        Returns:   None
    """
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))




vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
def vgg_preprocess(x):
    """
        Subtracts the mean RGB value, and transposes RGB to BGR.
        The mean RGB was computed on the image set used to train the VGG model.

        Args: 
            x: Image array (height x width x channels)
        Returns:
            Image array (height x width x transposed_channels)
    """
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr

def no_process(x):
    # return x[:, ::-1]
    return x

def create_model(input_size):
    """
        Creates the VGG16 network achitecture and loads the pretrained weights.

        Args:   None
        Returns:   None
    """

    model = Sequential()
    # model.add(Lambda(no_process, input_shape=(3,input_size,input_size), output_shape=(3,input_size,input_size)))
    model.add(Lambda(no_process, input_shape=(input_size,input_size, 3), output_shape=(input_size,input_size,3)))

    ConvBlock(model, 2, 16)
    ConvBlock(model, 2, 16)
    ConvBlock(model, 2, 64)
    ConvBlock(model, 2, 64)
    # ConvBlock(model, 2, 128)
    # ConvBlock(model, 3, 256)
    # ConvBlock(model, 3, 512)
    # ConvBlock(model, 3, 512)

    model.add(Flatten())
    FCBlock(model)
    FCBlock(model)
    FCBlock(model)
    FCBlock(model)

    model.add(Dense(1, activation='sigmoid'))

    return model


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-p", "--preprocess-img", nargs=0, help="preprocess images (load preprocessed images if missing)")
    (opts, args) = parser.parse_args()

    print sys.argv
    print opts.__dict__
    print args

    limit = 10000
    if opts.__dict__["preprocess_img"] is not None:
        print ("Preprocessing....")
        # cats : 0
        print ("Preprocessing cats images:")
        preprocessed_cats = preprocess_imgs('train/cats', 128, limit = limit)
        store(preprocessed_cats, 'preprocessed_cats.pckl')

        # dogs : 1
        print ("Preprocessing dogs images:")
        preprocessed_dogs = preprocess_imgs('train/dogs', 128, limit = limit)
        store(preprocessed_cats, 'preprocessed_dogs.pckl')
    else:
        print ("Loading preprocessed images....")
        preprocessed_cats = restore('preprocessed_cats.pckl')
        preprocessed_dogs = restore('preprocessed_dogs.pckl')

    limit = len(preprocessed_cats)

    # labelling the data
    # cats = [1 0] and dogs = [0 1]
    # zeros = np.zeros((limit, 1))
    # ones  = np.ones((limit, 1))
    # Ycats = np.array((ones, zeros))
    # Ydogs = np.array((zeros, ones))
    # Y = np.concatenate((Ycats, Ydogs), axis=1)

    # labelling the data
    Ycats = np.zeros((limit, 1))
    Ydogs = np.ones((limit, 1))
    Y = np.concatenate((Ycats, Ydogs))

    print Ycats.shape
    print Ydogs.shape
    print Y.shape



    Xcats = np.array(preprocessed_cats)
    Xdogs = np.array(preprocessed_dogs)
    print Xcats.shape
    print Xdogs.shape

    X = np.concatenate((Xcats, Xdogs))

    print X.shape

    model = create_model(128)
    model.summary()
    
    model.compile(loss='binary_crossentropy',
              # optimizer=SGD(lr=10.0),
              optimizer=RMSprop(),
              metrics=['accuracy'])

    if os.path.exists("model.h5"):
        model.load_weights("model.h5")

    model.fit(X,Y, epochs=1, batch_size=20)

    model.save_weights("model.h5")
    #model.load_weights("model.h5")



# im = Image.open("train/cats/cat.115.jpg")
# im.show()


# eg. {'cats': 0, 'dogs': 1}