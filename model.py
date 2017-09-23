# -*- coding: utf-8 -*-


import numpy as np

from keras.models import Sequential

from keras.utils.data_utils import get_file
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Adam

from tools import * #store arrays

import json


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
    model.add(Lambda(vgg_preprocess, input_shape=(3,input_size,input_size), output_shape=(3,input_size,input_size)))
    # model.add(Lambda(no_process, input_shape=(input_size,input_size, 3), output_shape=(input_size,input_size,3)))

    ConvBlock(model, 2, 64)
    ConvBlock(model, 2, 128)
    ConvBlock(model, 3, 256)
    ConvBlock(model, 3, 512)
    ConvBlock(model, 3, 512)

    model.add(Flatten())
    FCBlock(model)
    FCBlock(model)

    model.add(Dense(1000, activation='softmax'))


    fname = 'vgg16.h5'
    model.load_weights(fname)

    return model

def get_classes():
    """
        Downloads the Imagenet classes index file and loads it to self.classes.
        The file is downloaded only if it not already in the cache.
    """
    with open('imagenet_class_index.json') as f:
        class_dict = json.load(f)
    return [class_dict[str(i)][1] for i in range(len(class_dict))]



def save_convolution_outputs(X, model, fpath = 'conv_output.pckl', nb_dense_layers = 6):
    dense_models = []
    for i in range(nb_dense_layers):
        dense_models.append(model.layers[-1])
        model.pop()

    conv_output = model.predict(X)

    store(conv_output, fpath)

    #restore dense layers
    for i in range(6):
        pop_model = dense_models[-1-i]
        print len(dense_models)
        print type(pop_model)
        model.add(pop_model)

    return dense_models, conv_output


def load_convolution_outputs(fpath = 'conv_output.pckl'):
    conv_output = restore(fpath)

    return conv_output
