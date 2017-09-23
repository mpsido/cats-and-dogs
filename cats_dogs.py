# -*- coding: utf-8 -*-

#preprocessing images
#i need all the images to become a numpy array of the same size

#see convertImg.py

import os, sys
import numpy as np
from tools import *
from image_processing import *
from optparse import OptionParser

from model import *

from keras import backend


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-p", "--preprocess-img", nargs=0, help="preprocess images (load preprocessed images if missing)")
    (opts, args) = parser.parse_args()

    backend.set_image_dim_ordering('th')
    IMG_SIZE = 224

    limit = 1000
    skip_convolution = False
    if opts.__dict__["preprocess_img"] is not None:
        print ("Preprocessing....")
        # cats : 0
        print ("Preprocessing cats images:")
        preprocessed_cats = preprocess_imgs('train/cats', IMG_SIZE, limit = limit)
        store(preprocessed_cats, 'preprocessed_cats.pckl')

        # dogs : 1
        print ("Preprocessing dogs images:")
        preprocessed_dogs = preprocess_imgs('train/dogs', IMG_SIZE, limit = limit)
        store(preprocessed_cats, 'preprocessed_dogs.pckl')
        limit = len(preprocessed_cats)
    else:
        if os.path.exists('conv_output.pckl'):
            print "Loading convolutions weights"
            conv_output = load_convolution_outputs()
            skip_convolution = True
        else : 
            print ("Loading preprocessed images....")
            preprocessed_cats = restore('preprocessed_cats.pckl')
            preprocessed_dogs = restore('preprocessed_dogs.pckl')
            limit = len(preprocessed_cats)


    # labelling the data
    # cats = [1 0] and dogs = [0 1]
    zeros = np.zeros((limit, 1))
    ones  = np.ones((limit, 1))
    Ycats = np.array((ones, zeros))
    Ydogs = np.array((zeros, ones))
    Y = np.concatenate((Ycats, Ydogs), axis=1).reshape((2*limit, 2, 1)).squeeze()

    # labelling the data
    # Ycats = np.zeros((limit, 1))
    # Ydogs = np.ones((limit, 1))
    # Y = np.concatenate((Ycats, Ydogs))

    print (Ycats.shape)
    print (Ydogs.shape)
    print (Y.shape)

    if skip_convolution == True:

        model = create_model(IMG_SIZE)

        dense_models = []
        dense_model = Sequential(input_shape=(1000,))
        for i in range(1):
            dense_models.append(model.layers[-1])
            model.pop()
        for i in range(1):
            dense_model.add(dense_models.pop())

        dense_model.summary()
        # dense_model.pop()
        for layer in dense_model.layers: layer.trainable=False        
        
        dense_model.add(Dense(2, activation='sigmoid', input_shape=(1000,)))
        dense_model.summary()


        dense_model.compile(
                # loss='binary_crossentropy',
                  # optimizer=SGD(lr=10.0),
                  optimizer=RMSprop(),
                  # optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

        dense_model.fit(conv_output, Y)


        dense_model.save_weights("dense_model.h5")

    else:
        Xcats = np.array(preprocessed_cats)
        Xdogs = np.array(preprocessed_dogs)

        X = np.concatenate((Xcats, Xdogs))
        print ("X's shape:", X.shape)

        X = np.swapaxes(X, 1, 3)
        print ("X's reshaped shape:", X.shape)

        model = create_model(IMG_SIZE)

        model.pop()
        for layer in model.layers: layer.trainable=False
        
        
        model.add(Dense(2, activation='sigmoid', input_shape=(1000,)))
        model.summary()


        if os.path.exists("model.h5"):
            model.load_weights("model.h5")


        model.compile(
                # loss='binary_crossentropy',
                  # optimizer=SGD(lr=10.0),
                  optimizer=RMSprop(),
                  # optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        
        model.fit(X,Y, epochs=1, batch_size=50)

        model.save_weights("model.h5")

# eg. {'cats': 0, 'dogs': 1}