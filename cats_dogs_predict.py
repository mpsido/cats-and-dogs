# -*- coding: utf-8 -*-

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
    if opts.__dict__["preprocess_img"] is not None:
        print ("Preprocessing....")
        # # cats : 0
        print ("Preprocessing cats images:")
        preprocessed_cats = preprocess_imgs('train/cats', IMG_SIZE, limit = limit)
        store(preprocessed_cats, 'preprocessed_cats.pckl')

        # dogs : 1
        print ("Preprocessing dogs images:")
        preprocessed_dogs = preprocess_imgs('train/dogs', IMG_SIZE, limit = limit)
        store(preprocessed_cats, 'preprocessed_dogs.pckl')

        # preprocessed_imgs = get_batches('train', IMG_SIZE, limit = limit)
        # store(preprocessed_imgs, 'preprocessed_dogs.pckl')
    else:
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

    Xcats = np.array(preprocessed_cats)
    Xdogs = np.array(preprocessed_dogs)
    print (Xcats.shape)
    print (Xdogs.shape)

    X = np.concatenate((Xcats, Xdogs))
    print (X.shape)

    X = X.reshape(X.shape[0], 3, IMG_SIZE, IMG_SIZE)
    print (X.shape)

    model = create_model(IMG_SIZE)
    
    
    # model.add(Dense(2, activation='sigmoid', input_shape=(1000,)))
    # model.summary()


    # if os.path.exists("model.h5"):
    #     model.load_weights("model.h5")

    print ("Predict a cat: ")
    print ( model.predict(X[50:51,:,:]).argmax(axis = 1) )

    im = Image.open("train/cats/cat.50.jpg")
    im.show()


    print ("Predict a dog: ")
    print ( model.predict(X[1001:1002,:,:]).argmax(axis = 1) )

# eg. {'cats': 0, 'dogs': 1}