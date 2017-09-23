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

        # preprocessed_imgs = get_batches('train', IMG_SIZE)
        # store(preprocessed_imgs, 'preprocessed_dogs.pckl')
    else:
        if os.path.exists('conv_output.pckl'):
            print "Loading convolutions weights"
            dense_model, conv_output = load_convolution_outputs()
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

    print ("Y's shape:", Y.shape)

    Xcats = np.array(preprocessed_cats)
    Xdogs = np.array(preprocessed_dogs)

    X = np.concatenate((Xcats, Xdogs))
    print ("X's shape:", X.shape)

    X = np.swapaxes(X, 1, 3)
    print ("X's reshaped shape:", X.shape)

    model = create_model(IMG_SIZE)

    
    # model.add(Dense(2, activation='sigmoid', input_shape=(1000,)))
    # model.summary()


    # if os.path.exists("model.h5"):
    #     model.load_weights("model.h5")

    classes = get_classes()

    print ("Predict a cat: ")
    cat_prediction = model.predict(X[50,:,:].reshape(1,3,224,224)).argmax(axis = 1)[0]
    print ( cat_prediction, classes[cat_prediction] )

    # print np.rollaxis(X[50,:,:], 0,3).shape

    if False:
        im = Image.fromarray(np.swapaxes(X[50,:,:], 0 , 2) )
        # im = Image.open("train/cats/cat.50.jpg")
        im.show()


    print ("Predict a dog: ")
    dog_prediction = model.predict(X[1001,:,:].reshape(1,3,224,224)).argmax(axis = 1)[0]
    print (dog_prediction, classes[dog_prediction])


    if os.path.exists('conv_output.pckl'):
        print "Loading convolutions weights"
        dense_model, conv_output = load_convolution_outputs()
    else : 
        print ("Save convolution outputs")
        dense_model, conv_output = save_convolution_outputs(X, model)



    print ("Predict a cat: ")
    cat_prediction = model.predict(X[50,:,:].reshape(1,3,224,224)).argmax(axis = 1)[0]
    print ( cat_prediction, classes[cat_prediction] )

    print ("Predict a dog: ")
    dog_prediction = model.predict(X[1001,:,:].reshape(1,3,224,224)).argmax(axis = 1)[0]
    print (dog_prediction, classes[dog_prediction])


    print conv_output.shape

    # print ("Predict a cat: ")
    # cat_prediction = dense_model.predict(conv_output[0,:])
    # cat_prediction = model.predict(X[50,:,:].reshape(1,3,224,224)).argmax(axis = 1)[0]
    # print ( cat_prediction, classes[cat_prediction] )

    # print ("Predict a dog: ")
    # dog_prediction = model.predict(X[1001,:,:].reshape(1,3,224,224)).argmax(axis = 1)[0]
    # print (dog_prediction, classes[dog_prediction])


# eg. {'cats': 0, 'dogs': 1}