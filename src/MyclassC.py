#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:14:21 2018

@author: jqg
"""


from keras.preprocessing.image import img_to_array, load_img
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
import keras.backend as K
import tensorflow as tf
import random 
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(224,224), n_channels=3,
                 n_classes=28, shuffle=True,dataAug=False, imgGenerator=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.dataAug = dataAug
        self.imgGenerator = imgGenerator
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size,self.n_classes), dtype=int)
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            try:
                img = load_img(ID)
                tmp = img_to_array(img)
            except:
                tmp = []
                suffix = ['_green.png','_blue.png','_yellow.png','_red.png']
                for s in suffix:
                    img = load_img(ID+s)
                    tt = img_to_array(img)
                    if tmp == []:
                        tmp = tt
                    else:
                        tmp = np.concatenate((tmp,tt),axis=2)
            
            if self.dataAug:
                rr = random.randint(0,3)
                for r in range(rr):
                    tmp = np.rot90(tmp)
                X[i,] = self.imgGenerator.random_transform(tmp)  
            else:
                X[i,] = tmp

            # Store class
            y[i] = self.labels[ID]
        X = self.imgGenerator.standardize(X)
        return X, y
    
def loadData(csv,N_CLASS,DATA_DIR,suffix=None):
    Ids = []
    labels = {}
    with open(csv,'r') as f:
        next(f)
        for l in f:
            if suffix == None:
                name = l.strip().split(',')[0]
            else:
                name = l.strip().split(',')[0]+suffix
            l = l.strip().split(',')[1]
            l = l.split()
            l = list(map(lambda x:int(x),l))
            oneHot = oneHotR(l,N_CLASS)
            Ids.append(DATA_DIR+name)
            labels[DATA_DIR+name] = oneHot
    return Ids,labels

def oneHotR(array,nClass):
    res = np.zeros(nClass)
    res[array] = 1
    return res 

def multiPred(y_true, y_pred):
    y = K.greater(y_pred,0.5)
    y_p = K.cast(y,K.floatx())
    score = tf.reduce_all(K.equal(y_true,y_p),1)
    score = K.cast(score,K.floatx())
    return tf.reduce_mean(score)

class Histories(keras.callbacks.Callback):
    
	def on_train_begin(self, logs={}):
		self.aucs = []
		self.losses = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		y_pred = self.model.predict(self.validation_data[0])
		self.aucs.append(roc_auc_score(self.validation_data[1], y_pred))
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return