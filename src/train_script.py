#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 13:59:40 2018

@author: jqg
"""

from Myclass import loadData, DataGenerator, Histories
#import numpy as np
import random
#from keras.applications.resnet50 import ResNet50
from keras import metrics
from resnet import ResnetBuilder

N_CLASS = 28
DATA_DIR = 'dataset224/'
n_valid = 1000

Ids,labels = loadData('all.csv',N_CLASS,DATA_DIR)

index = random.sample(range(len(Ids)),1000)
train,_ = loadData('train.csv',N_CLASS,DATA_DIR)
validation,_ = loadData('validation.csv',N_CLASS,DATA_DIR)


params = {'dim': (224,224),
          'batch_size': 32,
          'n_classes': 28,
          'n_channels': 3,
          'shuffle': True}


training_generator = DataGenerator(train, labels, **params)
validation_generator = DataGenerator(validation, labels, **params)
histories = Histories()
model = ResnetBuilder.build_resnet_50((3,224, 224), 28)
model.compile(loss="categorical_crossentropy", optimizer="sgd",
              metrics=[metrics.categorical_accuracy])
print (len(Ids)-len(train)-len(validation))
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    callbacks=[histories],
                    epochs=1)