# -*- coding: utf-8 -*-
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Reshape,Flatten,Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

class Discriminator(object):
    def __init__(self,width = 28, height = 28, channels = 1, latent_size = 100):
        self.CAPACITY = width*height*channels
        self.SHAPE = (width,height,channels)
        self.OPTIMIZER = Adam(lr=0.0002,decay=8e-9)

        self.Discriminator = self.model()
        self.Discriminator.compile(loss='binary_crossentropy',optimizer=self.OPTIMIZER,metrics=['accuracy'])
        self.Discriminator.summary()
    
    def model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.SHAPE,name='Discriminator'))
        model.add(Dense(self.CAPACITY,input_shape=self.SHAPE))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(int(self.CAPACITY/2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1,activation='sigmoid'))
        return model
   
    def summary(self):
        return self.Discriminator.summary()
    
    def save_model(self):
        plot_model(self.Discriminator,to_file='./data/Discriminator_Model.png',show_shapes=True)
        
