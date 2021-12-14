# -*- coding: utf-8 -*-

import sys
import numpy as np
from tensorflow.keras.layers import Input,Dense,Reshape,Flatten,Dropout,BatchNormalization,Lambda,concatenate,Conv2D,LeakyReLU,Activation
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam,SGD,Nadam,Adamax
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model

class Discriminator(object):
    def __init__(self,width = 28,height = 28,channels = 1,latent_size = 100,model_type = 'simple'):
        self.W = width
        self.H = height
        self.C = channels
        self.CAPACITY = width*height*channels
        self.SHAPE = (width,height,channels)
        
        if model_type == 'simple':
            self.Discriminator = self.model()
            self.OPTIMIZER = Adam(lr = 0.0002,decay = 8e-9)
            self.Discriminator.compile(loss = 'binary_crossentropy',optimizer = self.OPTIMIZER, metrics = ['accuracy'])
        elif model_type == 'DCGAN':
            self.Discriminator = self.dc_model()
            self.OPTIMIZER = Adam(lr = 1e-4,beta_1 = 0.2) 
            self.Discriminator.compile(loss = 'binary_crossentropy',optimizer = self.OPTIMIZER, metrics = ['accuracy'])
        
        self.save_model()
        
        self.Discriminator.summary()
        
    def dc_model(self):
        model = Sequential()
        model.add(Conv2D(64,kernel_size=(5,5),strides=(2,2),padding='same',input_shape=(self.W,self.H,self.C),activation=LeakyReLU(alpha=0.2),name='dc_discriminator'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Conv2D(128,kernel_size=(5,5),strides=(2,2),padding='same',activation=LeakyReLU(alpha=0.2)))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(1,activation='sigmoid'))
        return model
    
    def model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.SHAPE,name='simple_dc'))
        model.add(Dense(self.CAPACITY,input_shape=self.SHAPE))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(int(self.CAPACITY/2))) 
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1,activation='sigmoid'))
        return model

    def summary(self):
        return self.Discriminator.summary()

    def save_model(self):
        plot_model(self.Discriminator,to_file='./data/Discriminator_model.png',show_shapes=True)        

















































