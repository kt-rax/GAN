# -*- coding: utf-8 -*-
import sys 
import numpy as np
from tensorflow.keras.models import Sequential,Model 
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.utils import plot_model

class GAN(object):
    #def __init__(self,generator,discriminator):
    def __init__(self,discriminator,generator):
        self.OPTIMIZER = SGD(lr = 2e-4,nesterov = True)
        self.Generator = generator
        
        self.Discriminator = discriminator
        self.Discriminator.trainable = False
        
        self.gan_model = self.model()
        self.gan_model.compile(loss = 'binary_crossentropy',optimizer = self.OPTIMIZER)
        self.save_model()
        self.gan_model.summary()
        
    def model(self):
        model = Sequential()
        model.add(self.Generator)
        model.add(self.Discriminator)
        
        return model
    
    def summary(self):
        return self.gan_model.summary()
    
    def save_model(self):
        plot_model(self.gan_model,to_file='./data/gan_model.png')