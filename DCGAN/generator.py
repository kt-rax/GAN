# -*- coding: utf-8 -*-

import sys
import numpy as np
from tensorflow.keras.layers import Dense,Reshape,Input,BatchNormalization,Conv2D,MaxPooling2D
from tensorflow.keras.layers import LeakyReLU,UpSampling2D,Conv2DTranspose
from tensorflow.keras.optimizers import Adam,SGD,Nadam,Adamax
from tensorflow.keras.models import Sequential,Model 
from tensorflow.keras.utils import plot_model
from tensorflow.keras import initializers

class Generator(object):
    def __init__(self,width = 28,height = 28,channels = 1,latent_size = 100,model_type = 'DCGAN'):
        self.W = width
        self.H = height
        self.C = channels
        self.LATENT_SPACE_SIZE = latent_size
        self.latent_space = np.random.normal(0,1,(self.LATENT_SPACE_SIZE,))
        
        if model_type == 'simple':
            self.Generator = self.model()
            self.OPTIMIZER = Adam(lr = 0.0002,decay = 8e-9)
            self.Generator.compile(loss = 'binary_crossentropy',optimizer = self.OPTIMIZER)
        elif model_type == 'DCGAN':
            self.Generator = self.dc_model()
            self.OPTIMIZER = Adam(lr = 1e-4,beta_1 = 0.2)
            self.Generator.compile(loss = 'binary_crossentropy',optimizer = self.OPTIMIZER,metrics = ['accuracy'])
        
        self.save_model()
        self.Generator.summary()
        
    def dc_model(self):
        model = Sequential()
        model.add(Dense(256*8*8,activation = LeakyReLU(alpha=0.2),input_dim = self.LATENT_SPACE_SIZE,name = 'dc_generator'))
        model.add(BatchNormalization())
        
        model.add(Reshape((8,8,256)))
        model.add(UpSampling2D())        
        
        model.add(Conv2D(128,kernel_size=(5,5),padding='same',activation=LeakyReLU(alpha=0.2)))
        model.add(BatchNormalization())
        model.add(UpSampling2D())
        
        model.add(Conv2D(64,kernel_size=(5,5),padding='same',activation=LeakyReLU(alpha=0.2)))
        model.add(BatchNormalization())
        model.add(UpSampling2D())        
        
        model.add(Conv2D(self.C,kernel_size=(5,5),padding='same',activation='tanh'))
        
        return model
    
    def model(self,block_starting_size=128,num_blocks=4):
        model = Sequential()
    
        block_size = block_starting_size
        model.add(Dense(block_size,input_shape=(self.LATENT_SPACE_SIZE,)))
        model.add(LeakyReLU(alpha=0.2)) 
        model.add(BatchNormalization(momentum=0.8))
        
        for i in range(num_blocks-1):
            block_size = block_size * 2
            model.add(Dense(block_size))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
        
        model.add(Dense(self.W * self.H * self.C,activation='tanh'))
        model.add(Reshape((self.W,self.H,self.C)))
        return model
    
    def summary(self):
        return self.Generator.summary()
    
    def save_model(self):
        plot_model(self.Generator,to_file='./data/Generator_model.png',show_shapes=True)
        
        
                



