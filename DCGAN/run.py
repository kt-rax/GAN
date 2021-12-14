# -*- coding: utf-8 -*-
from train import Trainer

HEIGHT = 64
WIDTH = 64
CHANNEL = 3
LATENT_SPACE_SIZE = 100
EPOCHS = 100
BATCH = 128
CHECKPOINT = 200
PATH = './data/church_outdoor_train_lmdb_color.npy'

trainer = Trainer(height=HEIGHT,width=WIDTH,channels=CHANNEL,latent_size=LATENT_SPACE_SIZE,epochs=EPOCHS,
                  batch=BATCH,checkpoint=CHECKPOINT,model_type='DCGAN',data_path=PATH)

trainer.train()

