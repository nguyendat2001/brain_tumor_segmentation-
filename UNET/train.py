import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
%matplotlib inline

import cv2
from tqdm import tqdm_notebook, tnrange
from glob import glob
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import models, layers, regularizers

from data import train_generator
from model import unet_plus, att_unet
from mertrics import specificity, precision, recall, dice_coef, dice_coef_loss, iou, Tversky
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time

# def train(model, loader, optimizer, loss_fn, device):
#     epoch_loss = 0.0

#     model.train()
#     for x, y in loader:
#         x = x.to(device, dtype=torch.float32)
#         y = y.to(device, dtype=torch.float32)

#         optimizer.zero_grad()
#         y_pred = model(x)
#         loss = loss_fn(y_pred, y)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()

#     epoch_loss = epoch_loss/len(loader)
#     return epoch_loss

# def evaluate(model, loader, loss_fn, device):
#     epoch_loss = 0.0

#     model.eval()
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device, dtype=torch.float32)
#             y = y.to(device, dtype=torch.float32)

#             y_pred = model(x)
#             loss = loss_fn(y_pred, y)
#             epoch_loss += loss.item()

#         epoch_loss = epoch_loss/len(loader)
#     return epoch_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Load dataset """
    

    train_files = []
    mask_files = glob('.data/*/*_mask*')

    for i in mask_files:
        train_files.append(i.replace('_mask',''))

    df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})
    df_train, df_test = train_test_split(df,test_size = 0.15)
    df_train, df_val = train_test_split(df_train,test_size = 0.1765)
    
    EPOCHS = 100
    BATCH_SIZE = 32
    learning_rate = 1e-4
    model = unet_plus()
    
    im_height, im_width = 256,256
    train_generator_args = dict(rotation_range=0.2,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                shear_range=0.05,
                                zoom_range=0.05,
                                horizontal_flip=True,
                                fill_mode='nearest')
    train_gen = train_generator(df_train, BATCH_SIZE,
                                    train_generator_args,
                                    target_size=(im_height, im_width))

    test_gener = train_generator(df_val, BATCH_SIZE,
                                    dict(),
                                    target_size=(im_height, im_width))
    
    earlystopping = EarlyStopping(monitor='val_dice_coef',
                              mode='max', 
                              verbose=1, 
                              patience=20
                             )
    # save the best model with lower validation loss

    reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef',
                                  mode='max',
                                  verbose=1,
                                  patience=10,
                                  min_delta=0.0001,
                                  factor=0.2
                                 )


    decay_rate = learning_rate / EPOCHS
    opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[iou, dice_coef, precision,recall,specificity])

    callbacks = [ModelCheckpoint('./results/best_model.hdf5',monitor='val_dice_coef', verbose=1, mode='max',save_best_only=True)]

    history = model.fit(train_gen,
                        steps_per_epoch=len(df_train) / BATCH_SIZE, 
                        epochs=EPOCHS, 
                        callbacks=callbacks,
                        validation_data = test_gener,
                        validation_steps=len(df_val) / BATCH_SIZE)
    
    pd.DataFrame.from_dict(history.history).to_csv('./results/history.csv',index=False)

    