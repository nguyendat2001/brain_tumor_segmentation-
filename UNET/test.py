

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

# def visual_his():
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'val'], loc='upper left')
#     plt.show()


if __name__ == "__main__":
    
    train_files = []
    mask_files = glob('.data/*/*_mask*')

    for i in mask_files:
        train_files.append(i.replace('_mask',''))

    df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})
    df_train, df_test = train_test_split(df,test_size = 0.15)
    df_train, df_val = train_test_split(df_train,test_size = 0.1765)
    im_height, im_width = 256,256
    
    model = load_model('./results/best_model.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef, 'precision': precision,'recall':recall, 'specificity':specificity })
    test_gen = train_generator(df_test, BATCH_SIZE,
                                dict(),
                                target_size=(im_height, im_width))

    val_gen = train_generator(df_val, BATCH_SIZE,
                                    dict(),
                                    target_size=(im_height, im_width))

    results_train = model.evaluate(train_gen, steps=len(df_train) / BATCH_SIZE)
    results_test = model.evaluate(test_gen, steps=len(df_test) / BATCH_SIZE)
    results_val = model.evaluate(val_gen, steps=len(df_val) / BATCH_SIZE)

    result = pd.DataFrame({  'train':({'loss':results_train[0],'iou':results_train[1],'dice_coef':results_train[2],'precision':results_train[3],'recall':results_train[4],'specificity':results_train[5]}),
                             'test':({'loss':results_test[0],'iou':results_test[1],'dice_coef':results_test[2],'precision':results_test[3],'recall':results_test[4],'specificity':results_test[5]}),
                             'val':({'loss':results_val[0],'iou':results_val[1],'dice_coef':results_val[2],'precision':results_val[3],'recall':results_val[4],'specificity':results_val[5]}) })

    pd.DataFrame.from_dict(result).to_csv('./results/result.csv',index=False)
    
    data = pd.read_csv('./results/history.csv')
    