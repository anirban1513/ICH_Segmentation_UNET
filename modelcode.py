import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
import math
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np

def jaccard_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1-jac)*100

def dice_coef(y_true, y_pred, smooth=1e-4):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
    return K.mean(intersection / union)

def dice_loss(y_true, y_pred, smooth=1e-4):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
    return 1 - K.mean(intersection / union)

#Windowing
def window_ct (ct_scan, w_level=40, w_width=120):  #for brain window,level is 40 and width of window is 120 as given in the paper. 
    w_min = w_level - w_width // 2 #Window minimum or left end side of the window. which is (40 - (120/2) = -20)
    w_max = w_level + w_width // 2 #Window maximum or right end side of the window. which is (40 + (120/2) = 100)
    window_image = ct_scan.copy()
    window_image[window_image < w_min] = w_min
    window_image[window_image > w_max] = w_max

    return window_image

class DataGenerator(tf.compat.v2.keras.utils.Sequence):
  def __init__(self, data_dir, mask_dir, batch_size, dim,
                 to_fit, shuffle = True):
    self.batch_size = batch_size
    self.data_list = os.listdir(data_dir)
    self.data_dir = data_dir
    self.mask_dir = mask_dir
    self.to_fit = to_fit
    self.dim = dim
    self.shuffle = shuffle
    self.n = 0
    self.on_epoch_end()
    

  def __next__(self):
   # Get one batch of data
   data = self.__getitem__(self.n)
   # Batch index
   self.n += 1

   # If we have processed the entire dataset then
   if self.n >= self.__len__():
     self.on_epoch_end
     self.n = 0

   return data
 
 
  def __len__(self):
   #Return the number of batches of the dataset
    return math.ceil(len(self.indexes)/self.batch_size)



  def __getitem__(self,index):
   # Generate indexes of the batch
   indexes = self.indexes[index*self.batch_size:
            (index+1)*self.batch_size]



   # Find list of IDs
   list_IDs_temp = [self.data_list[k] for k in indexes]
   X = self._generate_x(list_IDs_temp)
   if self.to_fit:
     if self.to_fit:
       y = self._generate_y(list_IDs_temp)
       return X, y
     else:
       return X


  def on_epoch_end(self):
    self.indexes = np.arange(len(self.data_list))

    if self.shuffle:
      np.random.shuffle(self.indexes)

  
  def _generate_x(self,list_IDs_temp):
    
    X = np.zeros((self.batch_size, self.dim,self.dim))
    for i,ID in enumerate(list_IDs_temp):
      tmpx= np.load(os.path.join(self.data_dir, list_IDs_temp[i]))    
      #Windowing Data and Scaling between 0 to 255
      X[i] = cv2.resize(window_ct(tmpx),(self.dim,self.dim))
      #Normalizing
      #X[i] = (X[i]/255).astype('float32')
    return X[:,:,:,np.newaxis]

   
  def _generate_y(self,list_IDs_temp):
    y = np.zeros((self.batch_size, self.dim, self.dim))

    for i,ID in enumerate(list_IDs_temp):
      tmpy = np.load(os.path.join(self.mask_dir, list_IDs_temp[i]))
      y[i] = cv2.resize(tmpy,(self.dim,self.dim))

    return y[:,:,:,np.newaxis]

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNELS = 1
IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

def unet(inputs= (IMAGE_HEIGHT,IMAGE_WIDTH,1),lr=3e-4):
    inputs = Input((IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    conv01 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(inputs)
    conv01 = BatchNormalization()(conv01)
    conv01 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv01)
    conv01 = BatchNormalization()(conv01)
    pool01 = MaxPooling2D(pool_size=(2, 2))(conv01)

    conv02 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool01)
    conv02 = BatchNormalization()(conv02)
    conv02 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv02)
    conv02 = BatchNormalization()(conv02)
    pool02 = MaxPooling2D(pool_size=(2, 2))(conv02)

    conv03 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool02)
    conv03 = BatchNormalization()(conv03)
    conv03 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv03)
    conv03 = BatchNormalization()(conv03)
    pool03 = MaxPooling2D(pool_size=(2, 2))(conv03)

    conv04 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool03)
    conv04 = BatchNormalization()(conv04)
    conv04 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv04)
    conv04 = BatchNormalization()(conv04)
    drop04 = Dropout(0.5)(conv04)
    pool04 = MaxPooling2D(pool_size=(2, 2))(drop04)

    conv05 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool04)
    conv05 = BatchNormalization()(conv05)
    conv05 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv05)
    conv05 = BatchNormalization()(conv05)
    drop05 = Dropout(0.5)(conv05)

    up06 = Conv2D(512, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop05))
    merge06 = concatenate([drop04, up06], axis=3)
    conv06 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge06)
    conv06 = BatchNormalization()(conv06)
    conv06 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv06)
    conv06 = BatchNormalization()(conv06)

    up07 = Conv2D(256, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv06))
    merge07 = concatenate([conv03, up07], axis=3)
    conv07 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge07)
    conv07 = BatchNormalization()(conv07)
    conv07 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv07)
    conv07 = BatchNormalization()(conv07)

    up08 = Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv07))
    merge08 = concatenate([conv02, up08], axis=3)
    conv08 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge08)
    conv08 = BatchNormalization()(conv08)
    conv08 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv08)
    conv08 = BatchNormalization()(conv08)

    up09 = Conv2D(64, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv08))
    merge09 = concatenate([conv01, up09], axis=3)
    conv09 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge09)
    conv09 = BatchNormalization()(conv09)
    conv09 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv09)
    conv09 = BatchNormalization()(conv09)

    conv09 = Conv2D(2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv09)
    conv09 = BatchNormalization()(conv09)
    conv09 = Conv2D(2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv09)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv09)
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=3e-4), loss=dice_loss, metrics=[dice_coef])
    #model.summary()

    return model




 


