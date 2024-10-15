from tensorflow.keras.layers import Input, concatenate, LeakyReLU, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, UpSampling2D, ZeroPadding2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from skimage.transform import resize
import os
import csv
import PIL
import numpy as np
import random
import cv2
import imutils
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2




def downsampling_block(input_tensor, n_filters):
  x = Conv2D(filters=n_filters, kernel_size=(3,3), padding='same')(input_tensor)
  x = LeakyReLU(alpha=0.2)(x)
  x = BatchNormalization()(x)

  x = Conv2D(filters=n_filters, kernel_size=(3,3), padding='same')(x)
  x = LeakyReLU(alpha=0.2)(x)
  x = BatchNormalization()(x)
  return x

def upsampling_block(input_tensor, n_filters, name, concat_with):
  x = UpSampling2D((2, 2), interpolation='bilinear', name=name)(input_tensor)
  x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', name=name+"_convA")(x)
  x = LeakyReLU(alpha=0.2)(x)

  x = concatenate([x, concat_with], axis=3)

  x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', name=name+"_convB")(x)
  x = LeakyReLU(alpha=0.2)(x)
  x = BatchNormalization()(x)

  x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', name=name+"_convC")(x)
  x = LeakyReLU(alpha=0.2)(x)
  x = BatchNormalization()(x)
  return x


def build(height, width, depth):
  # input
  i = Input(shape=(height, width, depth))

  iresnet = InceptionResNetV2(include_top = False, weights = "imagenet", input_tensor = i)
  iresnet.summary()

  # encoder

  conv1 = iresnet.get_layer("input_layer").output
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = iresnet.get_layer("activation").output
  conv2 = ZeroPadding2D(((1,0),(1,0)))(conv2)  
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = iresnet.get_layer("activation_3").output
  conv3 = ZeroPadding2D((1,1))(conv3)  
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = iresnet.get_layer("activation_74").output
  conv4 = ZeroPadding2D(((2,1),(2,1)))(conv4)  
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
  

  # bottleneck
  conv5 = iresnet.get_layer("activation_161").output
  conv5 = ZeroPadding2D((1,1))(conv5) 
  conv5 = LeakyReLU(alpha=0.2)(conv5)


  # decoder
  conv6 = upsampling_block(conv5, 256, "up1", concat_with=conv4)
  conv7 = upsampling_block(conv6, 128, "up2", concat_with=conv3)
  conv8 = upsampling_block(conv7, 64, "up3", concat_with=conv2)
  conv9 = upsampling_block(conv8, 32, "up4", concat_with=conv1)
  
  # output
  o = Conv2D(filters=1, kernel_size=3, strides=(1,1), activation='sigmoid', padding='same', name='conv10')(conv9)

  model = Model(inputs=i, outputs=o)
  return model

model = build(HEIGHT, WIDTH, 3)
