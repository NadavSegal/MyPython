import sys
sys.path.append("..")
import grading
import download_utils
import tqdm_utils
from tqdm import tqdm

import numpy as np


# import tensorflow as tf
import tensorflow.compat.v1 as tf
from keras_utils2 import reset_tf_session
s = reset_tf_session()

import keras
from keras.models import Sequential
from keras import layers as L


for layer in generator.layers:
       print("Encoder layer %s has: (%s units)"%(layer.name,layer.output_shape))
       
gen_optimizer = tf.train.AdamOptimizer(1e-4).minimize(g_loss,var_list=generator.trainable_weights)
disc_optimizer =  tf.train.GradientDescentOptimizer(1e-3).minimize(d_loss,var_list=discriminator.trainable_weights)       
