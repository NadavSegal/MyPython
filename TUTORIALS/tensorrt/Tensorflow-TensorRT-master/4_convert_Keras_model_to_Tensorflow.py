#!/usr/bin/env python
# coding: utf-8

# ### Convert Keras model to Tensorflow model

# In[1]:


# import the needed libraries
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#in default keral lives the BN param at trainable status. in order to optimize we need all to be un trainable
tf.keras.backend.set_learning_phase(0) #use this if we have batch norm layer in our network
from tensorflow.keras.models import load_model

# path we wanna save our converted TF-model
#MODEL_PATH = "./model/tensorflow/big/model1"
MODEL_PATH = "./model/tensorflow/small/model_small"

# load the Keras model
#model = load_model('./model/modelLeNet5.h5')
model = load_model('./model/modelLeNet5_small.h5')

# save the model to Tensorflow model
saver = tf.train.Saver()
#tf.train.export_meta_graph(filename='my-model.meta')

sess = tf.keras.backend.get_session()
save_path = saver.save(sess, MODEL_PATH)

print("Keras model is successfully converted to TF model in "+MODEL_PATH)


# ### Keras to TensorRT
# ![alt text](pictures/Keras_to_TensorRT.png)
# 
# ### Tensorflow to TensorRT
# ![alt text](pictures/tf-trt_workflow.png)
