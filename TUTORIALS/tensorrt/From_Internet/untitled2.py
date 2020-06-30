#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:09:09 2020

@author: nadav
"""

from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf
from tensorflow.python.framework.func_graph import def_function
from tensorflow.python.framework import tensor_spec

class MyModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv2 = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', data_format='channels_last')
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[None, 32, 32, 3], dtype=tf.float32)
    ])
    def call(self, inputs, training=True, **kwargs):
        inputs = self.conv2(inputs)  # remove this line, work fine!!
        x = tf.reshape(inputs, [-1, inputs.shape[-1]])
        x = self.dense(x)
        return x

model = MyModel()
out = model(tf.random.normal((2, 32, 32, 3)))
tf.saved_model.save(model, './saved_model/', {'model_key': model.call})

params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode='INT8',
                                                    use_calibration=True)
converter = trt.TrtGraphConverterV2(input_saved_model_dir='./saved_model/', conversion_params=params,
                                    input_saved_model_signature_key='model_key')
def input_fn():
    for i in range(10):
        yield tf.random.normal((1, 32, 32, 3))
converter.convert(calibration_input_fn=input_fn)
converter.save('./model.trt')