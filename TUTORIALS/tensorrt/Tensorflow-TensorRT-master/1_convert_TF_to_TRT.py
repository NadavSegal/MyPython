"""https://www.youtube.com/watch?v=AIGOSz2tFP8&list=PLkRkKTC6HZMwdtzv3PYJanRtR6ilSCZ4f"""

# ## What is TensorRT?
# 
# TensorRT is an optimization tool provided by NVIDIA that applies graph optimization and layer fusion, and finds the fastest implementation of a deep learning model. In other words, TensorRT will optimize our deep learning model so that we expect a faster inference time than the original model (before optimization), such as 5x faster or 2x faster. The bigger model we have, the bigger space for TensorRT to optimize the model. Furthermore, this TensorRT supports all NVIDIA GPU devices, such as 1080Ti, Titan XP for Desktop, and Jetson TX1, TX2 for embedded device.

# ## Standard workflow for optimizing Tensorflow model to TensorRT
# 
# ![alt text](pictures/tf-trt_workflow.png)
# 
# ## Library I use in this video series
# Pre-requrement: Install TensorRT by following this tutorial [here](https://medium.com/@ardianumam/installing-tensorrt-in-ubuntu-dekstop-1c7307e1dcf6) for Ubuntu dekstop or [here](https://medium.com/@ardianumam/installing-tensorrt-in-jetson-tx2-8d130c4438f5) for Jetson devices
# 1. Tensorflow 1.12
# 2. OpenCV 3.4.5
# 3. Pillow 5.2.0
# 4. Numpy 1.15.2
# 5. Matplotlib 3.0.0

# ### (a) Read input: Tensorflow model and (b) Convert to frozen model (*.pb)

# In[8]:
from memory_profiler import profile    
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#import tensorflow.contrib.tensorrt as trt
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.platform import gfile


# has to be use this setting to make a session for TensorRT optimization
with tf.compat.v1.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50))) as sess: 
    # import the meta graph of the tensorflow model
    #saver = tf.train.import_meta_graph("./model/tensorflow/big/model1.meta")
    saver = tf.train.import_meta_graph("./model/tensorflow/small/model_small.meta")
    # then, restore the weights to the meta graph
    #saver.restore(sess, "./model/tensorflow/big/model1")
    saver.restore(sess, "./model/tensorflow/small/model_small")
    
    # specify which tensor output you want to obtain 
    # (correspond to prediction result)
    your_outputs = ["output_tensor/Softmax"]
    
    # convert to frozen model
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess, # session
        tf.get_default_graph().as_graph_def(),# graph+weight from the session
        output_node_names=your_outputs)
    #write the TensorRT model to be used later for inference
    with gfile.FastGFile("./model/frozen_model.pb", 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    print("Frozen model is successfully stored!")


# ### (c) Optimize the frozen model to TensorRT graph

# In[2]:


# convert (optimize) frozen model to TensorRT model
trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,# frozen model
    outputs=your_outputs,
    max_batch_size=2,# specify your max batch size
    max_workspace_size_bytes=2*(10**9),# specify the max workspace
    precision_mode="FP32") # precision, can be "FP32" (32 floating point precision) or "FP16"

#write the TensorRT model to be used later for inference
with gfile.FastGFile("./model/TensorRT_model.pb", 'wb') as f:
    f.write(trt_graph.SerializeToString())
print("TensorRT model is successfully stored!")


# ### (optional) Count how many nodes/operations before and after optimization

# In[3]:


# check how many ops of the original frozen model
all_nodes = len([1 for n in frozen_graph.node])
print("numb. of all_nodes in frozen graph:", all_nodes)

# check how many ops that is converted to TensorRT engine
trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
all_nodes = len([1 for n in trt_graph.node])
print("numb. of all_nodes in TensorRT graph:", all_nodes)


# In[ ]:




