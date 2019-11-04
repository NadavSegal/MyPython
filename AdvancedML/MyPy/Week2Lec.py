
#import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
#import tensorflow as tf

#tf.enable_eager_execution()
tf.disable_eager_execution()


N = 1000
D=3
x = np.random.random((N,D))
w2 = np.random.random((D,1))
y = x @ w2 + np.random.randn(N,1)*0.20

tf.reset_default_graph()
features = tf.placeholder(tf.float32, shape=(None,D))
target = tf.placeholder(tf.float32,shape=(None,1))
weights = tf.get_variable('W',shape=(D,1),dtype = tf.float32,trainable = True)

predictions = features @ weights

loss = tf.reduce_mean((target - predictions)**2)

optimizer = tf.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(loss)

s = tf.InteractiveSession()
#s = tf.compat.v1.InteractiveSession()
#s = tf.Session()
#s = tf.compat.v1.Session()

#s = tf.compat.v1.InteractiveSession()
saver = tf.train.Saver(tf.trainable_variables())
s.run(tf.global_variables_initializer())
for i in range(3000):
       _,curr_loss,curr_weights = s.run([step,loss,weights],\
                                        feed_dict = {features:x,target:y})
       if i % 50 == 0:
              print(curr_loss,curr_weights)
              print()
              saver.save(s,"logs/2/model.ckpt",global_step = i)

#saver.restore(s,"logs/2/model.ckpt-50")
s.close
print(curr_weights.T)
print(w2.T)