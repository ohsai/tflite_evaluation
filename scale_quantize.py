import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tex_print import *
from tqdm import tqdm
print(tf.__version__)

quantize_bits = 8
quantize_scale = 2 ** quantize_bits

class MyModel(tf.Module):
    def __init__(self,N,M) :
        super(MyModel, self).__init__()
        self.left_scale = tf.Variable(tf.random.uniform([N,]),name="left_scale")
        self.right_scale = tf.Variable(tf.random.uniform([M,]),name="right_scale")
        #self.left_scale = tf.Variable(tf.ones([N,]) ,name="left_scale")
        #self.right_scale = tf.Variable(tf.ones([M,]) ,name="right_scale")
        self.scale = tf.constant(float(quantize_scale))

    @tf.function
    def __call__(self,x) :
        left_diag = tf.linalg.diag(self.left_scale)
        right_diag = tf.linalg.diag(self.right_scale)
        y = tf.matmul(tf.matmul(left_diag , x) , right_diag)
        y = tf.scalar_mul(self.scale, y)
        return y

N = 100
M = 200
model = MyModel(N,M)
subsection("Trainable Variables")
debug(model.trainable_variables)
train_loss = tf.keras.metrics.Mean(name='train_loss')

optimizer = tf.keras.optimizers.Adam(lr=0.01)

@tf.function
def quantize(arr_in) :
    scale = tf.constant(float(quantize_scale))
    return tf.math.round(tf.scalar_mul(scale,arr_in))

def loss_fn(arr_before,arr_after) :
    return tf.nn.l2_loss(quantize(arr_before) - arr_after)


@tf.function
def train_step(target_array):
    with tf.GradientTape() as tape:
        transformed_array = model(target_array)
        loss = loss_fn(target_array, transformed_array)   
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

EPOCHS = 10000

arr = np.random.rand(N,M).astype(np.float32)

section("Training")
for epoch in tqdm(range(EPOCHS)):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
        
    train_step(arr)

    template = 'Epoch {}, Loss: {}'
    if(epoch % 100 == 0):
        debug(template.format(epoch+1,
                            train_loss.result()))

debug(quantize(arr))
debug(model(arr))