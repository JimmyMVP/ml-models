import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

class TransformNet:


    def __init__(self, n=1024, shape=(3,3)):
        print("Created transform net")
        self.transformation = tf.ones(shape=shape)



transfromation = TransformNet().transformation




