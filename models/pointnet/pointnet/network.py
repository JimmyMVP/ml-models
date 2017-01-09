import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as clayers



class TransformNet:


    def __init__(self, input, shape=(3,3)):

        if(len(input.get_shape()) > 2):

            reshaped = tf.reshape(input, shape=(-1, shape[0]))
            self.transformation = tf.matmul(reshaped, tf.ones(shape=shape))
            print(self.transformation.get_shape())
            self.transformation = tf.reshape(self.transformation, shape=(-1, 1024, shape[0]))
        else:
            self.transformation = tf.matmul(input, tf.ones(shape=shape))






class PointNet:

    def __init__(self, n=1024, numclasses=40):

        self.n = n
        self.numclasses = numclasses
        self.createComputationGraph()

    def createComputationGraph(self):

        self.inputs = tf.placeholder(tf.float32, shape=(None, self.n, 3))

        with slim.arg_scope([slim.fully_connected], weights_initializer=clayers.xavier_initializer(), \
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = TransformNet(self.inputs).transformation

            net = slim.repeat(net, 2, slim.fully_connected, 64)
            net = TransformNet(net, shape=(64,64)).transformation

            net = slim.stack(net, slim.fully_connected, [64,128,1024])

            net = tf.reduce_max(net, axis=1)

            net = slim.stack(net, slim.fully_connected, [512,256, self.numclasses])

            self.labels = tf.placeholder(tf.float32, shape=(None, self.numclasses))

            self.outputs = net
            self.loss = slim.losses.softmax_cross_entropy(self.outputs, self.labels)

        # Classification
        self.classify = tf.argmax(self.inputs, axis=1, name="classification")

        return

    def train(self):


        optimizer = tf.train.AdamOptimizer(0.001, 0.9)

        optimize = slim.learning.create_train_op(total_loss=slim.losses.get_total_loss(add_regularization_losses=True), optimizer=optimizer)

        return optimize

    def __call__(self, *args, **kwargs):

        inputs = args[0]

        with tf.Session() as sess:

            feed_dict = {
                self.inputs: inputs
            }

            classes = sess.run(self.classify, feed_dict=feed_dict)

        return classes



