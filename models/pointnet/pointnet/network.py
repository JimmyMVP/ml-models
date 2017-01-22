import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as clayers
import numpy as np

# TODO Implement weight sharing
# TODO Transformation matrix weight regularization
# TODO Batch size doesn't influence network topology

def batch_norm_fully_connected(input, outputs, scope=None, weights_initializer=slim.xavier_initializer()):
    net = slim.fully_connected(input, outputs, scope=scope, weights_initializer=weights_initializer)
    return slim.batch_norm(net, decay=0.7)

def batch_norm_fully_connected_shared(input, outputs, scope=None):
    #net = slim.conv2d(input, outputs, kernel_size=1,stride=1)
    filters = tf.Variable(tf.random_normal([outputs,1,1]), dtype=tf.float64)
    net=tf.nn.conv1d(input, filters=filters, stride=1, padding='SAME')
    return slim.batch_norm(net, decay=0.5)

def identity_initializer(shape):
    return tf.constant_initializer(np.reshape(np.identity(shape[0], dtype=np.float64), shape[0]**2))

class TransformNet:


    def __init__(self, input, shape=(3,3), cloud_size=1024, batch_size=4):
        self.orth_loss = 0

        with slim.arg_scope([slim.fully_connected], weights_initializer=slim.init_ops.truncated_normal_initializer(), \
                            weights_regularizer=slim.l2_regularizer(0.0005), scope=None):

            net = batch_norm_fully_connected(input, 64)
            net = batch_norm_fully_connected(net, 128)
            net = batch_norm_fully_connected(net, 1024)

            net = tf.reduce_max(net, axis=1)

            net = batch_norm_fully_connected(net, 512)
            net = batch_norm_fully_connected(net, 256)
            net = slim.fully_connected(net, shape[0]**2)


            if(len(input.get_shape()) > 2):

                # Unpacked transformation matrices
                unstacked_net = tf.unstack(tf.reshape(net, (batch_size, shape[0], shape[0])), axis=0)
                unstacked_input = tf.unstack(input, axis=0)

                results = []
                orth_losses = []

                for transformation_matrix, input in zip(unstacked_net, unstacked_input):
                    transformed_input = tf.matmul(input,transformation_matrix)
                    results.append(transformed_input)

                    # Orthogonality loss
                    orth_loss = tf.reduce_sum(tf.matmul(transformation_matrix, transformation_matrix, transpose_b=True)-tf.constant(np.identity(shape[0], dtype=np.float32)))
                    orth_losses.append(orth_loss)

                #L2 Regularization for transformation matrixes
                self.matrix_reg_loss = tf.reduce_sum(tf.pow(tf.reduce_mean(net, axis=0), 2)) /10000
                #slim.losses.add_loss(self.matrix_reg_loss)

                #Calculate batch orthogonality loss
                orth_loss = tf.stack(orth_losses, axis=0)
                orth_loss = tf.reduce_mean(orth_loss, axis=0)

                self.orth_loss = orth_loss

                slim.losses.add_loss(self.orth_loss)


                # 4 x 1024 x 3 -> 4096 x 3
                # 4096 x 12
                self.transformation = tf.stack(results, axis=0)
                tf.summary.histogram("transformation", self.transformation)
            else:
                self.transformation = tf.matmul(input, net)
                self.transformation = tf.reshape(self.transformation, shape=(batch_size, cloud_size, shape[0]))







class PointNet:

    def __init__(self, n=1024, numclasses=40, batch_size=4):

        self.n = n
        self.numclasses = numclasses
        self.batch_size = batch_size
        self.createComputationGraph()

    def createComputationGraph(self):

        self.inputs = tf.placeholder(tf.float64, shape=(self.batch_size, self.n, 3), name="inputs")

        with slim.arg_scope([slim.fully_connected], weights_initializer=clayers.xavier_initializer(), \
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            self.transform_net_1 =  TransformNet(self.inputs, batch_size=self.batch_size)
            net = self.transform_net_1.transformation

            net = batch_norm_fully_connected(net, 64)
            net = batch_norm_fully_connected(net, 64)

            self.transform_net_2 = TransformNet(net, shape=(64,64), batch_size=self.batch_size)
            net = self.transform_net_2.transformation

            self.orth_loss = self.transform_net_2.orth_loss + self.transform_net_1.orth_loss
            self.matrix_reg_loss = self.transform_net_2.matrix_reg_loss + self.transform_net_1.matrix_reg_loss


            net = batch_norm_fully_connected(net, 64)
            net = batch_norm_fully_connected(net, 128)
            net = batch_norm_fully_connected(net, 1024)

            net = tf.reduce_max(net, axis=1)

            net = batch_norm_fully_connected(net, 512)
            tf.summary.histogram("output-2", net)
            net = batch_norm_fully_connected(net, 256)
            tf.summary.histogram("output-1", net)

            net = slim.fully_connected(net, self.numclasses)
            # Output summary
            tf.summary.histogram("output", net)

            self.labels = tf.placeholder(tf.int32, shape=(self.batch_size), name="labels")

            self.outputs = net

            print("Output shape: ",  net.get_shape())
            #Define loss function
            self.cross_entropy_loss = 100*tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.outputs, self.labels))
            slim.losses.add_loss(self.cross_entropy_loss)

            self.reg_loss = tf.reduce_sum(slim.losses.get_regularization_losses())

            self.loss = slim.losses.get_total_loss(add_regularization_losses=False)

            tf.summary.scalar("loss", self.loss)


        # Classification
        self.predictions = tf.argmax(self.outputs, axis=1, name="classification")
        print("Predictions shape: ", self.predictions.get_shape())

        #Define metrics
        self.metrics_op_map, self.updates = slim.metrics.aggregate_metric_map({
            "eval/recall": slim.metrics.streaming_recall(self.predictions, self.labels),
            "eval/accuracy": slim.metrics.streaming_accuracy(self.predictions, self.labels),
            "eval/precision": slim.metrics.streaming_precision(self.predictions, self.labels)
        })


        #Define summary
        self.summary = tf.summary.merge_all()


    def train(self):


        # Learning rate is dynamic
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        optimize = slim.learning.create_train_op(total_loss=slim.losses.get_total_loss(add_regularization_losses=False), optimizer=optimizer)

        #gradients = tf.gradients(self.loss, tf.trainable_variables())
        #tf.summary.histogram("gradients", gradients)


        return optimize

    def __call__(self, *args, **kwargs):

        inputs = args[0]

        with tf.Session() as sess:

            feed_dict = {
                self.inputs: inputs
            }

            classes = sess.run(self.classify, feed_dict=feed_dict)

        return classes



