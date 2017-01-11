import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as clayers

# TODO Implement weight sharing

class TransformNet:


    def __init__(self, input, shape=(3,3), cloud_size=1024, batch_size=4):


        with slim.arg_scope([slim.fully_connected], weights_initializer=clayers.xavier_initializer(), \
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.repeat(input, 2, slim.fully_connected, 64)
            net = slim.stack(net, slim.fully_connected, [64, 128, 1024])
            net = tf.reduce_max(net, axis=1)
            net = slim.stack(net, slim.fully_connected, [512, 256, shape[0]*shape[1]])


            if(len(input.get_shape()) > 2):

                # Unpacked transformation matrices
                unstacked_net = tf.unstack(tf.reshape(net, (batch_size, shape[0], shape[0])), axis=0)
                unstacked_input = tf.unstack(input, axis=0)

                results = []
                for transformation_matrix, input in zip(unstacked_net, unstacked_input):
                    results.append(tf.matmul(input,transformation_matrix))



                # 4 x 1024 x 3 -> 4096 x 3
                # 4096 x 12
                self.transformation = tf.stack(results, axis=0)
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

        self.inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.n, 3), name="inputs")

        with slim.arg_scope([slim.fully_connected], weights_initializer=clayers.xavier_initializer(), \
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            self.transform_net_1 = net = TransformNet(self.inputs).transformation

            net = slim.repeat(net, 2, slim.fully_connected, 64)
            self.transform_net_2 = net = TransformNet(net, shape=(64,64)).transformation

            net = slim.stack(net, slim.fully_connected, [64,128,1024])

            net = tf.reduce_max(net, axis=1)

            net = slim.stack(net, slim.fully_connected, [512,256, self.numclasses])

            self.labels = tf.placeholder(tf.int32, shape=(self.batch_size), name="labels")

            self.outputs = net

            print("Output shape: ",  net.get_shape())
            #Define loss function
            slim.losses.sparse_softmax_cross_entropy(self.outputs, self.labels)
            self.loss = slim.losses.get_total_loss(add_regularization_losses=True)

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
        self.summary = tf.merge_all_summaries()


    def train(self):


        optimizer = tf.train.AdamOptimizer(0.001)

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



