import os

import numpy as np
import tensorflow as tf

from pointnet.network import PointNet
import pdb

DATAROOT = "/data/vlasteli/ModelNet40/"
DATASET = ""
LOGDIR = "/data/vlasteli/pointnet_summary/"

SUMMARY_DIR = os.environ["SUMMARY_DIR"] + "shapenet40/summary"
LOG_DIR = os.environ["SUMMARY_DIR"] + "shapenet40/log"

train_files = []
test_files = []


"""
    Returns pointcloud vector
"""
def read_off(file_name):
    cloud = []
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines[2:-1]:
            points = [float(x) for x in line.split()]
            if(len(points) > 3):
                points = points[1:]
            cloud.append(points)

    return np.array(cloud)


def load_object(obj):

    print("Loading ", obj, " ...")

    #Training data
    training_dir = DATAROOT+obj + "/train/"
    test_dir = DATAROOT + obj + "/test/"

    training_data = np.array([read_off(training_dir + f) for f in os.listdir(training_dir)])

    test_data = np.array([read_off(test_dir + f) for f in os.listdir(test_dir)])

    return np.array((training_data, test_data))


def load_data(save=False):

    for obj in ["radio", "piano"]:
        object_data = load_object(obj)
        #np.save(DATAROOT+obj+"/data.np", object_data)
        print("Loaded object")


def load_files(objects=["guitar", "piano"]):

    for obj in os.listdir(DATAROOT):
        if not obj in objects:
            continue
        train = DATAROOT+obj +"/train/"
        test = DATAROOT+obj +"/test/"
        for f in os.listdir(train):
            train_files.append(train+f)
        for f in os.listdir(test):
            train_files.append(test + f)


load_files()

train_files = np.asarray(train_files)
test_files = np.asarray(test_files)
label_mapping = {}


def get_batch(batch_size, current):

    data = []
    labels = []
    for i in range(batch_size):
        object = train_files[current+i].split("/")[-3]
        cloud = read_off(train_files[current+i])
        indices = np.random.choice(np.arange(cloud.shape[0]), size=1024)

        data.append(cloud[indices])
        if not object in label_mapping:
            label_mapping[object] = len(label_mapping)
        labels.append(label_mapping[object])

    return np.array(data), np.array(labels)

network = PointNet(n=1024, numclasses=2)
optimise = network.train()

train_op = network.train()
epochs = 100
batch_size = 8


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    train_writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph)

    for epoch in range(epochs):

        np.random.shuffle(train_files)

        print("Training epoch %d." %(epoch))
        for batch in range(train_files.size // batch_size):

            data, labels = get_batch(batch_size, batch)

            feed_dict = {
                network.inputs: data,
                network.labels: labels
            }

            sess.run(list(network.updates.values()), feed_dict=feed_dict)
            #pdb.set_trace()



            results = sess.run([network.loss, network.summary, optimise] + list(network.metrics_op_map.values()), feed_dict=feed_dict)
            loss = results[0]
            summary = results[1]

            if(batch % 10 == 0):
                train_writer.add_summary(summary)
            metric_values = results[3:]

            print("Batch: %d/%d Loss: %f" %(batch, train_files.size // batch_size, loss))
            for key, value in zip(network.metrics_op_map.keys(), metric_values):
                print("%s: %f" % (key, value))





SUMMARY_DIR = os.environ["SUMMARY_DIR"] + "shapenet40/summary"
LOG_DIR = os.environ["SUMMARY_DIR"] + "shapenet40/log"




