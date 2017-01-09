import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os

from pointnet.network import PointNet


DATAROOT = "/data/vlasteli/ModelNet40/"
DATASET = ""
LOGDIR = ""

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
            points = np.asarray([float(x) for x in line.split()])
            cloud.append(points)

    return np.asarray(cloud)


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


def load_files():

    for obj in os.listdir(DATAROOT):
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


def get_batch(batch_size):

# TODO Uniform sampling of the input pointcloud

    data = []
    labels = []
    for i in range(batch_size):
        object = train_files[i].split("/")[-3]
        cloud = read_off(train_files[i])
        data.append(cloud)
        if not object in label_mapping:
            label_mapping[object] = len(label_mapping)
        labels.append(label_mapping[object])


    return np.array(data), np.array(labels)



network = PointNet(n=1024, numclasses=40)

train_op = network.train()
epochs = 100
batch_size = 32

with tf.Session() as sess:

    for epoch in range(epochs):
        print("Training epoch %d." %(epoch))
        for batch in range(batch_size):

            data, labels = get_batch(batch_size)
            feed_dict = {
                network.inputs: data,
                network.labels: labels
            }

            loss = sess.run([network.loss, network.optimize], feed_dict=feed_dict)

            print("Loss: %f" %(loss))



SUMMARY_DIR = os.environ["SUMMARY_DIR"] + "shapenet40/summary"
LOG_DIR = os.environ["SUMMARY_DIR"] + "shapenet40/log"




