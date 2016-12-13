# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
from utils.utils import voxel_grid, tf_confusion_metrics
import os
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pickle
import sys
import json
import pdb

# In[2]:

# Parameters

RES = (64, 64, 64)
LIMITS = (20, 20, 3)

# Helper functions (layer creation)

"""
    Default stride size is filterSize, which means no overlapping.
"""


def conv3d(x, numFilters, filterSize, stride=False, padding='SAME', activation=False):
    if (not stride):
        stride = filterSize
    inChannels = x.get_shape()[-1].value
    with tf.name_scope("conv"):
        # The filter weights must be of the form (width, height, depth, channels, outputChannels[numberOfFilters])
        filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, inChannels, numFilters], dtype=tf.float32, stddev=0.2))
        if(not activation):
            return tf.nn.conv3d(input=x, filter=filter, strides=[1, stride, stride, stride, 1], padding=padding)
        else:
            return activation(tf.nn.conv3d(input=x, filter=filter, strides=[1, stride, stride, stride, 1], padding=padding))

def conv3d_transpose(x, numFilters, filterSize, stride=False, padding='SAME'):
    if (not stride):
        stride = filterSize
    inChannels = x.get_shape()[-1].value
    with tf.name_scope("deconv"):
        # The filter weights must be of the form (width, height, depth, channels, outputChannels[numberOfFilters])
        filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, inChannels, numFilters], dtype=tf.float32))
        return tf.nn.conv3d_transpose(value=x, filter=filter, strides=[1, stride, stride, stride, 1], padding=padding)


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def pool3d(x, filterSize=3, stride=1, padding='SAME'):
    return tf.nn.max_pool3d(x, [1, filterSize, filterSize, filterSize, 1], [1, stride, stride, stride, 1], padding,
                            name="maxpool")


def fully_connected(x, size, activation=tf.nn.relu):
    with tf.name_scope("fully_connected"):
        b = tf.Variable(tf.truncated_normal([size]))
        weights = tf.Variable(tf.truncated_normal([x.get_shape()[1].value, size]))
        return activation(b + tf.matmul(x, weights))


# Helper functions batch reading

"""
    Returns tuple: (X, Y, labels)
"""


def get_batch(dataset, size=None, ind=0):
    if (size == None):
        size = dataset.size

    mini_batch_x = []
    mini_batch_y = []
    for i in range(size):
        obj = voxel_grid(np.genfromtxt(DATASET + dataset[ind + i], delimiter=",")[:, 3:6], res=RES, limits=LIMITS)
        mini_batch_x.append(obj[:, :, :, None])
        # mini_batch_x[i].append(object_csvs[ind+i].split(".")[0])
        # write code for labels

    return (np.array(mini_batch_x), labels[ind:ind + size], label_names[ind:ind + size])


# Random batch for testing
def get_random_batch(size=3, ind=0):
    mini_batch_x = []
    mini_batch_y = []

    return (np.random.rand(size, 64, 64, 64, 1), labels[ind:ind + size])


ENV = os.environ
DATASET = ENV["DATASETS"] + "sidney/objects/"
SUMMARY_DIR = ENV["SUMMARY_DIR"]

object_csvs = np.array(list(filter(lambda x: ".csv" in x, os.listdir(DATASET))))
# Shuffleing the csvs
np.random.shuffle(object_csvs)

label_names = np.array(list(map(lambda x: x.split(".")[0], object_csvs)))
labels = label_names.reshape((-1, 1))

print("Labels: ", labels)

le = LabelBinarizer(sparse_output=False)

labels = le.fit_transform(labels).reshape(-1, 26)

if not os.path.exists(SUMMARY_DIR + "inputs"):
    os.makedirs(SUMMARY_DIR + "inputs")

if not os.path.exists(SUMMARY_DIR + "outputs"):
    os.makedirs(SUMMARY_DIR + "outputs")

"""
Testing the input
"""
input_output = open(SUMMARY_DIR + "inputs/sidney", "wb")
learned_output = open(SUMMARY_DIR + "outputs/sidney", "wb")

x_b, _, _ = get_batch(object_csvs, 30, 0)

np.save(input_output, x_b)
# sys.exit(1)

# In[3]:

# Build model

# Architecture

x = tf.placeholder(shape=[None, RES[0], RES[1], RES[2], 1], dtype=tf.float32, name="Input")
y = tf.placeholder(shape=[None, labels[0].size], dtype=tf.float32, name="Labels")

batch_size = tf.shape(x)[0]

print("Input shape: ", x.get_shape())
print("Labels shape: ", y.get_shape())

conv1 = conv3d(x, 3, 28, stride=2, activation=tf.nn.relu)
print("Conv1: ", conv1.get_shape())

maxpool1 = pool3d(conv1, 2, stride=2)
print("Pool1: ", maxpool1.get_shape())

conv2 = conv3d(maxpool1, 3, 28, stride=2, activation=tf.nn.relu)
print("Conv2: ", conv2.get_shape())

maxpool2 = pool3d(conv2, 2, stride=22)
print("Pool2: ", maxpool2.get_shape())

# deconv1 = conv3d_transpose(conv2, 3, 28, stride=2)

flat = flatten(maxpool2)
print("Flatten layer: ", flat.get_shape())

fc1 = fully_connected(flat, np.sum([RES[0], RES[1]]))

# Dropout layer
keep_prob = tf.placeholder(tf.float32)
fc1_dropout = tf.nn.dropout(fc1, keep_prob)

output_logits = fully_connected(fc1_dropout, np.prod(RES), activation=tf.nn.relu)

output = tf.nn.softmax(output_logits)

print("Output shape: ", output.get_shape())

# Metrics
equals = tf.cast(tf.equal(tf.round(output), tf.minimum(flatten(x), 1)), tf.float32)
precision = tf.reduce_mean(equals)

mask = tf.cast(tf.minimum(flatten(x), 1), tf.bool)
recall = tf.reduce_sum(tf.boolean_mask(equals, mask)) / tf.reduce_sum(tf.minimum(flatten(x), 1))

# Loss
vars = tf.trainable_variables()
lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.2
loss = tf.reduce_mean(
    tf.nn.weighted_cross_entropy_with_logits(output_logits, flatten(x), flatten(x) * np.prod(RES) + 1)) + lossL2
print("Loss shape: ", loss.get_shape())

# Training
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss)

# # Model Training
# 
# Here comes the model training

BATCH_SIZE = 80
EPOCHS = 1000

VALIDATION_PERCENT = 0.01

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    def l():
        print(80 * "#")


    validation_size = round(object_csvs.size * VALIDATION_PERCENT)

    iter_dropout = 1
    for i in range(EPOCHS):

        if iter_dropout > 0.5:
            iter_dropout = 1 - float(i) / float(EPOCHS)

        training_size = object_csvs.size - validation_size
        training_files = object_csvs[0:training_size]
        validation_files = object_csvs[training_size:-1]

        print("Validation size: ", validation_files.size)

        for batch in range(training_files.size // BATCH_SIZE):

            batch_x, batch_y, class_names = get_batch(training_files, BATCH_SIZE, batch * BATCH_SIZE)
            feed_dict = {
                x: batch_x.astype(np.float32),
                y: batch_y.astype(np.float32),
                keep_prob: iter_dropout
            }
            _, r_loss, r_precision, r_output, r_recall = sess.run([train_step, loss, precision, output, recall],
                                                                  feed_dict=feed_dict)
            if (batch % 5 == 0):
                print("Number of classes: ", np.unique(class_names).size)
                print("Batch step %d,  loss: %.2f, precision: %.5f, recall: %.5f, dropout: %.2f" % (
                batch, r_loss, r_precision, r_recall, iter_dropout))
                r_output = r_output.reshape((-1, RES[0], RES[1], RES[2]))
                np.save(learned_output, r_output)

        x_valid, y_valid, class_names = get_batch(validation_files)
        # Turn off dropout for validation
        feed_dict = {
            x: x_valid,
            y: y_valid,
            keep_prob: 1
        }

        r_loss, r_precision, r_predictions, r_equals, r_recall = sess.run([loss, precision, output, equals, recall],
                                                                          feed_dict=feed_dict)
        l()
        print("Number of non zero in x: ", x_valid[[x_valid != 0]].size)
        print("Number of non zero in y: ", r_predictions[[r_predictions > 0.5]].size)
        print("Epoch %d loss: %.2f precision: %.5f recall: %.5f" % (i, r_loss, r_precision, r_recall))
        l()
