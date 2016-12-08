
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
from utils.utils import voxel_grid
import os
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from numpy.random import normal


# In[2]:

#Parameters

RES = (32,32,32)

#Helper functions (layer creation)

"""
    Default stride size is filterSize, which means no overlapping.
"""
def conv3d(x, numFilters, filterSize, stride=False, padding='VALID'):
    if(not stride):
        stride = filterSize
    inChannels = x.get_shape()[-1].value
    with tf.name_scope("conv"):
        #The filter weights must be of the form (width, height, depth, channels, outputChannels[numberOfFilters])
        filter=tf.Variable(tf.truncated_normal(shape=[3,3,3,inChannels,numFilters], dtype=tf.float32))
        return tf.nn.conv3d(input=x, filter=filter,  strides=[1,stride,stride,stride,1], padding=padding)


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def pool3d(x, filterSize=3, stride=1,padding='VALID'):
    return tf.nn.max_pool3d(x, [1, filterSize, filterSize, filterSize, 1], [1,stride,stride,stride,1], padding, name="maxpool")


def fully_connected(x, size, activation=tf.nn.relu):
    with tf.name_scope("fully_connected"):
        b = tf.Variable(tf.zeros([size]))
        weights = tf.Variable(tf.truncated_normal([x.get_shape()[1].value, size]))
        return activation(b + tf.matmul(x, weights))


#Helper functions batch reading

object_csvs = np.array(list(filter(lambda x: ".csv" in x, os.listdir("../dataset/objects"))))
#Shuffleing the csvs
np.random.shuffle(object_csvs)


label_names = np.array(list(map(lambda x: x.split(".")[0], object_csvs)))
labels = label_names.reshape((-1,1))

le = LabelBinarizer(sparse_output=False)

labels = le.fit_transform(labels).reshape(-1,26)

def get_batch(size=3, ind=0):
    
    mini_batch_x = []
    mini_batch_y = []
    for i in range(size):
        obj = voxel_grid(np.genfromtxt("../dataset/objects/"+object_csvs[ind+i], delimiter=",")[:,3:6], res=RES)
        mini_batch_x.append(obj[:,:,:,None])
        #mini_batch_x[i].append(object_csvs[ind+i].split(".")[0])
        #write code for labels
    
    
    return (np.array(mini_batch_x), labels[ind:ind+size], label_names[ind:ind+size])

#Random batch for testing
def get_random_batch(size=3, ind=0):
    
    mini_batch_x = []
    mini_batch_y = []
    
    return (np.random.rand(size, 64,64,64,1), labels[ind:ind+size])

#Generator
generator = {}
#Discriminator
discriminator = {}


# In[3]:

#Build model

#Architecture discriminator

def discriminator():

    d = {}

    with tf.name_scope("discriminator"):

        d['x'] = tf.placeholder(shape=[None,RES[0], RES[1], RES[2],1], dtype=tf.float32, name="Input")
        d['y'] = tf.placeholder(shape=[None, labels[0].size], dtype=tf.float32, name="Labels")


        d['batch_size'] = tf.shape(d['x'])[0] 

        print("Input shape: ", d['x'].get_shape())
        print("Labels shape: ", d['y'].get_shape())

        d['conv1'] = conv3d(d['x'], 3,3,2)
        print("Conv1: ", d['conv1'].get_shape())

        d['maxpool1'] = pool3d(d['conv1'],3,2)
        print("Pool1: ", d['maxpool1'].get_shape())

        d['conv2'] = conv3d(d['maxpool1'], 3,3,2)
        print("Conv2: ", d['conv2'].get_shape())

        d['maxpool2'] = pool3d(d['conv2'],3,2)
        print("Pool2: ", d['maxpool2'].get_shape())




        d['flat'] = flatten(d['maxpool2'])

        print("Flatten layer: ", d['flat'].get_shape())
        d['output'] = fully_connected(d['flat'], 26)

        print("Output shape: ", d['output'].get_shape())

        #Loss
        d['vars'] = tf.trainable_variables()
        d['lossL2'] = tf.add_n([tf.nn.l2_loss(v) for v in d['vars']])*0.01
        d['loss'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(d['output'], d['y'])) + d['lossL2']

        print("Loss shape: ", d['loss'].get_shape())

        #Training
        optimizer = tf.train.AdamOptimizer(0.04)
        d['train_step'] = optimizer.minimize(d['loss'])

        d['accuracy'] = tf.reduce_mean(tf.cast(tf.equal(tf.round(d['output']), d['y']), tf.float32))

    return d


#Generator

def generator():

    g = {}
    with tf.name_scope("generator"):

        g['x'] = tf.placeholder(shape=[None, 64], dtype=tf.float32, name="input")


        g['fc1'] = fully_connected(g['x'], 64, activation=tf.nn.tanh)

        g['fc2'] = fully_connected(g['fc1'], 128, activation=tf.nn.tanh)

        #Final layer has to have the same size of output as voxelized grid
        g['fc3'] = fully_connected(g['fc2'], np.prod(RES), activation=tf.nn.tanh)

        g['output'] = tf.reshape(g['fc3'], RES)

        g['loss'] = -discriminator['loss']

        optimizer = tf.train.AdamOptimizer(0.004)


        g['train_step'] = optimizer.minimize(g['loss'])

    return g


# # Model Training
# 
# Here comes the model training

def l(name=""):
    print(40*"#", name, 40*"#")


BATCH_SIZE = 2
EPOCHS = 10


discriminator = discriminator()
generator = generator()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    


    for i in range(EPOCHS):
        for batch in range(object_csvs.size//BATCH_SIZE):

            batch_x, batch_y, class_names = get_batch(BATCH_SIZE,0)
            feed_dict = {
                discriminator['x']: batch_x.astype(np.float32),
                discriminator['y']: batch_y.astype(np.float32),
                generator['x']: normal(size=BATCH_SIZE*64).reshape((BATCH_SIZE, 64))
            }
            #_, loss, accuracy= sess.run([discriminator['train_step'], discriminator['loss'], discriminator['accuracy']], feed_dict=feed_dict) 
            

            _, loss, accuracy, _, g_loss = sess.run([discriminator['train_step'], discriminator['loss'], discriminator['accuracy'], generator['train_step'], generator['loss']], feed_dict=feed_dict)

            if(batch%5==0):
                
                l("DISCRIMINATOR STATS")
                print("Number of classes: ", np.unique(class_names).size)
                print("Batch step %d,  loss: %.2f, accuracy: %.2f" % (batch, loss, accuracy))
                l("GENERATOR STATS")
                print("Number of classes: ", np.unique(class_names).size)
                print("Batch step %d,  loss: %.2f" % (batch, g_loss))




        print("Epoch %d loss: " %(i), sess.run([loss], feed_dict=feed_dict))
        
    
    

