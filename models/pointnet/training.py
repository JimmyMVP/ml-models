import os

import numpy as np
import tensorflow as tf


from pointnet.network import PointNet
import pdb


# TODO Data normalization

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


load_files(objects=["guitar", "piano"])

train_files = np.asarray(train_files)
test_files = np.asarray(test_files)
label_mapping = {}

loaded_data = []
loaded_data_labels = []
load_size = 1024


def load_data(load_size, load):


    loaded_data = []
    loaded_data_labels = []

    for i in range(load_size):
        object = train_files[load + i].split("/")[-3]
        cloud = read_off(train_files[load + i])
        # Choose points from cloud with uniform distribution
        indices = np.random.choice(np.arange(cloud.shape[0]), size=1024)
        loaded_data.append(cloud[indices])
        if not object in label_mapping:
            label_mapping[object] = len(label_mapping)

        loaded_data_labels.append(label_mapping[object])
    object = train_files[load + i].split("/")[-3]
    cloud = read_off(train_files[load + i])
    # Choose points from cloud with uniform distribution
    indices = np.random.choice(np.arange(cloud.shape[0]), size=1024)
    loaded_data.append(cloud[indices])
    if not object in label_mapping:
        label_mapping[object] = len(label_mapping)

    loaded_data_labels.append(label_mapping[object])

    return loaded_data, loaded_data_labels


def get_batch(batch_size, current, loaded_data, loaded_data_labels):

    data = []
    labels = []

    sl = slice(current%load_size, current%load_size+batch_size)
    #For every number in batch size read a file
    return np.array(loaded_data[sl])/1000, np.array(loaded_data_labels[sl])

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

epochs = 600
batch_size = 64
learning_rate = 0.001
learning_rate_decay = 0.5


network = PointNet(n=1024, numclasses=len(label_mapping), batch_size=batch_size)
optimise = network.train()

train_saver = tf.train.Saver()

save_path=SUMMARY_DIR + "/checkpoint.ckp"

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    try:
        pass
        #train_saver.restore(sess, save_path)
    except Exception:
        pass


    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    train_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    for epoch in range(epochs):

        np.random.shuffle(train_files)
        if(epoch%20 == 0):
            learning_rate *= learning_rate_decay

        print("Training epoch %d." %(epoch))
        load = 0
        for batch in range(train_files.size // batch_size):

            if(batch*batch_size > load*load_size):
                load_size = min(train_files.size - batch*batch_size, load_size)
                loaded_data, loaded_data_labels = load_data(load_size, load)
                load=+1


            data, labels = get_batch(batch_size, batch, loaded_data, loaded_data_labels)

            feed_dict = {
                network.inputs: data,
                network.labels: labels,
                network.learning_rate: learning_rate
            }

            sess.run(list(network.updates.values()), feed_dict=feed_dict)
            #pdb.set_trace()



            results = sess.run([network.loss, network.summary, optimise] + list(network.metrics_op_map.values()), feed_dict=feed_dict)
            loss = results[0]
            summary = results[1]

            if(batch % 10 == 0):
                train_writer.add_summary(summary)
            metric_values = results[3:]

            print(10*"#", "Epoch: %d/%d Batch: %d/%d Loss: %f" %(epoch, epochs,batch, train_files.size // batch_size, loss), 10*"#")
            for key, value in zip(network.metrics_op_map.keys(), metric_values):
                print("%s: %f" % (key, value))


        if(epoch % 10 == 0):
            train_saver.save(save_path=save_path, sess=sess)





