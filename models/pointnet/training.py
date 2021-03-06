import os

import numpy as np
import tensorflow as tf


from pointnet.network import PointNet
import pdb

from sklearn import preprocessing

import argparse
import sys



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


objects = ["guitar", "piano"]
def load_files(objects=objects):

    for obj in os.listdir(DATAROOT):
        if not obj in objects:
            continue
        train = DATAROOT+obj +"/train/"
        test = DATAROOT+obj +"/test/"
        for f in os.listdir(train):
            train_files.append(train+f)
        for f in os.listdir(test):
            train_files.append(test + f)


label_mapping = {}


def get_batch(batch_size, current, cloud_size=1024):

    data = []
    labels = []
    for i in range(batch_size):
        object = train_files[current+i].split("/")[-3]
        cloud = read_off(train_files[current+i])
        indices = np.random.choice(np.arange(cloud.shape[0]), size=cloud_size)

        data.append(cloud[indices])
        if not object in label_mapping:
            label_mapping[object] = len(label_mapping)
        labels.append(label_mapping[object])

    #Scale the so that it has zero mean and unit variance

    return preprocessing.scale(np.array(data).reshape(-1)).reshape(batch_size, cloud_size, -1), np.array(labels)


"""
    Training main function.
"""

if __name__ == '__main__':


    #Remove file name from args
    sys.argv.pop(0)

    #Arguement parsing
    parser = argparse.ArgumentParser(description="Parse parameters for training of the PointNet model.")

    parser.add_argument("--learning-rate", metavar="l", type=float, help="learning rate for optimizer",
                        default=0.001)
    parser.add_argument("--epochs", metavar="e", type=int, help="number of epochs",
                        default=600)
    parser.add_argument("--batch-size", metavar="bs", type=int, help="size of training batch",
                        default=64)
    parser.add_argument("--learning-rate-decay", metavar="ld", type=float, help="learning rate decay for optimizer",
                        default=0.5)
    parser.add_argument("--learning-rate-decay-freq", metavar="ldf", type=int, help="how frequently (in epochs) should the decay apply",
                        default=20)
    parser.add_argument("--save", metavar="ldf", type=int,
                        help="how frequently (in epochs) should the decay apply",
                        default=20)

    args = parser.parse_args(sys.argv)

    print("Loading training and test files...")
    load_files()
    train_files = np.asarray(train_files)
    test_files = np.asarray(test_files)

    device_name = "/gpu:0"


    print("Creating network...")

    network = PointNet(n=1024, numclasses=len(objects), batch_size=args.batch_size)
    optimise = network.train()


    train_saver = tf.train.Saver()

    save_path=SUMMARY_DIR + "/checkpoint.ckp"

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        try:
            pass
            #train_saver.restore(sess, save_path)
        except Exception:
            pass


        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        train_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

        for epoch in range(args.epochs):

            np.random.shuffle(train_files)
            if(epoch%20 == 0):
                args.learning_rate *= args.learning_rate_decay

            print("Training epoch %d." %(epoch))
            for batch in range(train_files.size // args.batch_size):

                data, labels = get_batch(args.batch_size, batch)

                feed_dict = {
                    network.inputs: data,
                    network.labels: labels,
                    network.learning_rate: args.learning_rate
                }

                sess.run(list(network.updates.values()), feed_dict=feed_dict)
                #pdb.set_trace()



                results = sess.run([network.loss, network.summary, network.reg_loss,
                                    network.orth_loss, network.cross_entropy_loss,optimise] + list(network.metrics_op_map.values()), feed_dict=feed_dict)
                loss = results[0]
                summary = results[1]
                reg_loss = results[2] /loss
                orth_loss = results[3] /loss
                cross_entropy = results[4] /loss

                if(batch % 10 == 0):
                    train_writer.add_summary(summary)
                metric_values = results[6:]
                #pdb.set_trace()
                print(10*"#", "Epoch: %d/%d Batch: %d/%d Loss: %f" %(epoch, args.epochs,batch, train_files.size // args.batch_size, loss), 10*"#")

                print("reg_loss=%f cross_entropy=%f orth_loss=%f percent of loss" %(reg_loss, orth_loss, cross_entropy), 10*"#")
                for key, value in zip(network.metrics_op_map.keys(), metric_values):
                    print("%s: %f" % (key, value))


            if(epoch % 10 == 0):
                train_saver.save(save_path=save_path, sess=sess)

