import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0')


tf.reset_default_graph()

"""
Implement the network
"""

inputs1 = tf.placeholder(shape=[1,16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

nextQ = tf.placeholder(shape=[1,4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ-Qout))

trainer = tf.train.GradientDescentOptimizer(learning_rate=.1)

trainStep = trainer.minimize(loss)

jList = []
rList = []

y = .99
e = .1
num_episodes = 2000

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        s = env.reset()
        rAll = 0
        d = False
        j = 0

        #The Q-Network
        for j in range(100):

            #Supply one hot encoding vector to network (current state)
            feed_dict = {inputs1: np.identity(16)[s:s+1]}

            #Predict action and reward for action
            a, allQ = sess.run([predict, Qout], feed_dict=feed_dict)

            #With certain probability use random action
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()

            s1, r, d, _ = env.step(a[0])

            Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[s:s+1]})

            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = r + y*maxQ1

            _, W1 = sess.run([trainStep, W], feed_dict={inputs1: np.identity(16)[s:s+1], nextQ: targetQ})

            rAll += r
            s = s1
            if d == True:
                e = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
print("Successful episodes: ", str(sum(rList)*100/num_episodes), "%")

