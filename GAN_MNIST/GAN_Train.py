# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:37:18 2018

@author: jbk48
"""

import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
os.chdir("C:\\Users\\jbk48\\OneDrive\\바탕 화면\\tf_model")
import GAN_MNIST


learning_rate = 0.0002
training_epochs = 100
batch_size = 100
display_step = 1

mnist = input_data.read_data_sets("./MNIST_DATA", one_hot=True)
train_X = mnist.train.images

X = tf.placeholder(tf.float32, [None, 784])   # 28 x 28 image
Z = tf.placeholder(tf.float32, [None, 128])  ## Random noise

gan = GAN_MNIST.GAN()
D_loss,G_loss = gan.model_build(X,Z)

total_var = tf.trainable_variables()
D_var_list = [var for var in total_var if 'Discriminator' in var.name]
G_var_list = [var for var in total_var if 'Generator' in var.name]

with tf.variable_scope("loss", reuse = tf.AUTO_REUSE):

    D_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-D_loss,var_list = D_var_list)
    G_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-G_loss,var_list = G_var_list)

# Initializing the variables
init = tf.global_variables_initializer()

modelName = "C:\\Users\\jbk48\\OneDrive\\바탕 화면\\tf_model\\GAN_model.ckpt"
saver = tf.train.Saver()

print("start training")

with tf.Session() as sess:

    start_time = time.time()
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        dl, gl = 0., 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        # Fit the line.
        for step in range(total_batch):

            batch_x, _ = mnist.train.next_batch(batch_size)
            batch_z = gan.Gaussian_noise(batch_size)
            # Fit training using batch data
            sess.run(D_optimizer, feed_dict={X: batch_x, Z: batch_z})
            sess.run(G_optimizer, feed_dict={Z: batch_z})
            dl += sess.run(D_loss, feed_dict={X: batch_x, Z: batch_z})/total_batch
            gl += sess.run(G_loss, feed_dict={Z: batch_z})/total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("========= Epoch : {} ==========".format(epoch+1))
            print("D_cost = {:.9f}".format(dl))
            print("G_cost = {:.9f}".format(gl))
            Generate_img = sess.run(gan.Generator(Z), feed_dict = {Z: gan.Gaussian_noise(10)})
            fig, ax = plt.subplots(1, 10, figsize = (10,1))
            for i in range(10):
                ax[i].set_axis_off()
                ax[i].imshow(np.reshape(Generate_img[i], (28,28)))
            if epoch % 10 == 0:
                plt.savefig("GAN_MNIST_epoch{}.png".format(epoch+1), bbox_inches='tight')
                plt.close(fig)


    print("Optimization Finished!")
    duration = time.time() - start_time
    minute = int(duration/60)
    second = int(duration)%60
    print("%dminutes %dseconds" % (minute,second))
    save_path = saver.save(sess, modelName)
    print ('save_path',save_path)
