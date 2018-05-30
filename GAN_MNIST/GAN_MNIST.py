# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:28:01 2018

@author: jbk48
"""

import tensorflow as tf
import numpy as np

class GAN():

    def __init__(self, batch_size = 100):

        with tf.variable_scope('Generator', reuse = tf.AUTO_REUSE):

            self.G_W1 = tf.get_variable("G_W1", shape = [128,256], initializer = tf.random_normal_initializer(0.0, 0.01))
            self.G_b1 = tf.Variable(tf.zeros([256]))
            self.G_W2 = tf.get_variable("G_W2", shape = [256,784], initializer = tf.random_normal_initializer(0.0, 0.01))
            self.G_b2 = tf.Variable(tf.zeros([784]))

        with tf.variable_scope('Discriminator', reuse = tf.AUTO_REUSE):

            self.D_W1 = tf.get_variable("D_W1", shape = [784,256], initializer = tf.random_normal_initializer(0.0, 0.01))
            self.D_b1 = tf.Variable(tf.zeros([256]))
            self.D_W2 = tf.get_variable("D_W2", shape = [256,1], initializer = tf.random_normal_initializer(0.0, 0.01))
            self.D_b2 = tf.Variable(tf.zeros([1]))

    def Generator(self, inputs):

        G_L1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
        G_L2 = tf.nn.relu(tf.matmul(G_L1, self.G_W2) + self.G_b2)
        return G_L2


    def Gaussian_noise(self, batch_size):

        return np.random.normal(size = [batch_size, 128])


    def Discriminator(self, inputs):

        with tf.variable_scope('Discriminator', reuse = tf.AUTO_REUSE):

            D_L1 = tf.nn.relu(tf.matmul(inputs,self.D_W1) + self.D_b1)
            D_L2 = tf.nn.sigmoid(tf.matmul(D_L1,self.D_W2) + self.D_b2)

        return D_L2

    def model_build(self,X,Z):

        self.D_loss = tf.reduce_mean(tf.log(self.Discriminator(X))+tf.log(1-self.Discriminator(self.Generator(Z))))
        self.G_loss = tf.reduce_mean(tf.log(self.Discriminator(self.Generator(Z))))
        tf.summary('D_loss', self.D_loss)
        tf.summary('G_loss', self.G_loss)
        merged = tf.summary.merge_all()
        
        return self.D_loss,self.G_loss,merged
