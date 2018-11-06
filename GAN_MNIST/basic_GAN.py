# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 09:43:43 2018

@author: jbk48
"""

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

import matplotlib.pyplot as plt


import numpy as np

class GAN():
    
    def __init__(self):
        
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the generator
        self.G = self.build_generator()
        self.G.compile(loss='binary_crossentropy',
            optimizer=optimizer)

        # Build and compile the discriminator
        self.D = self.build_discriminator()
        self.D.compile(loss='binary_crossentropy',
            optimizer=optimizer)

        # Build and compile the G_D
        self.D.trainable = False  ## D를 학습을 안시킨다
        self.G_D = Sequential()
        self.G_D.add(self.G)
        self.G_D.add(self.D)
        print("stacked G_D")
        self.G_D.summary()
        self.G_D.compile(loss='binary_crossentropy',
            optimizer=optimizer)



    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        print("Generator")
        model.summary()

        return model

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        print("Discriminator")
        model.summary()

        return model
    

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        real = np.ones((batch_size, 1))   ## batch_size만큼 1로
        fake = np.zeros((batch_size, 1))  ## batch_size만큼 0으로

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            fake_imgs = self.G.predict(noise)

            # Train the discriminator
            d_loss_real = self.D.train_on_batch(real_imgs, real)  ## 진짜는 진짜로 구분 --> 1
            d_loss_fake = self.D.train_on_batch(fake_imgs, fake)  ## 가짜는 가짜로 구분 --> 0
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as real)
            g_loss = self.G_D.train_on_batch(noise, real)

            # Plot the progress
            print ("{} [D loss: {}] [G loss: {}]".format(epoch+1, d_loss, g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.G.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./images/%d.png" % epoch)
        plt.close()
