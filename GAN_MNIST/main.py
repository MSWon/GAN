# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 10:31:43 2018

@author: jbk48
"""

import basic_GAN

gan = basic_GAN.GAN()
gan.train(epochs=30000, batch_size=32, sample_interval=200)
