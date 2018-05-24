# Basic GAN with Tensorflow
Generative Adversarial Nets with MNIST data using Tensorflow

## 1. Generative Adversarial Nets
![alt text](https://github.com/MSWon/GAN/blob/master/pic/pic_1.PNG "Model")

- According to **Ian J. Goodfellow** this framework corresponds to a minimax two-player game.

- **G(Generator)** and **D(Discriminator)** are definedby multilayer perceptrons, the entire system can be trained with backpropagation.

- The goal is to train **G(Generator)** generate fake data that can deceive **D(Discriminator)** and to train **D(Discriminator)** discriminate between fake and real.

## 2. Loss function

### - Discriminator

![alt text](https://github.com/MSWon/GAN/blob/master/pic/equation_1.PNG "eq_l")

- Update the **discriminator** by ascending its stochastic gradient

### - Generator

![alt text](https://github.com/MSWon/GAN/blob/master/pic/equation_2.PNG "eq_2")

- Update the **generator** by descending its stochastic gradient

## 3. Result
### Epoch - 1 -
![alt text](https://github.com/MSWon/GAN/blob/master/GAN_MNIST/image/GAN_MNIST_epoch1.png "ep_1")
### Epoch - 11 -
![alt text](https://github.com/MSWon/GAN/blob/master/GAN_MNIST/image/GAN_MNIST_epoch11.png "ep_11")
### Epoch - 21 -
![alt text](https://github.com/MSWon/GAN/blob/master/GAN_MNIST/image/GAN_MNIST_epoch21.png "ep_21")
### Epoch - 31 -
![alt text](https://github.com/MSWon/GAN/blob/master/GAN_MNIST/image/GAN_MNIST_epoch31.png "ep_31")
### Epoch - 41 -
![alt text](https://github.com/MSWon/GAN/blob/master/GAN_MNIST/image/GAN_MNIST_epoch41.png "ep_41")
### Epoch - 51 -
![alt text](https://github.com/MSWon/GAN/blob/master/GAN_MNIST/image/GAN_MNIST_epoch51.png "ep_51")
### Epoch - 61 -
![alt text](https://github.com/MSWon/GAN/blob/master/GAN_MNIST/image/GAN_MNIST_epoch61.png "ep_61")
### Epoch - 71 -
![alt text](https://github.com/MSWon/GAN/blob/master/GAN_MNIST/image/GAN_MNIST_epoch71.png "ep_71")
### Epoch - 81 -
![alt text](https://github.com/MSWon/GAN/blob/master/GAN_MNIST/image/GAN_MNIST_epoch81.png "ep_81")
### Epoch - 91 -
![alt text](https://github.com/MSWon/GAN/blob/master/GAN_MNIST/image/GAN_MNIST_epoch91.png "ep_91")
