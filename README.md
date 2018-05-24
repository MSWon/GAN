# Basic GAN with Tensorflow
Generative Adversarial Nets with MNIST data using Tensorflow

## 1. Generative Adversarial Nets
![alt text](https://github.com/MSWon/GAN/blob/master/pic/pic_1.PNG "Model")

- According to **Ian J. Goodfellow** this framework corresponds to a minimax two-player game.

- **G(Generator)** and **D(Discriminator)** are definedby multilayer perceptrons, the entire system can be trained with backpropagation.

- The goal is to train **G(Generator)** generate fake data that can deceive **D(Discriminator)** and to train **D(Discriminator)** discriminate between fake and real.

## 2. Loss function

### - Discriminator

![alt text](https://github.com/MSWon/GAN/blob/master/pic/equation_1.PNG "Model")

- Update the **discriminator** by ascending its stochastic gradient

### - Generator

![alt text](https://github.com/MSWon/GAN/blob/master/pic/equation_2.PNG "Model")

- Update the **generator** by descending its stochastic gradient
