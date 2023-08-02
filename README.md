# Deep Learning Specialization
This specialization is composed of 5 independent courses.

In this document, for each course, I summarize what I learned, how it enhanced my data scientist skills and what programming assignment consists to.

This GitHub repository contains all the coding material that I wrote and studied for this specialization.

## 01 - [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/home/week/1)

### How did this course enhance my Data Science skills?
- I know how a neural network is composed, both a shallow or a deep one. Moreover, I understand its mathematical representation.
- I can build a deep neural network from scratch in Python and implementing vectorization techniques.
- I understand the most common activation functions (ReLu, hyperbolic tangent, Softmax, Sigmoid, Leaky ReLu) and how to choose the best for each layer of my network. Moreover, I understand why we need non-linear activation functions in a nerual network.
- I am familiar with the random initialization concept, I know why it's done and how to implement it.
- I deeply understood the mathematical foundation of backpropagation, which is the reason why we are able to build functioning networks.

### What does this course deal?
- Supervised learning with Neural Networks
- Logistic Regression as a Neural Network
	- Logistic Regression **cost function**
	- **Gradient descent**
	- Derivatives
	- Gradient descent on $m$ examples
- **Vectorization**
	- Vectorization of Logistic Regression
	- Vectorization in Python
	- **Broadcasting**
- **Shallow Neural Networks**
	- Neural Network overview
	- Vectorized Neural Network implementation
	- **Activation functions**
	- Gradient descent for neural networks
	- **Backpropagation**
	- Random initialization
- **Deep Neural Networks**
	- **Forward propagation** in deep networks
	- Motivation for deep networks
	- **Backpropagation** in deep networks
	- **Parameters** vs. **Hyperparameters**

### Coding assignments
- **Building a Logistic Regression Neural Network from scratch**
- **Building a Classification Neural Network with one hidden layer**
- **Building a Deep Neural Network from scratch**

---
## 02 - [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network/home/welcome)

### How did this course enhance my Data Science skills?
- I know the importance of splitting the dataset and the correct proportions for train/dev/test sets, depending on the amount of available data
- I know how to detect an overfitting and underfitting model. Moreover, I know how to deal with an overfitting model by implementing different regularization techniques
- I understand the importance of normalizing the input features and of applying batch normalization to each deep network's layer
- I understand the mathematics behind the most used optimization methods (Mini-batch gradient descent, Momentum, RMSprop, Adam), and I know the effect of each hyperparameter
- I know how to perform the hyperparameter tuning process, and how to choose an appropriate scale for each hyperparameter
- I deeply understand the mathematics foundations of Softmax Regression
- I am familiar with the TensorFlow environment and with the `tf.Tensor` objects

### What does this course deal?
- Setting up a Machine Learning Application
	- **Train/development/test set** splitting
	- Detecting **high bias/variance**
- Neural Network Regularization
	- **L2 Regularization**
	- **Dropout Regularization**
- Neural Network optimization problem
	- **Inputs normalization**
	- **Exploding / vanishing gradients**
	- **Weight initialization** for deep networks
	- **Numerical approximation** of the gradients
	- **Gradient Checking**
- Optimization algorithms
	- **Mini-batch** gradient descent
	- **Exponentially Weighted Averages**
	- Gradient descent with **momentum**
	- **RMSprop**
	- **Adam**
	- **Learning rate decay**
- Hyperparameters tuning
	- Picking **hyperparameters' scale**
	- Tuning methods: **pandas vs. caviar**
- Batch Normalization
- Multi-class classification
	- **Softmax** regression
- Programming frameworks
	- **TensorFlow**

### Coding assignments
- **Initialization**: comparing and optimizing weights initialization (random, zeros, He initialization)
- **Regularization**: implementation of regularization techniques to reduce overfitting
- **Gradient Checking**: implementation of gradient checking to verify the accuracy of backpropagation in a fraud detection model
- **Optimization Methods**: implementing and comparing different optimization methods (Stochastic Gradient Descent, Momentum, RMSprop, Adam
- **TensorFlow Introduction**: building a neural network with low level TensorFlow implementation

---
## 04 - [Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning)

### How did this course enhance my Data Science skills?
This course allowed me to dive into CNNs and their most popular applications. I managed to learn how they work and how they perform complex tasks like Object Localization, Semantic Segmentation, Face Recognition and Neural Style Transfer.

### What does this course deal?
- Foundations of CNNs
	- **Edge** detection
	- **Padding**
	- **Striding**
	- Convolutions over volume
	- **Pooling** layers
- Deep Convolutional Models
	- Residual Networks
	- 1x1 convolutions
	- Inception network
	- MobileNet
	- EfficientNet
- Transfer Learning
- Data Augmentation
- Object Localization and Object Detection
- Sliding Windows and YOLO algorithms
- Semantic Segmentation with U-Net
- Face Recognition
- Neural Style Transfer

### Coding assignments
- **Convolutional model, Step by step**: building a convolutional neural network from scratch
- **Convolution Model Application**: building a mood classifier and a sign language digits identifier
- **Residual Networks**: building a Residual Network from scratch
- **Transfer Learning**: applying a transfer learning technique to built an alpaca/not alpaca classifier from a pre-trained MobileNet
- **Car detection with YOLO algorithm**: detecting cars in images using the popular YOLO algorithm
- **Image Segmentation with U-Net**: applying Image Segmentation to images to classify each pixel
- **Face Recognition**: building a simple face recognition system
- **Art Generation with Neural Style Transfer**: merging the content and the style of two images using NST

---
