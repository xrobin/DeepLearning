[![Build Status](https://travis-ci.org/xrobin/DeepLearning.svg?branch=master)](https://travis-ci.org/xrobin/DeepLearning)
[![Codecov coverage](https://codecov.io/github/xrobin/DeepLearning/branch/master/graphs/badge.svg)](https://codecov.io/github/xrobin/DeepLearning)

DeepLearning
============

R package for deep learning. It is currently experimental and things may change in the future.

Installation
-------

You will need a C++ compiler to install the package from source. 

```R
if (!requireNamespace("devtools")) install.packages("devtools")
devtools::install_github("xrobin/DeepLearning")
```

Getting started
-------

```R
library(DeepLearning)
?DeepLearning
```

Loading the MNIST dataset
-------
```R
devtools::install_github("xrobin/mnist")
library(mnist)
?mnist
data(mnist)
```

Basic usage
-------
```R
#### Initialize a network ####
# Initialize a 784-1000-500-250-30 layers DBN to process the MNIST data set
dbn.mnist <- DeepBeliefNet(Layers(c(784, 1000, 500, 250, 30), input="continuous", output="gaussian"))
print(dbn.mnist)

#### Pre-training ####
# Pre-train this DBN
pretrained.mnist <- pretrain(dbn.mnist, mnist$train$x, 
                      penalization = "l2", lambda=0.0002, epsilon=c(.1, .1, .1, .001), 
                      batchsize = 100, maxiters=100000)

# Load an already pre-trained network
data(pretrained.mnist) 

# Make predictions to 2 dimensions
predictions <- predict(pretrained.mnist, mnist$test$x)

# See how the data is reconstructed
reconstructions <- reconstruct(pretrained.mnist, mnist$test$x)
dim(predictions)

# And test the RMS error
error <- rmse(pretrained.mnist, mnist$test$x)
head(error)

# Plot predictions
plot.mnist(predictions = predictions, reconstructions = reconstructions)
par(family="mono")
legend("bottomleft", legend = sprintf("Mean error = %.3f", mean(error)), bty="n")

#### Fine-tuning ####
# Unrolling the network is the same as c(pretrained.mnist, rev(pretrained.mnist))
unrolled.mnist <- unroll(pretrained.mnist)
print(unrolled.mnist)

# Fine-tune the DBN with backpropagation
trained.mnist <- train(unrolled.mnist, mnist$train$x, maxiters = 2000, batchsize = 1000,
               optim.control = list(maxit = 10))

# Load an already trained network
data(trained.mnist) 

# Make predictions to 2 dimensions
predictions <- predict(trained.mnist, mnist$test$x)
dim(predictions)
# Use reconstruct to pass through the whole unrolled network
reconstructions <- reconstruct(trained.mnist, mnist$test$x)
dim(reconstructions)

# test the RMS error
error <- rmse(trained.mnist, mnist$test$x)
head(error)

# Plot predictions
plot.mnist(predictions = predictions, reconstructions = reconstructions)
par(family="mono")
legend("bottomleft", legend = sprintf("Mean error = %.3f", mean(error)), bty="n")
