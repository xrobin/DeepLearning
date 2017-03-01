library(testthat)
library(DeepLearning)
library(mnist)
test.dat <- mnist$test$x[1:10,]

test_check("DeepLearning")