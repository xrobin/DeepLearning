library(testthat)
library(DeepLearning)
library(mnist)
data(mnist)
test.dat <- mnist$test$x[1:10,]

# Set environment variable RUN_SLOW_TESTS=true to run the slower tests
run_slow_tests <- identical(Sys.getenv("RUN_SLOW_TESTS"), "true")

test_check("DeepLearning")