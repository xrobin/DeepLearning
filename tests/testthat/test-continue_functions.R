context("Continue Functions")

# Create a DBN
dbn <- DeepBeliefNet(Layer(3, "c"), Layer(4, "b"), Layer(2, "g"))
weights <- c(
	c(-3.4, 0.8, 3.0), # b1
	# Eigen will represent this as a 4x3 matrix, in column-major:
	c(16, 0.14, -0.3, 0.8,
	  0.03, -0.02, -0.3, 0.25,
	  0.01, 0.3, 0.6, -0.3), # W1
	c(1.4, 0.2, 0.3, -0.1), # c1 = b2
	c(-1.2, -3.1,
	  3.3, -2.4,
	  -1.3, 0.7,
	  -0.5, 0.8), # W2
	c(2.4, -3.2) #c2
)
assign("weights", weights, dbn$weights.env)

# And an input vector
f <- t(c(0, .5, 1))


test_that("Custom continue function is called as many times as expected with train and an RBM", {
	keepiters.frame <- sys.frame()
	assign("keepiters", c(), envir = keepiters.frame)
	testf <- function(error, iter, batchsize, maxiters, layer) {
		assign("keepiters", c(get("keepiters", envir = keepiters.frame), iter), envir = keepiters.frame)
		print(get("keepiters", envir = keepiters.frame))
		#browser()
		return(TRUE)
	}
	rbm <- dbn[[1]]
	# Frequency 1, 3:7
	trained <- pretrain(rbm, f, batchsize=1, miniters = 3, maxiters = 7, continue.function = testf, continue.function.frequency = 1)
	expect_equal(keepiters, 3:7)
	
	#Frequency 2, 5:13
	assign("keepiters", c(), envir = keepiters.frame)
	trained <- pretrain(rbm, f, batchsize=1, miniters = 4, maxiters = 12, continue.function = testf, continue.function.frequency = 2)
	expect_equal(keepiters, c(4, 6, 8, 10, 12))
	
	# This one is run exactly once
	assign("keepiters", c(), envir = keepiters.frame)
	trained <- pretrain(rbm, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = testf, continue.function.frequency = 1)
	expect_equal(keepiters, c(1))
	
	# Function that returns FALSE when iter > 10
	testf <- function(error, iter, batchsize, maxiters, layer) {
		assign("keepiters", c(get("keepiters", envir = keepiters.frame), iter), envir = keepiters.frame)
		print(get("keepiters", envir = keepiters.frame))
		#browser()
		return(iter <= 10)
	}
	
	#Frequency 3, exit immediately
	assign("keepiters", c(), envir = keepiters.frame)
	trained <- pretrain(rbm, f, batchsize=1, miniters = 3, maxiters = 20, continue.function = testf, continue.function.frequency = 3, continue.stop.limit = 1)
	expect_equal(keepiters, c(3, 6, 9, 12))
	
	#Frequency 3, keep going for 4 times
	assign("keepiters", c(), envir = keepiters.frame)
	trained <- pretrain(rbm, f, batchsize=1, miniters = 3, maxiters = 30, continue.function = testf, continue.function.frequency = 3, continue.stop.limit = 4)
	expect_equal(keepiters, c(3, 6, 9, 12, 15, 18, 21))
})

test_that("Custom continue function is called as many times as expected with pretrain and a DBN", {
	keepiters.frame <- sys.frame()
	assign("keepiters", c(), envir = keepiters.frame)
	testf <- function(error, iter, batchsize, maxiters, layer) {
		assign("keepiters", c(get("keepiters", envir = keepiters.frame), iter), envir = keepiters.frame)
		print(get("keepiters", envir = keepiters.frame))
		#browser()
		return(TRUE)
	}
	# Frequency 1, 3:7
	trained <- pretrain(dbn, f, batchsize=1, miniters = 3, maxiters = 7, continue.function = testf, continue.function.frequency = 1)
	expect_equal(keepiters, rep(3:7, 2))
	
	#Frequency 2, 5:13
	assign("keepiters", c(), envir = keepiters.frame)
	trained <- pretrain(dbn, f, batchsize=1, miniters = 4, maxiters = 12, continue.function = testf, continue.function.frequency = 2)
	expect_equal(keepiters, rep(c(4, 6, 8, 10, 12), 2))
	
	# This one is run exactly once
	assign("keepiters", c(), envir = keepiters.frame)
	trained <- pretrain(dbn, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = testf, continue.function.frequency = 1)
	expect_equal(keepiters, c(1, 1))
	
	# Function that returns FALSE when iter > 10
	testf <- function(error, iter, batchsize, maxiters, layer) {
		assign("keepiters", c(get("keepiters", envir = keepiters.frame), iter), envir = keepiters.frame)
		print(get("keepiters", envir = keepiters.frame))
		#browser()
		return(iter <= 10)
	}
	
	#Frequency 3, exit immediately
	assign("keepiters", c(), envir = keepiters.frame)
	trained <- pretrain(dbn, f, batchsize=1, miniters = 3, maxiters = 20, continue.function = testf, continue.function.frequency = 3, continue.stop.limit = 1)
	expect_equal(keepiters, rep(c(3, 6, 9, 12), 2))
	
	#Frequency 3, keep going for 4 times
	assign("keepiters", c(), envir = keepiters.frame)
	trained <- pretrain(dbn, f, batchsize=1, miniters = 3, maxiters = 30, continue.function = testf, continue.function.frequency = 3, continue.stop.limit = 4)
	expect_equal(keepiters, rep(c(3, 6, 9, 12, 15, 18, 21), 2))
})

test_that("Custom continue function is called as many times as expected with train", {
	keepiters.frame <- sys.frame()
	assign("keepiters", c(), envir = keepiters.frame)
	testf <- function(error, iter, batchsize, maxiters, layer) {
		assign("keepiters", c(get("keepiters", envir = keepiters.frame), iter), envir = keepiters.frame)
		print(get("keepiters", envir = keepiters.frame))
		#browser()
		return(TRUE)
	}
	unrolled <- unroll(dbn)
	# Frequency 1, 3:7
	trained <- train(unrolled, f, batchsize=1, miniters = 3, maxiters = 7, continue.function = testf, continue.function.frequency = 1)
	expect_equal(keepiters, 3:7)
	
	#Frequency 2, 5:13
	assign("keepiters", c(), envir = keepiters.frame)
	trained <- train(unrolled, f, batchsize=1, miniters = 4, maxiters = 12, continue.function = testf, continue.function.frequency = 2)
	expect_equal(keepiters, c(4, 6, 8, 10, 12))
	
	# This one is run exactly once
	assign("keepiters", c(), envir = keepiters.frame)
	trained <- train(unrolled, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = testf, continue.function.frequency = 1)
	expect_equal(keepiters, c(1))
	
	# Function that returns FALSE when iter > 10
	testf <- function(error, iter, batchsize, maxiters, layer) {
		assign("keepiters", c(get("keepiters", envir = keepiters.frame), iter), envir = keepiters.frame)
		print(get("keepiters", envir = keepiters.frame))
		#browser()
		return(iter <= 10)
	}
	
	#Frequency 3, exit immediately
	assign("keepiters", c(), envir = keepiters.frame)
	trained <- train(unrolled, f, batchsize=1, miniters = 3, maxiters = 20, continue.function = testf, continue.function.frequency = 3, continue.stop.limit = 1)
	expect_equal(keepiters, c(3, 6, 9, 12))
	
	#Frequency 3, keep going for 4 times
	assign("keepiters", c(), envir = keepiters.frame)
	trained <- train(unrolled, f, batchsize=1, miniters = 3, maxiters = 30, continue.function = testf, continue.function.frequency = 3, continue.stop.limit = 4)
	expect_equal(keepiters, c(3, 6, 9, 12, 15, 18, 21))
})

test_that("Continue functions run at all with pretrain and an RBM", {
	rbm <- dbn[[1]]
	trained <- pretrain(rbm, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = continue.function.always, continue.function.frequency = 1)
	expect_false(is(trained, "try-error"))
	trained <- pretrain(rbm, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = continue.function.exponential, continue.function.frequency = 1)
	expect_false(is(trained, "try-error"))
	trained <- pretrain(rbm, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = continue.function.exponential.aic, continue.function.frequency = 1)
	expect_false(is(trained, "try-error"))
	trained <- pretrain(rbm, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = continue.function.exponential.bic, continue.function.frequency = 1)
	expect_false(is(trained, "try-error"))
	trained <- pretrain(rbm, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = continue.function.random, continue.function.frequency = 1)
	expect_false(is(trained, "try-error"))
	1
})

test_that("Continue functions run at all with pretrain and a DBN", {
	trained <- pretrain(dbn, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = continue.function.always, continue.function.frequency = 1)
	expect_false(is(trained, "try-error"))
	trained <- pretrain(dbn, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = continue.function.exponential, continue.function.frequency = 1)
	expect_false(is(trained, "try-error"))
	trained <- pretrain(dbn, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = continue.function.exponential.aic, continue.function.frequency = 1)
	expect_false(is(trained, "try-error"))
	trained <- pretrain(dbn, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = continue.function.exponential.bic, continue.function.frequency = 1)
	expect_false(is(trained, "try-error"))
	trained <- pretrain(dbn, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = continue.function.random, continue.function.frequency = 1)
	expect_false(is(trained, "try-error"))
	1
})
	
test_that("Continue functions run at all with train", {
	unrolled <- unroll(dbn)
	trained <- train(unrolled, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = continue.function.always, continue.function.frequency = 1)
	expect_false(is(trained, "try-error"))
	trained <- train(unrolled, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = continue.function.exponential, continue.function.frequency = 1)
	expect_false(is(trained, "try-error"))
	trained <- train(unrolled, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = continue.function.exponential.aic, continue.function.frequency = 1)
	expect_false(is(trained, "try-error"))
	trained <- train(unrolled, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = continue.function.exponential.bic, continue.function.frequency = 1)
	expect_false(is(trained, "try-error"))
	trained <- train(unrolled, f, batchsize=1, miniters = 1, maxiters = 1, continue.function = continue.function.random, continue.function.frequency = 1)
	expect_false(is(trained, "try-error"))
	1
})
	
	
