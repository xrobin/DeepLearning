context("DBN initialization")

l3c <- Layer(3, "c")
l4b <- Layer(4, "b")
l2g <- Layer(2, "g")

b1 <- c(-3.4, 0.8, 3.0)
W1 <- c(16, 0.14, -0.3, 0.8,
		0.03, -0.02, -0.3, 0.25,
		0.01, 0.3, 0.6, -0.3)
W1m <- matrix(W1, nrow = 4)
c1 <- c(1.4, 0.2, 0.3, -0.1)
b2 <- c1
W2 <- c(-1.2, -3.1,
		3.3, -2.4,
		-1.3, 0.7,
		-0.5, 0.8)
W2m <- matrix(W2, nrow = 2)
c2 <- c(2.4, -3.2)

weights <- c(b1, W1, c1, W2, c2)

test_that("DBN initialization with 0", {
	# By default, DBN has 0 weights
	dbn <- DeepBeliefNet(l3c, l4b, l2g)
	expect_true(all(dbn$weights.env$weights == 0))
	expect_equal(sum(abs(dbn$weights.env$weights)), 0)
	
	# Can initialize a DBN with 0 weights explicitly
	dbn <- DeepBeliefNet(l3c, l4b, l2g, initialize = "0")
	expect_true(all(dbn$weights.env$weights == 0))
	expect_equal(sum(abs(dbn$weights.env$weights)), 0)
})

test_that("Can assign weights to DBN through RBMs", {
	dbn <- DeepBeliefNet(l3c, l4b, l2g)
	# Assign weights to RBM
	dbn[[1]]$b <- b1
	dbn[[1]]$W <- W1m
	dbn[[1]]$c <- c1
	dbn[[2]]$W <- W2m
	dbn[[2]]$c <- c2
	expect_identical(dbn$weights.env$weights, weights)
	
	# Also with initialize = "0"
	dbn <- DeepBeliefNet(l3c, l4b, l2g, initialize = "0")
	# Assign weights to RBM
	dbn[[1]]$b <- b1
	dbn[[1]]$W <- W1m
	dbn[[1]]$c <- c1
	dbn[[2]]$W <- W2m
	dbn[[2]]$c <- c2
	expect_identical(dbn$weights.env$weights, weights)
})


#test_that("DBN initialization with given weights", {
# Skip: not implemented
#	dbn <- DeepBeliefNet(l3c, l4b, l2g, weights = weights)
#})

test_that("DBN initialization with random uniform weights", {
	dbn <- DeepBeliefNet(l3c, l4b, l2g, initialize = "uniform")
	expect_gt(sum(abs(dbn$weights.env$weights)), 0)
	
	# Expect that the rbms also have the weights set
	expect_gt(sum(abs(dbn[[1]]$b)), 0)
	expect_gt(sum(abs(dbn[[1]]$W)), 0)
	expect_gt(sum(abs(dbn[[1]]$c)), 0)
	expect_gt(sum(abs(dbn[[2]]$b)), 0)
	expect_gt(sum(abs(dbn[[2]]$W)), 0)
	expect_gt(sum(abs(dbn[[2]]$c)), 0)
})

test_that("DBN initialization with invalid initialize argument fails", {
	expect_error(DeepBeliefNet(l3c, l4b, l2g, initialize = "invalid"))
})
