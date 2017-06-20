context("clone")


test_that("pre-test: assign of an RBM keeps the structures linked", {
	rbm1 <- pretrained.mnist[[1]]
	rbm2 <- rbm1
	
	# Make sure rbm1 and 2 are linked with simple assignment
	expect_true(identical(rbm2$weights.env, rbm1$weights.env))
	
	# High-level assignment breaks the link...
	# initial.weight <- rbm2$W[1,1]
	# rbm2$W[1,1] <- 100
	# expect_identical(rbm2$W, rbm1$W)
	# rbm1$W[1,1]
	# rbm2$W[1,1]
	
	# Use low-level direct assignment
	initial.weight <- rbm2$weights.env$weights[1]
	rbm2$weights.env$weights[1] <- 100
	expect_identical(rbm1$weights.env$weights[1], 100)
})


test_that("pre-test: assign of a DBN keeps the structures linked", {
	dbn1 <- clone(trained.mnist) # Make a clone copy already to not pollute trained.mnist and fail other tests
	dbn2 <- dbn1
	
	# Make sure rbm1 and 2 are linked with simple assignment
	expect_true(identical(dbn1$weights.env, dbn2$weights.env))

	# Use low-level direct assignment
	initial.weight <- dbn2$weights.env$weights[1]
	dbn2$weights.env$weights[1] <- 100
	expect_identical(dbn1$weights.env$weights[1], 100)
})


test_that("Clone makes a copy of an RBM", {
	rbm1 <- pretrained.mnist[[1]]
	rbm2 <- clone(rbm1)
	
	# Make sure rbm1 and 2 are linked with simple assignment
	expect_false(identical(rbm1$weights.env, rbm2$weights.env))
	
	# Use low-level direct assignment
	initial.weight <- rbm2$weights.env$weights[1]
	rbm2$weights.env$weights[1] <- 100
	expect_identical(rbm1$weights.env$weights[1], initial.weight)
})



test_that("Clone makes a copy of a DBN", {
	dbn1 <- trained.mnist
	dbn2 <- clone(dbn1)
	
	# Make sure rbm1 and 2 are linked with simple assignment
	expect_false(identical(dbn1$weights.env, dbn2$weights.env))
	
	# Use low-level direct assignment
	initial.weight <- dbn2$weights.env$weights[1]
	dbn2$weights.env$weights[1] <- 100
	expect_identical(dbn1$weights.env$weights[1], initial.weight)
})
