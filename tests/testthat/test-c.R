context("Concatenation")

l4g <- Layer(4, "gaussian")
l4b <- Layer(4, "binary")
l4c <- Layer(4, "continuous")

l3g <- Layer(3, "gaussian")
l3b <- Layer(3, "binary")
l3c <- Layer(3, "continuous")


rbm1 <- RestrictedBolzmannMachine(l4g, l3g)
rbm2 <- RestrictedBolzmannMachine(l3g, l4b)
rbm3 <- RestrictedBolzmannMachine(l4b, l3c)
dbn1 <- DeepBeliefNet(l4g, l3g)
dbn2 <- DeepBeliefNet(l3g, l4b)
dbn3 <- DeepBeliefNet(l4b, l3c)

test_that("concatenation works for layers", {
	r <- c(l4g, l3g, l4b, l3c)
	expect_is(r, "DeepBeliefNet")
	expect_identical(length(r), 4L)
})

test_that("concatenation works for RBMs", {
	r <- c(rbm1, rbm2, rbm3)
	expect_is(r, "DeepBeliefNet")
	expect_identical(length(r), 4L)
})

test_that("concatenation works for DBNs", {
	r <- c(dbn1, dbn2, dbn3)
	expect_is(r, "DeepBeliefNet")
	expect_identical(length(r), 4L)
})

test_that("concatenation works for mixed types", {
	r <- c(rbm1, dbn2, l3c)
	expect_equal(r, DeepBeliefNet(l4g, l3g, l4b, l3c))
	# In different orders
	r <- c(dbn1, rbm2, l3c)
	expect_equal(r, DeepBeliefNet(l4g, l3g, l4b, l3c))
	r <- c(l3c, dbn1, rbm2)
	expect_equal(r, DeepBeliefNet(l3c, l4g, l3g, l4b))
})


test_that("cannot concatenate incompatible sizes of RBMs, DBNs and Layers", {
	expect_error(c(rbm1, rbm3))
	expect_error(c(dbn1, dbn3))
	expect_error(c(rbm1, dbn3))
	expect_error(c(rbm1, dbn3))
})


test_that("cannot concatenate non-RBMs, DBNs and Layers", {
	expect_error(c(rbm1, dbn2, l3c, 4))
	expect_error(c(rbm1, dbn2, l3c, "gaussian"))
})

