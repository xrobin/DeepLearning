context("resample")

test_that("resample.dbn works", {
	sample <- resample(trained.mnist, test.dat)
	expect_identical(dim(sample), c(10L, 784L))
})


test_that("resample.rbm works", {
	sample <- resample(pretrained.mnist[[1]], test.dat)
	expect_identical(dim(sample), c(10L, 1000L))
})


test_that("drop works on RBM", {
	sample <- resample(pretrained.mnist[[1]], test.dat[1,, drop = FALSE])
	# With drop
	expect_is(sample, "numeric")
	expect_length(sample, 1000)
	# Without drop
	sample <- resample(pretrained.mnist[[1]], test.dat[1,, drop = FALSE], drop = FALSE)
	expect_is(sample, "matrix")
	expect_identical(dim(sample), c(1L, 1000L))
})



test_that("drop works on DBN", {
	sample <- resample(trained.mnist, test.dat[1,, drop = FALSE])
	# With drop
	expect_is(sample, "numeric")
	expect_length(sample, 784)
	# Without drop
	sample <- resample(trained.mnist, test.dat[1,, drop = FALSE], drop = FALSE)
	expect_is(sample, "matrix")
	expect_identical(dim(sample), c(1L, 784L))
})
