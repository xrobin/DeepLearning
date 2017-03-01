context("error")

test.dat <- mnist$test$x[1:10,]

test_that("error.dbn works as expected", {
	err <- error(trained.mnist, test.dat)
	expected_error <- c(0.155460830388072, 0.263534247331889, 0.0685053807177542, 0.185973450892959, 
						0.168769961889794, 0.0858605877640525, 0.252296270157863, 0.21549470185196, 
						0.29723905773599, 0.173839543887414)
	expect_equal(err, expected_error) # somehow expected_error is a bit imprecise
	expect_identical(err, rmse(trained.mnist, test.dat))
	expect_equal(sum(err), errorSum(trained.mnist, test.dat)) # might have rounding -> equal
	
	# Works with 1 row
	expect_identical(error(trained.mnist, test.dat[1,, drop = FALSE]), err[1])
})


test_that("error.rbm works as expected", {
	rbm <- pretrained.mnist[[1]]
	err <- error(rbm, test.dat)
	expected_error <- c(0.165268639716517, 0.179716464245748, 0.15527685653651, 0.123360496570759, 
						0.181857971990638, 0.129981198058071, 0.184642068390986, 0.193730170753474, 
						0.133422329315982, 0.138088564679852)
	expect_equal(err, expected_error) # somehow expected_error is a bit imprecise
	expect_identical(err, rmse(rbm, test.dat))
	expect_equal(sum(err), errorSum(rbm, test.dat)) # might have rounding -> equal
	
	# Works with 1 row
	expect_identical(error(rbm, test.dat[1,, drop = FALSE]), err[1])
})


test_that("error.dbn errors if passed invalid data", {
	# Don't accept a vector
	expect_error(error(trained.mnist, test.dat[1,, drop = TRUE]))
	expect_error(error(pretrained.mnist[[1]], test.dat[1,, drop = TRUE]))
	
	# Don't accept wrong dimensions
	expect_error(error(trained.mnist, test.dat[, 1:20, drop = FALSE]), regexp = "column")
	expect_error(error(trained.mnist[[1]], test.dat[, 1:20, drop = FALSE]), regexp = "column")
})
