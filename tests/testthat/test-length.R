context("Length")

l3g <- Layer(3, "gaussian")
l4b <- Layer(4, "binary")
l2c <- Layer(2, "continuous")

dbn <- c(l3g, l4b, l2c) # Tested in c.R
unrolled.dbn <- unroll(dbn) # Tested in unroll.R

test_that("Length works as expected", {
	expect_that(length(dbn), equals(3))
	expect_that(length(unrolled.dbn), equals(5))
})