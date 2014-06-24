context("Layers")

test_that("Can create layers", {
	l4g <- Layer(4, "gaussian")
	l4b <- Layer(4, "binary")
	l4c <- Layer(4, "continuous")
	
	# We created Layers
	expect_that(l4g, is_a("Layer"))
	expect_that(l4b, is_a("Layer"))
	expect_that(l4c, is_a("Layer"))

	# Abbreviated layer types
	expect_that(l4g, is_identical_to(Layer(4, "g")))
	expect_that(l4c, is_identical_to(Layer(4, "c")))
	expect_that(l4b, is_identical_to(Layer(4, "b")))
	
	# Size makes a difference
	expect_that(identical(l4g, Layer(10, "gaussian")), is_false())
	expect_that(identical(l4b, Layer(10, "binary")), is_false())
	expect_that(identical(l4c, Layer(10, "continuous")), is_false())
	
	# And so does type
	expect_that(identical(l4g, l4b), is_false())
	expect_that(identical(l4g, l4c), is_false())
	expect_that(identical(l4c, l4b), is_false())
})
	
test_that("Cannot create invalid layers", {
	# invalid type
	expect_that(Layer(4, "invalid"), throws_error())
	expect_that(Layer(4, "i"), throws_error())
	
	#invalid lengths
	expect_that(Layer(0, "gaussian"), throws_error())
	expect_that(Layer(-4, "gaussian"), throws_error())
	expect_that(Layer(c(4, 10), "gaussian"), throws_error())
	
	
	expect_that(Layer(4, "continuous"), is_identical_to(Layer(4, "c")))
	expect_that(Layer(4, "binary"), is_identical_to(Layer(4, "b")))
})


test_that("Layers() works", {
	# Works at all
	l4g10b5b2c <- Layers(c(4, 10, 5, 2), "gaussian", "continuous")
	expect_that(l4g10b5b2c, is_a("list"))
	
	# Of the right length
	expect_that(length(l4g10b5b2c), equals(4))
	
	# And contains the right layers
	l4g <- Layer(4, "gaussian")
	l10b <- Layer(10, "binary")
	l5b <- Layer(5, "binary")
	l2c <- Layer(2, "continuous")
	expect_that(l4g10b5b2c[[1]], is_identical_to(l4g))
	expect_that(l4g10b5b2c[[2]], is_identical_to(l10b))
	expect_that(l4g10b5b2c[[3]], is_identical_to(l5b))
	expect_that(l4g10b5b2c[[4]], is_identical_to(l2c))
	
	# Also with a different set of layers
	l8b20c10c5c1g <- Layers(c(8, 20, 10, 5, 1), "binary", "gaussian", "continuous")
	l8b <- Layer(8, "binary")
	l20c <- Layer(20, "continuous")
	l10c <- Layer(10, "continuous")
	l5c <- Layer(5, "continuous")
	l1g <- Layer(1, "gaussian")
	expect_that(length(l8b20c10c5c1g), equals(5))
	expect_that(l8b20c10c5c1g[[1]], is_identical_to(l8b))
	expect_that(l8b20c10c5c1g[[2]], is_identical_to(l20c))
	expect_that(l8b20c10c5c1g[[3]], is_identical_to(l10c))
	expect_that(l8b20c10c5c1g[[4]], is_identical_to(l5c))
	expect_that(l8b20c10c5c1g[[5]], is_identical_to(l1g))
})