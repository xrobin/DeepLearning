context("RestrictedBolzmannMachine")

test_that("Can create RestrictedBolzmannMachine", {
	# Create an RBM
	l4g <- Layer(4, "gaussian")
	l10b <- Layer(10, "binary")
	anRBM <- RestrictedBolzmannMachine(l4g, l10b)
	
	# Check contents
	expect_that(anRBM, is_a("RestrictedBolzmannMachine"))
	expect_that(anRBM$input, is_identical_to(l4g))
	expect_that(anRBM$output, is_identical_to(l10b))
	expect_that(anRBM$weights.env$weights, is_equivalent_to(rep(0, 4+10+4*10)))
	expect_that(anRBM$weights.env$breaks, is_equivalent_to(c(0, 4, 44, 54)))
	
	# Same specifying weights
	anRBMWithWeights <- RestrictedBolzmannMachine(l4g, l10b, 1:54)
	expect_that(anRBMWithWeights, is_a("RestrictedBolzmannMachine"))
	expect_that(anRBMWithWeights$input, is_identical_to(l4g))
	expect_that(anRBMWithWeights$output, is_identical_to(l10b))
	expect_that(anRBMWithWeights$weights.env$weights, is_equivalent_to(1:54))
	expect_that(anRBMWithWeights$weights.env$breaks, is_equivalent_to(c(0, 4, 44, 54)))
})
	
test_that("Cannot create invalid RestrictedBolzmannMachine", {
	# Create some Layers
	l4g <- Layer(4, "gaussian")
	l10b <- Layer(10, "binary")
	
	# invalid weights
	expect_that(RestrictedBolzmannMachine(l4g, l10b, 1:55), throws_error())
	expect_that(RestrictedBolzmannMachine(l4g, l10b, numeric(0)), throws_error())
	expect_that(RestrictedBolzmannMachine(l4g, l10b, 54), throws_error())
	expect_that(RestrictedBolzmannMachine(l4g, l10b, as.character(1:54)), throws_error())
	expect_that(RestrictedBolzmannMachine(l4g, l10b, matrix(1:54)), throws_error())
	
	# Accidentally passing a Layer as weights
	expect_that(RestrictedBolzmannMachine(l4g, l10b, Layer(4, "gaussian")), throws_error())
	
	# not passing layers as input/output
	expect_that(RestrictedBolzmannMachine(c(4, 10), c("gaussian", "binary"), 1:54), throws_error())
	expect_that(RestrictedBolzmannMachine(4, 10, 1:54), throws_error())
	expect_that(RestrictedBolzmannMachine(l4g, 10, 1:54), throws_error())
	expect_that(RestrictedBolzmannMachine(4, l10b, 1:54), throws_error())
	
	# same with no weights
	expect_that(RestrictedBolzmannMachine(c(4, 10), c("gaussian", "binary")), throws_error())
	expect_that(RestrictedBolzmannMachine(4, 10), throws_error())
	expect_that(RestrictedBolzmannMachine(l4g, 10), throws_error())
	expect_that(RestrictedBolzmannMachine(4, l10b), throws_error())

})


test_that("getWeightsFromEnv() works", {
	l4g <- Layer(4, "gaussian")
	l10b <- Layer(10, "binary")
	anRBMWithWeights <- RestrictedBolzmannMachine(l4g, l10b, 1:54)
	
	# Works with weights.env
	expect_that(DeepLearning:::getWeightsFromEnv(anRBMWithWeights$weights.env, "all"), is_equivalent_to(1:54))
	expect_that(DeepLearning:::getWeightsFromEnv(anRBMWithWeights$weights.env, "b"), is_equivalent_to(1:4))
	expect_that(DeepLearning:::getWeightsFromEnv(anRBMWithWeights$weights.env, "w"), is_equivalent_to(5:44))
	expect_that(DeepLearning:::getWeightsFromEnv(anRBMWithWeights$weights.env, "c"), is_equivalent_to(45:54))
	expect_that(DeepLearning:::getWeightsFromEnv(anRBMWithWeights$weights.env, "bw"), is_equivalent_to(1:44))
	expect_that(DeepLearning:::getWeightsFromEnv(anRBMWithWeights$weights.env, "wc"), is_equivalent_to(5:54))
	
	# Uppercase
	expect_that(DeepLearning:::getWeightsFromEnv(anRBMWithWeights$weights.env, "All"), is_equivalent_to(1:54))
	expect_that(DeepLearning:::getWeightsFromEnv(anRBMWithWeights$weights.env, "B"), is_equivalent_to(1:4))
	expect_that(DeepLearning:::getWeightsFromEnv(anRBMWithWeights$weights.env, "W"), is_equivalent_to(5:44))
	expect_that(DeepLearning:::getWeightsFromEnv(anRBMWithWeights$weights.env, "C"), is_equivalent_to(45:54))
	expect_that(DeepLearning:::getWeightsFromEnv(anRBMWithWeights$weights.env, "BW"), is_equivalent_to(1:44))
	expect_that(DeepLearning:::getWeightsFromEnv(anRBMWithWeights$weights.env, "WC"), is_equivalent_to(5:54))
	
	
	# Works with weights & breaks
	expect_that(DeepLearning:::getWeightsFromEnv(which="all", breaks=anRBMWithWeights$weights.env$breaks, weights=anRBMWithWeights$weights.env$weights), is_equivalent_to(1:54))
	expect_that(DeepLearning:::getWeightsFromEnv(which="b", breaks=anRBMWithWeights$weights.env$breaks, weights=anRBMWithWeights$weights.env$weights), is_equivalent_to(1:4))
	expect_that(DeepLearning:::getWeightsFromEnv(which="w", breaks=anRBMWithWeights$weights.env$breaks, weights=anRBMWithWeights$weights.env$weights), is_equivalent_to(5:44))
	expect_that(DeepLearning:::getWeightsFromEnv(which="c", breaks=anRBMWithWeights$weights.env$breaks, weights=anRBMWithWeights$weights.env$weights), is_equivalent_to(45:54))
	expect_that(DeepLearning:::getWeightsFromEnv(which="bw", breaks=anRBMWithWeights$weights.env$breaks, weights=anRBMWithWeights$weights.env$weights), is_equivalent_to(1:44))
	expect_that(DeepLearning:::getWeightsFromEnv(which="wc", breaks=anRBMWithWeights$weights.env$breaks, weights=anRBMWithWeights$weights.env$weights), is_equivalent_to(5:54))

})


test_that("Can create RestrictedBolzmannMachine with uniform weights", {
	# Create an RBM
	l4g <- Layer(4, "gaussian")
	l10b <- Layer(10, "binary")
	anRBM <- RestrictedBolzmannMachine(l4g, l10b, initialize = "uniform")
	expect_gt(sum(abs(anRBM$weights.env$weights)), 0)
})
