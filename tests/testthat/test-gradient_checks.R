context("DBN_propagation")

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

# # Compute gradients 
# gradientF <- DeepLearning:::unit_DbnGradient(unroll(dbn), matrix(f, 1, 3, byrow=TRUE))
# 
# binaryDerivative <- function(a) exp(-a) / (1 + exp(-a)) ^2
# continuousDerivative <- function(a) 1 / a ^2 - 1 / (exp(a) + exp(-a) -2) 
# 
# 
# # A trained
# trained <- train(unroll(dbn), matrix(rep(f, 2), 2, 3, byrow=TRUE), maxiters=1, batchsize=1)


test_that("Can predict proper data", {
	
	# Check foward pass
	# first layer
	pred1 <- predict(dbn[1], f)
	pred1.expected <- c(0.80612106217614, 0.62010643234309, 0.679178699175393, 0.431680016521752)
	# Second layer
	pred2 <- predict(dbn[2], pred1)
	pred2.expected <- c(2.38023363493194, -6.36646162772927)
	
	expect_that(pred1, equals(pred1.expected))
	expect_that(pred2, equals(pred2.expected))
	# Alternatives
	expect_that(predict(dbn[[1]], f), equals(pred1.expected))
	expect_that(predict(dbn[[2]], pred1), equals(pred2.expected))
	expect_that(predict(dbn, f), equals(pred2.expected))
	expect_that(predict(unroll(dbn), f), equals(pred2.expected))
	expect_that(predict(unroll(dbn)[1], f), equals(pred1.expected))
	expect_that(predict(unroll(dbn)[2], pred1), equals(pred2.expected))
	expect_that(predict(unroll(dbn)[[1]], f), equals(pred1.expected))
	expect_that(predict(unroll(dbn)[[2]], pred1), equals(pred2.expected))
	
	# Backward pass
	reverse.pred2 <- predict(rev(dbn[[2]]), pred2)
	reverse.pred2.expected <- c(0.999999988486559, 0.99999999992654, 0.000709084256067969, 0.00168671191831906)
	reverse.pred1 <- predict(rev(dbn[[1]]), reverse.pred2)
	reverse.pred1.expected <- c(0.921516993798264, 0.566790087698008, 0.735781154062391)
	
	expect_that(reverse.pred2, equals(reverse.pred2.expected))
	expect_that(reverse.pred1, equals(reverse.pred1.expected))
	# Alternatives
	expect_that(predict(rev(dbn[2]), pred2), equals(reverse.pred2.expected))
	expect_that(predict(rev(dbn[1]), reverse.pred2), equals(reverse.pred1.expected))
	expect_that(predict(rev(dbn)[1], pred2), equals(reverse.pred2.expected))
	expect_that(predict(rev(dbn)[2], reverse.pred2), equals(reverse.pred1.expected))
	expect_that(predict(rev(dbn)[[1]], pred2), equals(reverse.pred2.expected))
	expect_that(predict(rev(dbn)[[2]], reverse.pred2), equals(reverse.pred1.expected))
	expect_that(reconstruct(dbn[2], pred1), equals(reverse.pred2.expected))
	# expect_that(reconstruct(dbn[1], f), equals(reverse.pred1.expected)) # We haven't passed to layer 2... 
	expect_that(reconstruct(dbn[[2]], pred1), equals(reverse.pred2.expected))
	# expect_that(reconstruct(dbn[[1]], f), equals(reverse.pred1.expected)) # We haven't passed to layer 2... 
	expect_that(reconstruct(dbn, f), equals(reverse.pred1.expected))
	# And with unrolling
	expect_that(reconstruct(unroll(dbn), f), equals(reverse.pred1.expected))
	
	
})

test_that("Can compute gradient properly", {
	
	expected.gradient <- c(
		# W0
		as.vector(t(matrix(c(0, -2.08343629206131e-06, -4.16687258412261e-06,
							 0,  -1.32620281922223e-06, -2.65240563844447e-06,
							 0,   2.76210449274862e-07,  5.52420898549723e-07,
							 0,  6.2365102645147e-07,   1.24730205290294e-06), 3, 4))),
		
		# C0
		c(-4.16687258412261e-06, -2.65240563844447e-06, 5.52420898549723e-07, 1.24730205290294e-06),
		
		# W1 
		as.vector(t(matrix(c(1.78828303855696e-06, 1.37563185865117e-06, 1.50667660835681e-06, 9.57630420356343e-07,
							 6.24070864242369e-06, 4.80064813230306e-06, 5.25796505831505e-06, 3.34191641463432e-06), 4, 2))),
	
	
	# c1 
		c(2.21838024394184e-06, 7.74165188734403e-06),
		
		# W2
		as.vector(t(matrix(c(2.4885247590919e-09, -6.65611020527647e-09,
							 -5.9979504776637e-13,  1.60428459629587e-12,
							 -1.94839533105864e-05,  5.21141451359517e-05,
							 4.00917756460577e-05, -0.000107234326703166), 2, 4))),
		
		# c2
		c(1.045496006178e-09, -2.51989989118661e-13, -8.18573144444432e-06, 1.68436304141228e-05),
	
		# W3
		as.vector(t(matrix(c(0.00567388485780531,   0.00567388492271444,  4.02326246973376e-06,  9.57020932301627e-06,
							 0.00538781061402395,   0.0053878106756604,  3.82041172506634e-06,  9.08768448095062e-06,
							 -0.0137229521993177,    -0.013722952356308, -9.73072946334363e-06, -2.31466672956097e-05), 4, 3))),
		# c3
		c(5.673885e-03, 0.00538781067605621, -0.0137229523573161)
	)
	
	observed.gradient <- DeepLearning:::unit_DbnGradient(unroll(dbn), matrix(f, 1, 3, byrow=TRUE))
	
	#df <- data.frame(exp=expected.gradient, obs=observed.gradient[-(1:3)], diff=expected.gradient - observed.gradient[-(1:3)])
	#df[order(df$diff),]
	
	expect_that(expected.gradient, equals(observed.gradient[-(1:3)]))

})

test_that("Can train a minimal example", {
	
	unrolled <- unroll(dbn)
	trained <- train(unrolled, f, maxiters=50, batchsize=1, continue.function = continue.function.always)
	trained.100 <- train(unrolled, f, maxiters=100, batchsize=1, optim.control=list(maxit=100), continue.function = continue.function.always)
	
	pretrained.error <- rmse(unrolled, f)
	trained.error <- rmse(trained, f)
	trained.100.error <- rmse(trained.100, f)
	
	expect_that(trained.error, is_less_than(pretrained.error)) # We train at all
	expect_that(trained.100.error, is_less_than(0.1)) # With 100 iterations it should really do it!
	
})