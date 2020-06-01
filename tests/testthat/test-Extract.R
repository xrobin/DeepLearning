context("Extraction")

l3g <- Layer(3, "gaussian")
l4b <- Layer(4, "binary")
l2c <- Layer(2, "continuous")

dbn <- c(l3g, l4b, l2c) # Tested in c.R
pretrained.dbn <- dbn
pretrained.dbn$pretrained <- TRUE; 
pretrained.dbn[[1]]$pretrained <- TRUE; pretrained.dbn[[2]]$pretrained <- TRUE
unrolled.dbn <- unroll(pretrained.dbn) # Tested in unroll.R
trained.dbn <- unrolled.dbn; trained.dbn$finetuned <- TRUE

test_that("Extraction of DBN with [.DeepBeliefNet", {
	
	# Cannot extract layer 0	
	expect_that(dbn[0], throws_error("subscript out of bounds"))
	expect_that(trained.dbn[0], throws_error("subscript out of bounds"))
	
	# Cannot extract layer 3	
	expect_that(dbn[3], throws_error("subscript out of bounds"))
	expect_that(trained.dbn[5], throws_error("subscript out of bounds"))
	
	# NULL i
	expect_that(dbn[NULL], throws_error("attempt to select less than one layer"))
	
	# Missing i
	expect_that(length(dbn[]), equals(length(dbn))) # length is tested elsewhere
	expect_that(dbn[], is_a("DeepBeliefNet"))
	expect_that(dbn[], equals(dbn))
	expect_that(length(trained.dbn[]), equals(length(trained.dbn)))
	expect_that(trained.dbn[], is_a("DeepBeliefNet"))
	expect_that(trained.dbn[]$rbms, equals(trained.dbn$rbms)) # The DBN themselves are not equal - see attributes below
	
	
	# 1
	expect_that(length(dbn[1]), equals(2))
	expect_that(dbn[1], is_a("DeepBeliefNet"))
	expect_that(dbn[1], equals(DeepBeliefNet(l3g, l4b)))

	# 2
	expect_that(length(dbn[2]), equals(2))
	expect_that(dbn[2], is_a("DeepBeliefNet"))
	expect_that(dbn[2], equals(DeepBeliefNet(l4b, l2c)))
	
	# 3
	expect_that(length(trained.dbn[3]), equals(2))
	expect_that(trained.dbn[3], is_a("DeepBeliefNet"))
	expect_that(trained.dbn[3]$rbms, equals(DeepBeliefNet(l2c, l4b)$rbms))
	
	# 4
	expect_that(length(trained.dbn[4]), equals(2))
	expect_that(trained.dbn[4], is_a("DeepBeliefNet"))
	expect_that(trained.dbn[4]$rbms, equals(DeepBeliefNet(l4b, l3g)$rbms))
	
	# Multiple
	expect_that(trained.dbn[1:2]$rbms, equals(DeepBeliefNet(l3g, l4b, l2c)$rbms))
	expect_that(trained.dbn[1:3]$rbms, equals(DeepBeliefNet(l3g, l4b, l2c, l4b)$rbms))
	expect_that(trained.dbn[1:4]$rbms, equals(DeepBeliefNet(l3g, l4b, l2c, l4b, l3g)$rbms))
	expect_that(trained.dbn[2:4]$rbms, equals(DeepBeliefNet(l4b, l2c, l4b, l3g)$rbms))
	expect_that(trained.dbn[2:3]$rbms, equals(DeepBeliefNet(l4b, l2c, l4b)$rbms))
	expect_that(trained.dbn[3:4]$rbms, equals(DeepBeliefNet(l2c, l4b, l3g)$rbms))
	expect_that(trained.dbn[c(1, 4)]$rbms, equals(DeepBeliefNet(l3g, l4b, l3g)$rbms))
	
	# Errors
	expect_that(trained.dbn[c(1, 3)], throws_error("incompatible layers cannot be stacked"))
	expect_that(trained.dbn[c(2, 4)], throws_error("incompatible layers cannot be stacked"))
	
	
	# Negative indices
	expect_that(trained.dbn[-1]$rbms, equals(DeepBeliefNet(l4b, l2c, l4b, l3g)$rbms))
	expect_that(trained.dbn[-2], throws_error("incompatible layers cannot be stacked"))
	expect_that(trained.dbn[-3], throws_error("incompatible layers cannot be stacked"))
	expect_that(trained.dbn[-4]$rbms, equals(DeepBeliefNet(l3g, l4b, l2c, l4b)$rbms))
	
	# Multiple negative indices
	expect_that(trained.dbn[-c(1, 2)]$rbms, equals(DeepBeliefNet(l2c, l4b, l3g)$rbms))
	expect_that(trained.dbn[-c(1, 3)], throws_error("incompatible layers cannot be stacked"))
	expect_that(trained.dbn[-c(1, 4)]$rbms, equals(DeepBeliefNet(l4b, l2c, l4b)$rbms))
	expect_that(trained.dbn[-c(2, 3)]$rbms, equals(DeepBeliefNet(l3g, l4b, l3g)$rbms))
	expect_that(trained.dbn[-c(2, 4)], throws_error("incompatible layers cannot be stacked"))
	expect_that(trained.dbn[-c(3, 4)]$rbms, equals(DeepBeliefNet(l3g, l4b, l2c)$rbms))
	expect_that(trained.dbn[-c(1:3)]$rbms, equals(DeepBeliefNet(l4b, l3g)$rbms))
	expect_that(trained.dbn[-c(1, 2, 4)]$rbms, equals(DeepBeliefNet(l2c, l4b)$rbms))
	expect_that(trained.dbn[-c(1:4)]$rbms, throws_error("subscript out of bounds"))

	expect_that(trained.dbn[-c(2:4)]$rbms, equals(DeepBeliefNet(l3g, l4b)$rbms))
	
	# Multiple negative indices shuffled shouldn't make a difference	expect_that(trained.dbn[-c(1, 2)]$rbms, equals(DeepBeliefNet(l2c, l4b, l3g)$rbms))
	expect_that(trained.dbn[-c(3, 1)], throws_error("incompatible layers cannot be stacked"))
	expect_that(trained.dbn[-c(4, 1)]$rbms, equals(DeepBeliefNet(l4b, l2c, l4b)$rbms))
	expect_that(trained.dbn[-c(3, 2)]$rbms, equals(DeepBeliefNet(l3g, l4b, l3g)$rbms))
	expect_that(trained.dbn[-c(4, 2)], throws_error("incompatible layers cannot be stacked"))
	expect_that(trained.dbn[-c(4, 3)]$rbms, equals(DeepBeliefNet(l3g, l4b, l2c)$rbms))
	expect_that(trained.dbn[-c(3:1)]$rbms, equals(DeepBeliefNet(l4b, l3g)$rbms))
	expect_that(trained.dbn[-c(2, 1, 3)]$rbms, equals(DeepBeliefNet(l4b, l3g)$rbms))
	expect_that(trained.dbn[-c(3, 1, 2)]$rbms, equals(DeepBeliefNet(l4b, l3g)$rbms))
	expect_that(trained.dbn[-c(4, 1, 2)]$rbms, equals(DeepBeliefNet(l2c, l4b)$rbms))
	expect_that(trained.dbn[-c(4, 2, 1)]$rbms, equals(DeepBeliefNet(l2c, l4b)$rbms))
	expect_that(trained.dbn[-c(2, 1, 4)]$rbms, equals(DeepBeliefNet(l2c, l4b)$rbms))
	expect_that(trained.dbn[-c(4:1)]$rbms, throws_error("subscript out of bounds"))
	expect_that(trained.dbn[-c(4:2)]$rbms, equals(DeepBeliefNet(l3g, l4b)$rbms))
	
	# Out of bound negative indices are ok
	expect_that(trained.dbn[-5]$rbms, equals(DeepBeliefNet(l3g, l4b, l2c, l4b, l3g)$rbms))
	expect_that(trained.dbn[-6]$rbms, equals(DeepBeliefNet(l3g, l4b, l2c, l4b, l3g)$rbms))
	expect_that(trained.dbn[-c(5, 6)]$rbms, equals(DeepBeliefNet(l3g, l4b, l2c, l4b, l3g)$rbms))

})


test_that("Extraction of DBN with [.DeepBeliefNet and drop", {
	
	expect_that(dbn[1, drop=FALSE], equals(DeepBeliefNet(l3g, l4b)))
	expect_that(dbn[1, drop=FALSE], is_a("DeepBeliefNet"))
	expect_that(dbn[1, drop=TRUE], equals(dbn[[1]]))
	expect_that(dbn[1, drop=TRUE], is_a("RestrictedBolzmannMachine"))
	
	expect_that(dbn[2, drop=FALSE], equals(DeepBeliefNet(l4b, l2c)))
	expect_that(dbn[2, drop=FALSE], is_a("DeepBeliefNet"))
	expect_that(dbn[2, drop=TRUE], equals(dbn[[2]]))
	expect_that(dbn[2, drop=TRUE], is_a("RestrictedBolzmannMachine"))
	
	# Ignored when length > 2
	expect_that(dbn[1:2, drop=TRUE], equals(dbn))
	expect_that(dbn[1:2, drop=TRUE], is_a("DeepBeliefNet")) # Drop ignored
	
	# Chaining
	expect_that(dbn[1, drop=FALSE][drop=TRUE], equals(dbn[[1]]))
	expect_that(dbn[1, drop=FALSE][drop=TRUE], is_a("RestrictedBolzmannMachine"))
	
})

test_that("Attributes of DBN with [.DeepBeliefNet", {
	
	# Missing i
	expect_that(dbn[]$pretrained, is_false())
	expect_that(dbn[]$unrolled, is_false())
	expect_that(dbn[]$finetuned, is_false())
	
	expect_that(pretrained.dbn[]$pretrained, is_true())
	expect_that(pretrained.dbn[]$unrolled, is_false())
	expect_that(pretrained.dbn[]$finetuned, is_false())
	
	expect_that(unrolled.dbn[]$pretrained, is_true())
	expect_that(unrolled.dbn[]$unrolled, is_false())
	expect_that(unrolled.dbn[]$finetuned, is_false())
	
	expect_that(trained.dbn[]$pretrained, is_true())
	expect_that(trained.dbn[]$unrolled, is_false())
	expect_that(trained.dbn[]$finetuned, is_true())
	
})


test_that("Extraction of RBM with [[.DeepBeliefNet", {
	
	expect_that(trained.dbn[[1]], equals(RestrictedBolzmannMachine(l3g, l4b)))
	expect_that(trained.dbn[[2]], equals(RestrictedBolzmannMachine(l4b, l2c)))
	expect_that(trained.dbn[[3]], equals(RestrictedBolzmannMachine(l2c, l4b)))
	expect_that(trained.dbn[[4]], equals(RestrictedBolzmannMachine(l4b, l3g)))
	
	# Wrong extractions throw
	expect_that(dbn[[0]], throws_error("attempt to select less than one element"))
	expect_that(dbn[[]], throws_error()) # I actually don't quite care which error should be thrown here
	expect_that(dbn[[1:2]], throws_error()) # I actually don't quite care which error should be thrown here
	
	# Negative indices are acceptable in some cases
	expect_that(dbn[[-1]], equals(RestrictedBolzmannMachine(l4b, l2c)))
	expect_that(dbn[[-2]], equals(RestrictedBolzmannMachine(l3g, l4b)))
	
	# TODO: This should work but doesn't. It's an edge case so don't bother too much
	# expect_that(dbn[[-(2:3)]], equals(RestrictedBolzmannMachine(l3g, l4b)))
	
	# But not in other
	expect_that(dbn[[-3]], throws_error())
	
	
})

test_that("Replacement of RBM with [[.DeepBeliefNet<-", {
	# Replacing with the same thing should make no difference
	dbn.test1 <- dbn
	dbn.test1[[1]] <- RestrictedBolzmannMachine(l3g, l4b)
	expect_that(dbn.test1, equals(dbn))
	dbn.test1[[2]] <- RestrictedBolzmannMachine(l4b, l2c)
	expect_that(dbn.test1, equals(dbn))
	
	# Test attributes
	expect_that(dbn.test1$pretrained, is_false())
	expect_that(dbn.test1$unrolled, is_false())
	expect_that(dbn.test1$finetuned, is_false())
	
	# Now test if pretrained is on
	dbn.test1[[1]] <- pretrained.dbn[[1]] # pretrained
	dbn.test1[[2]] <- pretrained.dbn[[2]] # pretrained
	# pretrained should be on
	expect_that(dbn.test1$pretrained, is_true())
	expect_that(dbn.test1$unrolled, is_false())
	expect_that(dbn.test1$finetuned, is_false())
	
	# Can replace the first and last layers with something else
	l5c <- Layer(5, "continuous")
	dbn.test2 <- trained.dbn
	dbn.test2[[1]] <- RestrictedBolzmannMachine(l5c, l4b)
	expect_that(dbn.test2$rbms, equals(DeepBeliefNet(l5c, l4b, l2c, l4b, l3g)$rbms))
	dbn.test2[[4]] <- RestrictedBolzmannMachine(l4b, l5c)
	expect_that(dbn.test2$rbms, equals(DeepBeliefNet(l5c, l4b, l2c, l4b, l5c)$rbms))
	
	# But not in the middle
	dbn.test3 <- trained.dbn
	expect_that(dbn.test3[[1]] <- RestrictedBolzmannMachine(l3g, l3g), throws_error("incompatible layers cannot be stacked"))
	expect_that(dbn.test3[[2]] <- RestrictedBolzmannMachine(l4b, l5c), throws_error("incompatible layers cannot be stacked"))
	expect_that(dbn.test3[[2]] <- RestrictedBolzmannMachine(l5c, l2c), throws_error("incompatible layers cannot be stacked"))
	expect_that(dbn.test3[[4]] <- RestrictedBolzmannMachine(l5c, l3g), throws_error("incompatible layers cannot be stacked"))
	
	# We can replace the middle layers with compatible ones
	dbn.test3[[2]] <- RestrictedBolzmannMachine(l4b, l2c)
	dbn.test3[[3]] <- RestrictedBolzmannMachine(l2c, l4b)
	expect_that(dbn.test3$rbms, equals(DeepBeliefNet(l3g, l4b, l2c, l4b, l3g)$rbms))

	# In the end, the network isn't unrolled anymore
	# pretrained should be on
	expect_that(dbn.test3$pretrained, is_false()) # We replaced the whole thing with fresh RBMs
	expect_that(dbn.test3$unrolled, is_false())
	expect_that(dbn.test3$finetuned, is_false())
})

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

test_that("Can extract weights from RBM", {
	rbm <- dbn[[1]]
	expect_identical(rbm$w, 
					 matrix(c(16, 0.14, -0.3, 0.8,
					 		0.03, -0.02, -0.3, 0.25,
					 		0.01, 0.3, 0.6, -0.3), 4, 3))
	expect_identical(rbm$b, c(-3.4, 0.8, 3.0))
	expect_identical(rbm$c, c(1.4, 0.2, 0.3, -0.1))
	
	# RBM2 with upper case
	rbm <- dbn[[2]]
	expect_identical(rbm$W, 
					 matrix(c(-1.2, -3.1,
					 		 3.3, -2.4,
					 		 -1.3, 0.7,
					 		 -0.5, 0.8), 2, 4))
	expect_identical(rbm$B, c(1.4, 0.2, 0.3, -0.1))
	expect_identical(rbm$C, c(2.4, -3.2))
})



test_that("Can assign weights to RBM", {
	rbm <- dbn[[1]]
	rbm$w[] <- as.numeric(1:12) # not integer
	rbm$b <- as.numeric(30:32) 
	rbm$c <- as.numeric(20:23)
	expect_identical(rbm$w, matrix(as.numeric(1:12), 4, 3))
	expect_identical(rbm$b, as.numeric(30:32))
	expect_identical(rbm$c, as.numeric(20:23))
	
	rbm <- dbn[[2]]
	m2 <- matrix(as.numeric(1:8), 2, 4)
	rbm$W <- m2
	rbm$B <- as.numeric(120:123)
	rbm$C <- as.numeric(1:2)
	expect_identical(rbm$W, m2)
	expect_identical(rbm$b, as.numeric(120:123))
	expect_identical(rbm$c, as.numeric(1:2))
})