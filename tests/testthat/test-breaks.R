initial.breaks.exp <- c(0L, 784L, 784784L, 785784L, 1285784L, 1286284L, 1411284L, 1411534L, 
						1412034L, 1412036L)

unrolled.breaks.exp <-	c(0L, 784L, 784784L, 785784L, 1285784L, 1286284L, 1411284L, 1411534L, 
	  1412034L, 1412036L, 1412536L, 1412786L, 1537786L, 1538286L, 2038286L, 
	  2039286L, 2823286L, 2824070L)


dbn <- DeepBeliefNet(Layers(c(784, 1000, 500, 250, 2), input="continuous", output="gaussian"))
unrolled <- unroll(dbn)

test_that("Breaks match expectations", {
	expect_identical(dbn$weights.env$breaks, initial.breaks.exp)
	expect_identical(unrolled$weights.env$breaks, unrolled.breaks.exp)
})
