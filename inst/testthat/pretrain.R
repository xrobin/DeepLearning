context("Pretrain works")

# Create a DBN
dbn <- DeepBeliefNet(Layer(3, "c"), Layer(4, "b"), Layer(2, "g"))

# And an input vector
set.seed(42)
f <- jitter(matrix(c(0, .5, 1), 100, 3, byrow=TRUE))



test_that("Can pretrain a minimal example", {
	pretrained.dbn <- pretrain(dbn, f, maxiters=10); print(pretrained.dbn$weights.env$weights)
	pretrained.dbn <- pretrain(dbn, f, maxiters=10, train.b = FALSE, train.c = FALSE); print(pretrained.dbn$weights.env$weights)
	pretrained.dbn <- pretrain(dbn, f, maxiters=10, train.b = FALSE, train.c = 2); print(pretrained.dbn$weights.env$weights)
	pretrained.dbn <- pretrain(dbn, f, maxiters=10, train.b = FALSE, train.c = TRUE); print(pretrained.dbn$weights.env$weights)
})