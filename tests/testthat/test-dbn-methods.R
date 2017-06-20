context("DBN methods")

l4g <- Layer(4, "gaussian")
l4b <- Layer(4, "binary")

l3g <- Layer(3, "gaussian")
l3c <- Layer(3, "continuous")


test_that("print works for DBN", {
	expect_output(DeepBeliefNet(l4g))
	expect_output(print(DeepBeliefNet(l4g, l3g)))
	expect_output(print(DeepBeliefNet(l4g, l3g, l4b)))
	expect_output(print(DeepBeliefNet(l4g, l3g, l4b, l3c)))
})


test_that("Can print different stages of DBN", {
	expect_output(print(pretrained.mnist), "Pre-trained")
	expect_output(print(trained.mnist), "Fine-tuned")
})
