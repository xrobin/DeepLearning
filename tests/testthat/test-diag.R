skip_if_not(run_slow_tests, message = "Slow test skipped")

format.timediff <- function(start.time) {
	diff = as.numeric(difftime(Sys.time(), start.time, units="mins"))
	hr <- diff%/%60
	min <- floor(diff - hr * 60)
	sec <- round(diff%%1 * 60,digits=2)
	return(paste(hr,min,sec,sep=':'))
}

library(DeepLearning)
library(mnist)
data(mnist)

dbn <- DeepBeliefNet(Layers(c(784, 1000, 500, 250, 2), input = "continuous", output = "binary"), initialize = "uniform")

maxiters.pretrain <- 300L
batchsize.pretrain <- 10L
sprintf.fmt.iter <- sprintf("%%0%dd", nchar(sprintf("%d", maxiters.pretrain)))

diag.called.times <- 0
diag.data.dim <- NA
diag.batch.dim <- NA
diag <- list(rate = "none", data = NULL, f = function(rbm, batch, data, iter, batchsize, maxiters, layer) {
	print(sprintf("%s[%s/%s] in %s", layer, iter, maxiters, format.timediff(start.time)))
	diag.called.times <<- diag.called.times + 1
	diag.batch.dim <<- dim(batch)
	diag.data.dim <<- dim(data)
})

test_that("Diag not called with rate = 'none' in pretrain", {
	expect_output(dbn[[1]] <<- pretrain(dbn[[1]], mnist$train$x,  penalization = "l2", lambda=0.0002,
									   epsilon=.1, batchsize = batchsize.pretrain, maxiters=maxiters.pretrain,
									   continue.function = continue.function.always, diag = diag),
				  regexp = "Pre-training until stopCounter reaches 30"
	)
	expect_equal(diag.called.times, 0)
	expect_true(is.na(diag.data.dim))
	expect_true(is.na(diag.batch.dim))
})

test_that("Diag called with rate = 'each' in pretrain", {
	diag$rate <- "each"
	diag$data <-  predict(dbn[[1]], mnist$test$x)
	start.time <<- Sys.time()
	expect_output(dbn[[2]] <<- pretrain(dbn[[2]], predict(dbn[[1]], mnist$train$x),  penalization = "l2", lambda=0.0002,
										epsilon=.1, batchsize = batchsize.pretrain, maxiters=maxiters.pretrain,
										continue.function = continue.function.always, diag = diag),
				  regexp = "1\\[300/300\\]"
	)
	expect_equal(diag.called.times, 301)
	expect_identical(diag.data.dim, c(10000L, 1000L))
	expect_identical(diag.batch.dim, c(batchsize.pretrain, 1000L))
})

test_that("Diag called with rate = 'accelerate' in pretrain", {
	diag$rate = "accelerate"
	diag$data <-  predict(dbn[1:2], mnist$test$x)
	start.time <<- Sys.time()
	expect_output(dbn.34 <<- pretrain(dbn[3:4], predict(dbn[1:2], mnist$train$x),  penalization = "l2", lambda=0.0002,
										epsilon=c(.1, 0.001), batchsize = batchsize.pretrain, maxiters=maxiters.pretrain,
										continue.function = continue.function.always, diag = diag),
				  regexp = "2\\[300/300\\]"
	)
	expect_identical(diag.data.dim, c(10000L, 250L)) # 250 is input size of the last layer
	expect_identical(diag.batch.dim, c(batchsize.pretrain, 250L))
	expect_equal(diag.called.times, 805)
})

#### Now train
dbn <- unroll(pretrained.mnist)

maxiters.train <- 210L
batchsize.train <- 10L
sprintf.fmt.iter <- sprintf("%%0%dd", nchar(sprintf("%d", maxiters.train)))

diag.called.times <- 0
diag.data.dim <- NA
diag.batch.dim <- NA
diag <- list(rate = "none", data = NULL, f = function(rbm, batch, data, iter, batchsize, maxiters) {
	print(sprintf("[%s/%s] in %s", iter, maxiters, format.timediff(start.time)))
	diag.called.times <<- diag.called.times + 1
	diag.batch.dim <<- dim(batch)
	diag.data.dim <<- dim(data)
})

test_that("Diag not called with rate = 'none' in train", {
	expect_output(dbn <<- train(dbn, mnist$train$x, batchsize = batchsize.train, maxiters=10, # 10 is enough to ensure diag isn't called
										continue.function = continue.function.always, diag = diag),
				  regexp = "Applying gradient to gradientRBMs"
	)
	expect_equal(diag.called.times, 0)
	expect_true(is.na(diag.data.dim))
	expect_true(is.na(diag.batch.dim))
})

test_that("Diag called with rate = 'each' in train", {
	diag$rate <- "each"
	diag$data <-  mnist$test$x
	start.time <<- Sys.time()
	expect_output(dbn <<- train(dbn, mnist$train$x, batchsize = batchsize.train, maxiters=maxiters.train,
								continue.function = continue.function.always, diag = diag),
				  regexp = "\\[210/210\\]"
	)
	expect_equal(diag.called.times, 211)
	expect_identical(diag.data.dim, c(10000L, 784L))
	expect_identical(diag.batch.dim, c(batchsize.train, 784L))
})


test_that("Diag called with rate = 'each' in train", {
	diag$rate <- "accelerate"
	diag$data <-  mnist$test$x
	start.time <<- Sys.time()
	expect_output(dbn <<- train(dbn, mnist$train$x, batchsize = batchsize.train, maxiters=maxiters.train,
								continue.function = continue.function.always, diag = diag),
				  regexp = "\\[210/210\\]"
	)
	expect_equal(diag.called.times, 211)
	expect_identical(diag.data.dim, c(10000L, 784L))
	expect_identical(diag.batch.dim, c(batchsize.train, 784L))
})
