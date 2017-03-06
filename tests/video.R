# This test creates a video to showcase the pre-training (last layer) and training of the DBN
# As it is very slow it is disabled by default


if (0) {
	library(DeepLearning)
	library(mnist)
	data(mnist)
	
	maxiters.pretrain <- 1e6
	sprintf.fmt.iter <- sprintf("%%0%dd", nchar(sprintf("%d", maxiters.pretrain)))
	dbn <- DeepBeliefNet(Layers(c(784, 1000, 500, 250, 2), input = "continuous", output = "binary"))
	mnist.data.layer <- mnist
	for (i in 1:3) {
		rbm <- pretrain(dbn[[i]], mnist.data.layer$train$x,  penalization = "l2", lambda=0.0002,
							 epsilon=c(.1, .1, .1, .001)[i], batchsize = 100, maxiters=maxiters.pretrain,
							 continue.function = continue.function.always)
		mnist.data.layer$train$x <- predict(dbn[[i]], mnist.data.layer$train$x)
		mnist.data.layer$test$x <- predict(dbn[[i]], mnist.data.layer$test$x)
		save(rbm, file = sprintf("video/rbm-%s-final.RData", i))
		dbn[[i]] <- rbm
		
	}
	
	diag <- list(rate = "accelerate", data = NULL, f = function(rbm, batch, data, iter, batchsize, maxiters, layer) {
		save(rbm, file = sprintf("video/rbm-4-%s.RData", sprintf(sprintf.fmt.iter, iter)))
	})
	
	rbm <- pretrain(dbn[[4]], mnist.data.layer$train$x,  penalization = "l2", lambda=0.0002,
						 epsilon=c(.1, .1, .1, .001)[i], batchsize = 100, maxiters=maxiters.pretrain,
						 continue.function = continue.function.always, diag = diag)
	save(rbm, file = sprintf("video/rbm-4-final.RData", i))
	dbn[[4]] <- rbm
	
}


