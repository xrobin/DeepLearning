# This test creates a video to showcase the pre-training (last layer) and training of the DBN
# As it is very slow it is disabled by default


if (0) {
	library(DeepLearning)
	library(mnist)
	data(mnist)
	
	maxiters.pretrain <- 1e6
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
		save(rbm, file = sprintf("video/rbm-4-%s.RData", iter))
		#pdf(sprintf("pretrain-%s.png", iter), width = 1920, height = 1080) # hd output
		#pdf(sprintf("pretrain-%s.pdf", iter), width = 16, height = 9) # hd output
		#dbn[[4]] <- rbm
		#predictions <- predict(dbn, mnist$test$x)
		#reconstructions <- reconstruct(dbn, mnist$test$x)
		#plot.mnist(model = dbn, x = mnist$test$x, label = mnist$test$y, predictions = predictions, reconstructions = reconstructions,
		#		   digits.col = 1:10, pch.bg = 1:10)
		#dev.off()
	})
	
	rbm <- pretrain(dbn[[4]], mnist.data.layer$train$x,  penalization = "l2", lambda=0.0002,
						 epsilon=c(.1, .1, .1, .001)[i], batchsize = 100, maxiters=maxiters.pretrain,
						 continue.function = continue.function.always, diag = diag)
	save(rbm, file = sprintf("video/rbm-4-final.RData", i))
	dbn[[4]] <- rbm
	
}


