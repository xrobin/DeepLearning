# This test creates a video to showcase the pre-training (last layer) and training of the DBN
# As it is very slow it is disabled by default

do.run <- 0
if (do.run) {
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
	
	# Fine-tune
	maxiters.train <- 10000
	sprintf.fmt.iter <- sprintf("%%0%dd", nchar(sprintf("%d", maxiters.train)))
	diag <- list(rate = "each", data = NULL, f = function(dbn, batch, data, iter, batchsize, maxiters, layer) {
		save(dbn, file = sprintf("video/dbn-finetune-%s.RData", sprintf(sprintf.fmt.iter, iter)))
	})
	dbn <- train(dbn, mnist$train$x, batchsize = 100, maxiters=maxiters.train,
					continue.function = continue.function.always, diag = diag)
	save(dbn, file = sprintf("video/dbn-finetune-final.RData", i))
	
}

if (do.run) {
	library(DeepLearning)
	library(mnist)
	data(mnist)
	
	dbn <- DeepBeliefNet(Layers(c(784, 1000, 500, 250, 2), input = "continuous", output = "binary"))
	for (i in 1:3) {
		load(sprintf("video/rbm-%d-final.RData", i))
		dbn[[i]] <- rbm
	}
	
	for (file in list.files("video", pattern = "rbm-4-.+\\.RData", full.names = TRUE)) {
		load(file)
		dbn[[4]] <- rbm
		
		png(sub(".RData", ".png", file), width = 1920, height = 1080) # hd output
		predictions <- predict(dbn, mnist$test$x)
		reconstructions <- reconstruct(dbn, mnist$test$x)
		plot.mnist(model = dbn, x = mnist$test$x, label = mnist$test$y+1, predictions = predictions, reconstructions = reconstructions			   )
		dev.off()
	}

	
	# Fine-tuning
	for (file in list.files("video", pattern = "video/dbn-finetune-.+\\.RData", full.names = TRUE)) {
		load(file)
		
		png(sub(".RData", ".png", file), width = 1920, height = 1080) # hd output
		predictions <- predict(dbn, mnist$test$x)
		reconstructions <- reconstruct(dbn, mnist$test$x)
		plot.mnist(model = dbn, x = mnist$test$x, label = mnist$test$y+1, predictions = predictions, reconstructions = reconstructions)
		dev.off()
	}
}
