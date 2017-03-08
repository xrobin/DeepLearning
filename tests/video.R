# This test creates a video to showcase the pre-training (last layer) and training of the DBN
# As it is very slow it is disabled by default

format.timediff <- function(start.time) {
    diff = as.numeric(difftime(Sys.time(), start.time, units="mins"))
    hr <- diff%/%60
    min <- floor(diff - hr * 60)
    sec <- round(diff%%1 * 60,digits=2)
    return(paste(hr,min,sec,sep=':'))
}

do.run <- 0
if (do.run) {
	library(DeepLearning)
	library(mnist)
	data(mnist)
	
	maxiters.pretrain <- 1e6 # Typically takes around 1 day per layer (first ones) when optimization is on with -O2
	sprintf.fmt.iter <- sprintf("%%0%dd", nchar(sprintf("%d", maxiters.pretrain)))
	dbn <- DeepBeliefNet(Layers(c(784, 1000, 500, 250, 2), input = "continuous", output = "binary"), initialize = "uniform")
	mnist.data.layer <- mnist
	for (i in 1:3) {
		print(head(dbn[[i]]$b))
		start.time <- Sys.time()
		diag <- list(rate = "accelerate", data = NULL, f = function(rbm, batch, data, iter, batchsize, maxiters, layer) {
			print(sprintf("%s[%s/%s] in %s", layer, iter, maxiters, format.timediff(start.time)))
		})
		rbm <- pretrain(dbn[[i]], mnist.data.layer$train$x,  penalization = "l2", lambda=0.0002,
							 epsilon=c(.1, .1, .1, .001)[i], batchsize = 100, maxiters=maxiters.pretrain,
							 continue.function = continue.function.always, diag = diag)
		mnist.data.layer$train$x <- predict(rbm, mnist.data.layer$train$x)
		mnist.data.layer$test$x <- predict(rbm, mnist.data.layer$test$x)
		save(rbm, file = sprintf("video/rbm-%s-%s.RData", i, "final"))
		dbn[[i]] <- rbm
	}
	
	start.time <- Sys.time()
	diag <- list(rate = "accelerate", data = NULL, f = function(rbm, batch, data, iter, batchsize, maxiters, layer) {
		save(rbm, file = sprintf("video/rbm-4-%s.RData", sprintf(sprintf.fmt.iter, iter)))
		print(sprintf("%s[%s/%s] in %s", layer, iter, maxiters, format.timediff(start.time)))
	})
	
	print(head(dbn[[4]]$b))
	rbm <- pretrain(dbn[[4]], mnist.data.layer$train$x,  penalization = "l2", lambda=0.0002,
						 epsilon=.001, batchsize = 100, maxiters=maxiters.pretrain,
						 continue.function = continue.function.always, diag = diag)
	save(rbm, file = sprintf("video/rbm-4-%s.RData", "final"))
	dbn[[4]] <- rbm
	
	# Fine-tune
	maxiters.train <- 10000
	start.time <- Sys.time()
	sprintf.fmt.iter <- sprintf("%%0%dd", nchar(sprintf("%d", maxiters.train)))
	diag <- list(rate = "each", data = NULL, f = function(dbn, batch, data, iter, batchsize, maxiters, layer) {
		save(dbn, file = sprintf("video/dbn-finetune-%s.RData", sprintf(sprintf.fmt.iter, iter)))
		print(sprintf("%s[%s/%s] in %s", layer, iter, maxiters, format.timediff(start.time)))
	})
	dbn <- train(unroll(dbn), mnist$train$x, batchsize = 100, maxiters=maxiters.train,
					continue.function = continue.function.always, diag = diag)
	save(dbn, file = sprintf("video/dbn-finetune-%s.RData", "final"))
	
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
		iter <- stringr::str_match(file, "rbm-4-(.+)\\.RData")[,2]
		png(sub(".RData", ".png", file), width = 1920, height = 1080) # hd output
		predictions <- predict(dbn, mnist$test$x)
		reconstructions <- reconstruct(dbn, mnist$test$x)
		plot.mnist(model = dbn, x = mnist$test$x, label = mnist$test$y+1, predictions = predictions, reconstructions = reconstructions, ncol = 16)
		par(family="mono")
		legend("bottomleft", legend = sprintf("Mean error = %.3f", mean(error)), bty="n")
		legend("bottomright", legend = sprintf("Iteration = %s", iter), bty="n")
		dev.off()
	}

	# Fine-tuning
	for (file in list.files("video", pattern = "video/dbn-finetune-.+\\.RData", full.names = TRUE)) {
		load(file)
		iter <- stringr::str_match(file, "video/dbn-finetune-.+\\.RData")[,2]
		png(sub(".RData", ".png", file), width = 1920, height = 1080) # hd output
		predictions <- predict(dbn, mnist$test$x)
		reconstructions <- reconstruct(dbn, mnist$test$x)
		plot.mnist(model = dbn, x = mnist$test$x, label = mnist$test$y+1, predictions = predictions, reconstructions = reconstructions, ncol = 16)
		par(family="mono")
		legend("bottomleft", legend = sprintf("Mean error = %.3f", mean(error)), bty="n")
		legend("bottomright", legend = sprintf("Iteration = %s", iter), bty="n")
		dev.off()
	}
}
