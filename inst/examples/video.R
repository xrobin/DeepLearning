# This test creates a video to showcase the pre-training (last layer) and training of the DBN
# This script does it in a single pass and saves HD PNG files along the way

library(DeepLearning)
library(mnist)
data(mnist)


# Create the DBN
dbn <- DeepBeliefNet(Layers(c(784, 1000, 500, 250, 2), input = "continuous", output = "binary"), initialize = "uniform")


# Pre-training
# Setup a progress bar, and save RBMs of layer 4
diag <- list(rate = "accelerate", data = mnist$test$x, f = function(rbm, batch, data, iter, batchsize, maxiters, layer) {
	if (iter == 0) {
		DBNprogressBar <<- txtProgressBar(min = 0, max = maxiters, initial = 0, width = NA, style = 3)
		#start.time <<- Sys.time()
	}
	else if (iter == maxiters) {
		setTxtProgressBar(DBNprogressBar, iter)
		close(DBNprogressBar)
	}
	else {
		setTxtProgressBar(DBNprogressBar, iter)
	}
	
	if (layer == 4) {
		# Save an image
		iter.fmt <- sprintf(sprintf("%%0%dd", nchar(sprintf("%d", maxiters))), iter)
		filename <- file.path("video", sprintf("rbm-%s-%s.png", layer, iter.fmt))
		png(filename, width = 1920, height = 1080) # hd output
		predictions <- predict(rbm, data)
		dbn[[layer]] <- rbm
		reconstructions <- predict(rev(dbn), predictions)
		error <- errorSum(dbn, mnist$test$x) / nrow(mnist$test$x)
		plot.mnist(model = dbn, x = mnist$test$x, label = mnist$test$y+1, predictions = predictions, reconstructions = reconstructions, ncol = 16)
		par(family="mono")
		legend("bottomleft", legend = sprintf("Mean error = %.3f", error), bty="n")
		legend("bottomright", legend = sprintf("Iteration = %s", iter.fmt), bty="n")
		dev.off()
	}
})


# Pre-train
dbn <- pretrain(dbn, mnist$train$x,  penalization = "l2", lambda=0.0002,
					 epsilon=c(.1, .1, .1, .001), batchsize = 100, maxiters=1e6,
					 continue.function = continue.function.always, diag = diag)


# Fine-tuning
diag <- list(rate = "each", data = NULL, f = function(dbn, batch, data, iter, batchsize, maxiters) {
	if (iter == 0) {
		DBNprogressBar <<- txtProgressBar(min = 0, max = maxiters, initial = 0, width = NA, style = 3)
		#start.time <<- Sys.time()
	}
	else if (iter == maxiters) {
		setTxtProgressBar(DBNprogressBar, iter)
		close(DBNprogressBar)
	}
	else {
		setTxtProgressBar(DBNprogressBar, iter)
	}
	
	# Save an image
	iter.fmt <- sprintf(sprintf("%%0%dd", nchar(sprintf("%d", maxiters))), iter)
	filename <- file.path("video", sprintf("dbn-%s-%s.png", "finetune", iter.fmt))
	png(sub(".RData", ".png", file), width = 1920, height = 1080) # hd output
	predictions <- predict(dbn, mnist$test$x)
	reconstructions <- reconstruct(dbn, mnist$test$x)
	error <- errorSum(dbn, mnist$test$x) / nrow(mnist$test$x)
	plot.mnist(model = dbn, x = mnist$test$x, label = mnist$test$y+1, predictions = predictions, reconstructions = reconstructions, ncol = 16)
	par(family="mono")
	legend("bottomleft", legend = sprintf("Mean error = %.3f", error), bty="n")
	legend("bottomright", legend = sprintf("Iteration = %s", iter), bty="n")
	dev.off()
})

# Fine-tune
dbn <- train(unroll(dbn), mnist$train$x, batchsize = 100, maxiters=10000,
				continue.function = continue.function.always, diag = diag)
