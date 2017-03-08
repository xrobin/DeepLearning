# This test creates a video to showcase the pre-training (last layer) and training of the DBN
# This script does it in 2 passes, saving the network first, and then producing HD PNG files with more control and knowledge of the final network

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
		# Only save the RBM for later re-processing
		iter.fmt <- sprintf(sprintf("%%0%dd", nchar(sprintf("%d", maxiters))), iter)
		save(rbm, file = file.path("video", sprintf("rbm-%s-%s.RData", layer, iter.fmt)))
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
	}
	else if (iter == maxiters) {
		setTxtProgressBar(DBNprogressBar, iter)
		close(DBNprogressBar)
	}
	else {
		setTxtProgressBar(DBNprogressBar, iter)
	}
	
	# Only save the DBN for later re-processing
	iter.fmt <- sprintf(sprintf("%%0%dd", nchar(sprintf("%d", maxiters))), iter)
	save(dbn, file = file.path("video", sprintf("dbn-%s-%s.RData", "finetune", iter.fmt)))
})

# Fine-tune
train(unroll(dbn), mnist$train$x, batchsize = 100, maxiters=10000,
			 continue.function = continue.function.always, diag = diag)


# Re-process the saved images
for (file in list.files("video", pattern = "rbm-4-.+\\.RData", full.names = TRUE)) {
	load(file)
	dbn[[4]] <- rbm
	iter <- stringr::str_match(file, "rbm-4-(.+)\\.RData")[,2]
	png(sub(".RData", ".png", file), width = 1920, height = 1080) # hd output
	predictions <- predict(dbn, mnist$test$x)
	reconstructions <- reconstruct(dbn, mnist$test$x)
	error <- errorSum(dbn, mnist$test$x) / nrow(mnist$test$x)
	plot.mnist(model = dbn, x = mnist$test$x, label = mnist$test$y+1, predictions = predictions, reconstructions = reconstructions, ncol = 16)
	par(family="mono")
	legend("bottomleft", legend = sprintf("Mean error = %.3f", error), bty="n")
	legend("bottomright", legend = sprintf("Iteration = %s", iter), bty="n")
	dev.off()
}

# Fine-tuning
for (file in list.files("video", pattern = "dbn-finetune-.+\\.RData", full.names = TRUE)) {
	load(file)
	iter <- stringr::str_match(file, "dbn-finetune-(.+)\\.RData")[,2]
	png(sub(".RData", ".png", file), width = 1920, height = 1080) # hd output
	predictions <- predict(dbn, mnist$test$x)
	reconstructions <- reconstruct(dbn, mnist$test$x)
	error <- errorSum(dbn, mnist$test$x) / nrow(mnist$test$x)
	plot.mnist(model = dbn, x = mnist$test$x, label = mnist$test$y+1, predictions = predictions, reconstructions = reconstructions, ncol = 16)
	par(family="mono")
	legend("bottomleft", legend = sprintf("Mean error = %.3f", error), bty="n")
	legend("bottomright", legend = sprintf("Iteration = %s", iter), bty="n")
	dev.off()
}
