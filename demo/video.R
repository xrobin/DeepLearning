# This test creates a video to showcase the pre-training (last layer) and training of the DBN
# As it is very slow it is disabled by default

output.folder <- tempfile()
maxiters.pretrain <- 1e5
maxiters.train <- 2000
highlight.digits = c(72, 3, 83, 91, 6688, 7860, 92, 1, 180, 13)


### Create the temporary folder
cat(sprintf("Saving temporary output in %s", output.folder))
if (dir.exists(output.folder)) {
	warning(sprintf("%s already exists", output.folder))
} else {
	success <- dir.create(output.folder, recursive = TRUE)
	if (! success) {
		stop(sprintf("Cannot create temporary output folder %s", output.folder))
	}
}


### Create DBN
library(DeepLearning)
library(mnist)
data(mnist)
dbn <- DeepBeliefNet(Layers(c(784, 1000, 500, 250, 2), input = "continuous", output = "gaussian"), initialize = "0")


### Pre-train RBM 1-3
mnist.data.layer <- mnist
for (i in 1:3) {
	# Get current RBM
	rbm <- dbn[[i]]
	save(rbm, file = file.path(output.folder, sprintf("rbm-%s-%s.RData", i, "initial")))
	# Progress bar
	pb <- txtProgressBar(0, maxiters.pretrain, style = 3)
	diag <- list(rate = "accelerate", data = NULL, f = function(rbm, batch, data, iter, batchsize, maxiters, layer) {
		setTxtProgressBar(pb, iter)
		#print(sprintf("%s[%s/%s] in %s", i, iter, maxiters, format.timediff(start.time)))
	})
	start.time <- Sys.time()
	# Pre-train current RBM
	rbm <- pretrain(rbm, mnist.data.layer$train$x,  penalization = "l2", lambda=0.0002, momentum = c(0.5, 0.9),
						 epsilon=c(.1, .1, .1, .001)[i], batchsize = 100, maxiters=maxiters.pretrain,
						 continue.function = continue.function.always, diag = diag)
	# Close progress bar
	setTxtProgressBar(pb, maxiters.pretrain)
	close(pb)
	# Save pre-train
	save(rbm, file = file.path(output.folder, sprintf("rbm-%s-%s.RData", i, "final")))
	dbn[[i]] <- rbm
	# Propagate data for next layer
	mnist.data.layer$train$x <- predict(rbm, mnist.data.layer$train$x)
	mnist.data.layer$test$x <- predict(rbm, mnist.data.layer$test$x)
}


### Pre-train RBM 4
# Get current RBM
rbm <- dbn[[4]]
save(rbm, file = file.path(output.folder, sprintf("rbm-%s-%s.RData", 4, "initial")))
# Progress bar
sprintf.fmt.iter <- sprintf("%%0%dd", nchar(sprintf("%d", maxiters.pretrain)))
pb <- txtProgressBar(0, maxiters.pretrain, style = 3)
diag <- list(rate = "accelerate", data = NULL, f = function(rbm, batch, data, iter, batchsize, maxiters, layer) {
	save(rbm, file = file.path(output.folder, sprintf("rbm-4-%s.RData", sprintf(sprintf.fmt.iter, iter))))
	setTxtProgressBar(pb, iter)
})
start.time <- Sys.time()
# Pre-train RBM 4
rbm <- pretrain(rbm, mnist.data.layer$train$x,  penalization = "l2", lambda=0.0002,
					 epsilon=.001, batchsize = 100, maxiters=maxiters.pretrain,
					 continue.function = continue.function.always, diag = diag)
# Close progress bar
setTxtProgressBar(pb, maxiters.pretrain)
close(pb)
# Save pre-train
save(rbm, file = file.path(output.folder, sprintf("rbm-4-%s.RData", "final")))
dbn[[4]] <- rbm


### Fine-tune
save(dbn, file = file.path(output.folder, sprintf("dbn-finetune-%s.RData", "initial")))
# Progress bar
sprintf.fmt.iter <- sprintf("%%0%dd", nchar(sprintf("%d", maxiters.train)))
pb <- txtProgressBar(0, maxiters.train, style = 3)
diag <- list(rate = "each", data = NULL, f = function(dbn, batch, data, iter, batchsize, maxiters) {
	save(dbn, file = file.path(output.folder, sprintf("dbn-finetune-%s.RData", sprintf(sprintf.fmt.iter, iter))))
	print(sprintf("Fine-tune [%s/%s] in %s", iter, maxiters, format.timediff(start.time)))
})
start.time <- Sys.time()
# Train
dbn <- train(unroll(dbn), mnist$train$x, batchsize = 100, maxiters=maxiters.train,
				continue.function = continue.function.always, diag = diag)
# Close progress bar
setTxtProgressBar(pb, maxiters.pretrain)
close(pb)
# Save final model
save(dbn, file = file.path(output.folder, sprintf("dbn-finetune-%s.RData", "final")))


### Generate images for video	
for (i in 1:3) {
	load(file.path(output.folder, sprintf("rbm-%d-final.RData", i)))
	dbn[[i]] <- rbm
}

xlim.range.rbm <- ylim.range.rbm <- c(0, 0)	
for (file in list.files(output.folder, pattern = "rbm-4-.+\\.RData", full.names = TRUE)) {
	load(file)
	dbn[[4]] <- rbm
	iter <- stringr::str_match(file, "rbm-4-(.+)\\.RData")[,2]
	#png(sub(".RData", ".png", file), width = 1920, height = 1080) # full hd output
	png(sub(".RData", ".png", file), width = 1280, height = 720) # hd output
	predictions <- predict(dbn, mnist$test$x)
	reconstructions <- reconstruct(dbn, mnist$test$x)
	iteration.error <- errorSum(dbn, mnist$test$x) / nrow(mnist$test$x)
	# Calculate new range
	xlim.range.rbm <- range(xlim.range.rbm, predictions[,1])
	ylim.range.rbm <- range(ylim.range.rbm, predictions[,2])
	cat(sprintf("RBM-4-%s: %.5f, %.5f; %.5f, %.5f\n", iter, xlim.range.rbm[1], xlim.range.rbm[2], ylim.range.rbm[1], ylim.range.rbm[2]))
	# Plot
	plot.mnist(model = dbn, x = mnist$test$x, label = mnist$test$y+1, predictions = predictions, reconstructions = reconstructions, 
	                   ncol = 16, highlight.digits = highlight.digits,
			   xlim = xlim.range.rbm, ylim = ylim.range.rbm)
	par(family="mono")
	legend("bottomleft", legend = sprintf("Mean error = %.3f", iteration.error), bty="n", cex=3)
	legend("bottomright", legend = sprintf("Iteration = %s", iter), bty="n", cex=3)
	dev.off()
}

# Fine-tuning
xlim.range.dbn <- xlim.range.rbm
ylim.range.dbn <- ylim.range.rbm
for (file in list.files(output.folder, pattern = "dbn-finetune-.+\\.RData", full.names = TRUE)) {
	load(file)
	iter <- stringr::str_match(file, "video/dbn-finetune-(.+)\\.RData")[,2]
	#png(sub(".RData", ".png", file), width = 1920, height = 1080) # full hd output
	png(sub(".RData", ".png", file), width = 1280, height = 720) # hd output
	predictions <- predict(dbn, mnist$test$x)
	reconstructions <- reconstruct(dbn, mnist$test$x)
	iteration.error <- errorSum(dbn, mnist$test$x) / nrow(mnist$test$x)
	# Calculate new range
	xlim.range.dbn <- range(xlim.range.dbn, predictions[,1])
	ylim.range.dbn <- range(ylim.range.dbn, predictions[,2])
	cat(sprintf("DBN-finetune-%s: %.5f, %.5f; %.5f, %.5f\n", iter, xlim.range.dbn[1], xlim.range.dbn[2], ylim.range.dbn[1], ylim.range.dbn[2]))
	# Plot
	plot.mnist(model = dbn, x = mnist$test$x, label = mnist$test$y+1, predictions = predictions, reconstructions = reconstructions,
	                   ncol = 16, highlight.digits = highlight.digits,
			   xlim = xlim.range.dbn, ylim = ylim.range.dbn)
	par(family="mono")
	legend("bottomleft", legend = sprintf("Mean error = %.3f", iteration.error), bty="n", cex=3)
	legend("bottomright", legend = sprintf("Iteration = %s", iter), bty="n", cex=3)
	dev.off()
}
