#' @title Pre-trains the DeepBeliefNet or RestrictedBolzmannMachine. A contrastive divergence method is used to train each layer sequentially.
#' @param x the \code{\link{DeepBeliefNet}} or \code{\link{RestrictedBolzmannMachine}} object
#' @param data the dataset, either as matrix or data.frame. The number of columns must match the number of nodes of the network input
#' @param miniters,maxiters minimum and maximum number of iterations to perform
#' @param batchsize the size of the minibatches
#' @param skip numeric vector of the RestrictedBolzmannMachine of the DeepBeliefNet to be skipped.
#' @param momentum the momentum, between 0 (no momentum) and 1 (no training). See the Momentums section below.
#' @param penalization the penalization mode. Either \dQuote{l1} (sparse), \dQuote{l2} (quadratic) or \dQuote{none}.
#' @param lambda penalty on large weights (weight-decay). Alternatively one can define \code{lambda.b}, \code{lambda.c} and \code{lambda.W} to constrain 
#' \code{b}s, \code{c}s and \code{W}s, respectively. Default: 0 = no penalization (equivalent to \code{penalization="none"}).
#' @param lambda.b,lambda.c,lambda.W separate penalty rates for \code{b}s, \code{c}s and \code{W}s. Take precedence over \code{lambda}.
#' @param epsilon learning rate. Alternatively one can define \code{epsilon.b}, \code{epsilon.c} and \code{epsilon.W} (see below)
#' to learn \code{b}s, \code{c}s and \code{W}s, respectively, at different speeds. Defaut: 0.1 (for layers where all inputs and outputs are binary or continuous)
#'  or 0.001 (for layers with gaussian input or output).
#' @param epsilon.b,epsilon.c,epsilon.W separate learning rates for \code{b}s, \code{c}s and \code{W}s. Take precedence over \code{epsilon}.
#' @param train.b,train.c whether (\code{\link{RestrictedBolzmannMachine}}) or on which layers (\code{\link{DeepBeliefNet}}) to update the \code{b}s and \code{c}s. For a \code{\link{RestrictedBolzmannMachine}}, must be a logical of length 1. For a \code{\link{DeepBeliefNet}} must be a logical (can be recycled) or numeric index of layers.
#' @param diag,diag.rate,diag.data diagnmostic specifications.
#' @param n.proc number of cores to be used for Eigen computations
#' @param ... ignored
#' @section Pretraining Layers of the Deep Belief Net with Different Parameters:
#' It is possible to pre-train the layers of a DeepBeliefNet with different parameters. The following parameters can be supplied as vectors with length of the network - 1:
#' \code{batchsize}, \code{penalization}, \code{labmda}, \code{lambda.b}, \code{lambda.c}, \code{lambda.W}, 
#' \code{epsilon}, \code{epsilon.b}, \code{epsilon.c} and \code{epsilon.W}.
#' The values will be recycled if necessary (with essentially no warning if the lengths doesn't match). The special case of the \code{momentum} parameters is described below.
#' @section Momentums:
#'  The \code{momentum} parameter can take several length, and will be interpreted accordingly:
#' \itemize{
#' \item \code{1}: constant momentum
#' \item \code{2}: a gradient, will be interpreted as seq(momentum[1], momentum[2], length.out=maxiters)
#' \item \code{maxiter}: encodes the momentum per iteration
#' }
#' To specify different \code{momentum}s for the different layers of a DeepBeliefNet, they must be passed as a \code{\link{list}} of the same length than the number
#' of RestrictedBolzmannMachines to pretrain,
#' and they will be interpreted per layer as described above.
#' 
#' @return pre-trained object with the \code{pretrained} switch set to \code{TRUE}.
#' @export
pretrain <- function(x, data, ...)
	UseMethod("pretrain", x)


#' @rdname pretrain
#' @export
pretrain.RestrictedBolzmannMachine <- function(x, data, miniters = 100, maxiters = floor(dim(data)[1] / batchsize), batchsize = 100, 
						 momentum = 0, penalization = c("l1", "l2", "none"),
						 lambda = 0, lambda.b = lambda, lambda.c = lambda, lambda.W = lambda,
						 epsilon = ifelse(x$output$type == "gaussian", 0.001, 0.1), epsilon.b = epsilon, epsilon.c = epsilon, epsilon.W = epsilon,
						 train.b = TRUE, train.c = TRUE,
						 continue.function = continue.function.exponential, continue.function.frequency = 100, continue.stop.limit = 3,
						 diag = list(rate = diag.rate, data=diag.data), diag.rate = c("none", "each", "accelerate"), diag.data = NULL,
						 n.proc = detectCores() - 1, ...) {
	sample.size <- nrow(data)
	
	# Check for ignored arguments
	ignored.args <- names(list(...))
	if (length(ignored.args) > 0) {
		warning(paste("The following arguments were ignored in pretrain.RestrictedBolzmannMachine:", paste(ignored.args, collapse=", ")))
	}
	
	# Validate and prepare the momentums, learning rates and penalizations
	momentum <- make.momentum(momentum, maxiters)
	if (is.null(lambda.b))
		lambda.b <- 0
	if (is.null(lambda.c))
		lambda.c <- 0
	if (is.null(lambda.W))
		lambda.W <- 0
	if (is.null(epsilon.b))
		epsilon.b <- ifelse(x$output$type == "gaussian", 0.001, 0.1)
	if (is.null(epsilon.c))
		epsilon.c <- ifelse(x$output$type == "gaussian", 0.001, 0.1)
	if (is.null(epsilon.W))
		epsilon.W <- ifelse(x$output$type == "gaussian", 0.001, 0.1)
	
	penalization <- match.arg(penalization)
	
	# Build diagnostic function
	if (is.null(diag.data) && is.null(diag.function)) {
		diag$rate <- "none"
	}
	else {
		diag$rate <- match.arg(diag$rate, c("none", "each", "accelerate"))
	}
	
	# Build continue function
	continue.function <- list(
		continue.function = continue.function,
		continue.function.frequency = continue.function.frequency,
		continue.stop.limit = continue.stop.limit
	)

	ensure.data.validity(data, x$input)

	pretrainParams <- list(
		maxiters = maxiters, miniters = miniters, batchsize = batchsize,
		momentum = momentum, penalization = penalization,
		lambda.b = lambda.b, lambda.c = lambda.c, lambda.W = lambda.W,
		epsilon.b = epsilon.b, epsilon.c = epsilon.c, epsilon.W = epsilon.W,
		train.b = train.b, train.c = train.c,
		n.proc = n.proc)
	ret <- pretrainRbmCpp(x, data, pretrainParams, diag, continue.function)

# Below is a block of legacy pre-c++ code that we can probably safely remove.
# 		# Execute the diag function
# 		# Random choice of data points to start with. Sample with replacement when maxiters > sample.size
# 		s <- matrix(sample(1:sample.size, size=maxiters*batchsize, replace = maxiters*batchsize > sample.size), nrow = maxiters)
# 
# 		# Prepare the weight increments for momentums
# 		x$Winc <- x$W - x$W
# 		x$bInc <- x$b - x$b
# 		x$cInc <- x$c - x$c
# 		
# 		# Actually do the training
# 		for (i in 1:(maxiters)) {
# 			batch <- s[i,]
# 			#batch <- s[(((i-1)*batchsize)+1):(i*batchsize)]
# 			x <- rbm.update.batch(x, data[batch, ], momentum = momentum[i], penalization = penalization,
# 								  lambda.b = lambda, lambda.c = lambda, lambda.W = lambda,
# 								  epsilon.b = epsilon, epsilon.c = epsilon, epsilon.W = epsilon)
# 			if (is.function(diag.function)) {
# 				ll <- list(x = x, batch = i, step = "pretrain")
# 				do.call(diag.function, c(ll, diag.args))
# 			}
# 		}
# 		
# 		# Remove the momentum weights
# 		x$Winc <- NULL
# 		x$bInc <- NULL
# 		x$cInc <- NULL
#
#	# Return the pretrained object
#	x$pretrained <- TRUE

	return(ret)
}

#' @rdname pretrain
#' @export
pretrain.DeepBeliefNet <- function(x, data, 
						 # Arguments for rbm.pretrain(_with_diagnostics)
						 miniters = 100, maxiters = floor(dim(data)[1] / batchsize), batchsize = 100,
						 skip = numeric(0),
						 # Arguments for rbm.update
						 momentum = 0, penalization = "l1", 
						 lambda = 0.0002, lambda.b = lambda, lambda.c = lambda, lambda.W = lambda,
						 epsilon = 0.1, epsilon.b = epsilon, epsilon.c = epsilon, epsilon.W = epsilon,
						 train.b = TRUE, train.c = length(x) - 1,
						 continue.function = continue.function.exponential, continue.function.frequency = 100, continue.stop.limit = 3,
						 diag = list(rate = diag.rate, data = diag.data, f = diag.function), diag.rate = c("none", "each", "accelerate"), diag.data = NULL, diag.function = NULL,
						 n.proc = detectCores() - 1,
						 ...) {
	sample.size <- dim(data)[1]
	
	# Check for ignored arguments
	ignored.args <- names(list(...))
	if (length(ignored.args) > 0) {
		warning(paste("The following arguments were ignored in pretrain.DeepBeliefNet:", paste(ignored.args, collapse=", ")))
	}
	
	ensure.data.validity(data, x[[1]]$input)
	
	# What layers to train?
	train.layers <- seq_along(x$rbms)
	if (length(skip) > 0) {
		skip <- as.integer(skip)
		# Ensure skipping only layers that exist
		if (!all(skipped <- skip %in% train.layers)) {
			stop(paste("Invalid skip values:", paste(skip[!skipped], collapse=", ")))
		}
		train.layers <- train.layers[-skip]
	}
	
	# Make arguments of length #layers to train
	len <- length(train.layers)
	
	# Make sure penalization is a character, not a factor or numeric:
	penalization <- as.character(penalization)
	
	# Fix expilon - default by layer type
	#if (is.null(epsilon)) {
	#	gaussian.layers <- sapply(x$rbms, function(rbm) {return(rbm$input$type == "gaussian" || rbm$output$type == "gaussian")})
	#	epsilon <- ifelse(gaussian.layers, 0.001, 0.1) # This automatically sets epsilon.b .c and .W if they were missing!
	#}
	
	if (is.numeric(train.b)) {
		train.b <- seq_len(length(x) - 1) %in% train.b
	}
	if (is.numeric(train.c)) {
		train.c <- seq_len(length(x) - 1) %in% train.c
	}
	
	parameters <- data.frame(
		miniters = rep(miniters, length.out = len),
		maxiters = rep(maxiters, length.out = len),
		batchsize = rep(batchsize, length.out = len),
		penalization = rep(sapply(penalization, match.arg, choices = c("l1", "l2", "none"), several.ok=TRUE), length.out = len),
		lambda.b = rep(lambda.b, length.out = len),
		lambda.c = rep(lambda.c, length.out = len),
		lambda.W = rep(lambda.W, length.out = len),
		epsilon.b = rep(epsilon.b, length.out = len),
		epsilon.c = rep(epsilon.c, length.out = len),
		epsilon.W = rep(epsilon.W, length.out = len),
		train.b = rep(train.b, length.out = len),
		train.c = rep(train.c, length.out = len),
		stringsAsFactors = FALSE
	)
	
	# Build diagnostic function
	if (is.null(diag.data) && is.null(diag.function)) {
		diag$rate <- "none"
	}
	else {
		diag$rate <- match.arg(diag$rate, c("none", "each", "accelerate"))
	}
	
	# Build continue function
	continue.function <- list(
		continue.function = continue.function,
		continue.function.frequency = continue.function.frequency,
		continue.stop.limit = continue.stop.limit
	)
	
	if (is.list(momentum)) {
		if (length(momentum) != len) {
			stop("If 'momentum' is be a list, it must be of length equal to the number of RestrictedBolzmannMachine to train in the network.")
		}
		for (i in seq_along(momentum)) {
			momentum[[i]] <- make.momentum(momentum[[i]], parameters$maxiters[i])
		}
		parameters$momentum <- momentum
	}
	else {
		parameters$momentum <- sapply(len, function(i) make.momentum(momentum, maxiters), simplify=FALSE)
	}
	
	if (length(skip) > 0) {
		# If we skip layers, we'll have too few parameters - put them in the right place!
		parameters[train.layers,] <- parameters
		rownames(parameters) <- seq_along(x$rbms)
	}

	parameters <- split(parameters, rownames(parameters))
	
	pretrained <- pretrainDbnCpp(x, data, parameters, diag, continue.function, skip)

	return(pretrained)
}

make.momentum <- function(momentum, maxiters) {
	if (length(momentum) == 1)
		return(rep(momentum, maxiters))
	else if (length(momentum) == 2)
		return(seq(momentum[1], momentum[2], length.out=maxiters))
	else if (length(momentum) == maxiters)
		return(momentum)
	else
		stop("momentum must be of length 1, 2 or maxiters")
}
