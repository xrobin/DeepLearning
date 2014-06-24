#' @title Print a Deep Belief Net
#' @name print
#' @param x the RestrictedBolzmannMachine or DeepBeliefNet object to be printed
#' @param ... ignored
#' @return Returns x invisibly
#' @examples 
#' dbn <- DeepBeliefNet(Layers(c(784, 1000, 500, 250, 30), input="continuous", output="gaussian"))
#' print(dbn)
#' @seealso \code{\link{RestrictedBolzmannMachine}},  \code{\link{DeepBeliefNet}}, \code{\link{print}}
#' @export
print.DeepBeliefNet <- function(x, ...) {
	# Assess status
	training.state <- "Initialized"
	if (x$finetuned)
		training.state <- "Fine-tuned"
	else if (x$unrolled)
		training.state <- "Unrolled"
	else if (x$pretrained)
		training.state <- "Pre-trained"
	
	cat("Deep Belief Network with ", length(x$layers), " layers (", length(x$rbms), " Restricted Bolzman Machines).\n", sep = "")
	# Get the classes of the layers. Take input of 1st and output of all (including 1st as we have 1 less RBM than layers)
	types <- c(x$rbms[[1]]$input$type, sapply(x$rbms, function(rbm) rbm$output$type))
	sizes <- c(x$rbms[[1]]$input$size, sapply(x$rbms, function(rbm) rbm$output$size))
	# Pad the layer sizes to match the classes and align properly
	layers <- sprintf(sprintf("%% %ii", nchar(types)), sizes)
	# Now cat all this
	cat(paste(layers, collapse = " -> "), "\n", sep="")
	cat(paste(types, collapse = " -> "), "\n", sep="")
	cat("Status: ", training.state, "\n", sep="")
	invisible(x)
}

#' @title Reverse a Deep Belief Net
#' @name rev
#' @description \code{rev} returns a reversed \code{DeepBeliefNet} object. Precisely, the output is converted into the input of the network, and conversely.
#' @param x the DeepBeliefNet to reverse
#' @return the reversed DeepBeliefNet
#' @seealso \code{\link{RestrictedBolzmannMachine}}, \code{\link{DeepBeliefNet}}, \code{\link{rev}}
#' @export
rev.DeepBeliefNet <- function(x) {
	return(reverseDbnCpp(x))
}

#' @title Extract or Replace Parts of a Deep Belief Net
#' @name Extract
#' @aliases [[.DeepBeliefNet [[
#' @description Operators to extract or replace parts of a \code{\link{DeepBeliefNet}} object.
#' In extraction methods, the \code{pretrained} and \code{finetuned} switches will match those of the DeepBeliefNet that was supplied, while \code{unrolled} will be set to \code{FALSE}.
#' In replacement methods, the \code{pretrained} switch will be on if all the \code{RestrictedBolzmannMachines} are pretrained, while \code{finetuned} and \code{unrolled} will be set to \code{FALSE}.
#' @param x the DBN
#' @param i indices specifying elements to extract or replace
#' @param value the RestrictedBolzmannMachine to insert.
#' @param drop whether to drop the DeepBeliefNet if it contains a single RestrictedBolzmannMachine.
#' @details
#' \code{[[} extracts (and \code{[[<-} replaces) \emph{exactly} one \code{\link{RestrictedBolzmannMachine}} layer of the DeepBeliefNet.
#' 
#' \code{[} extracts one or more layers of the DeepBeliefNet. If the returned DeepBeliefNet. has exactly one RestrictedBolzmannMachine, the \code{drop} 
#' argument controls whether the function returns an \code{\link{RestrictedBolzmannMachine}} object (\code{TRUE})
#' or a \code{\link{DeepBeliefNet}} object containing one single \code{\link{RestrictedBolzmannMachine}} (\code{FALSE}).
#' 
#' @section Note:
#' If the DeepBeliefNet contains N layers, there are N-1 RestrictedBolzmannMachines.
#' @examples
#' dbn <- DeepBeliefNet(Layers(c(784, 1000, 500, 250, 30), input="continuous", output="gaussian"))
#' # Extract a RBM
#' dbn[[2]]
#' 
#' @seealso \code{\link{DeepBeliefNet}}, \code{\link{RestrictedBolzmannMachine}}
#' @export
"[[.DeepBeliefNet" <- function(x, i) {
	if (any(i > length(x$rbms)))
		stop("subscript out of bounds")
	rbm <- x$rbms[[i]]
	return(clone(x$rbms[[i]]))
}

#' @rdname Extract
#' @usage \method{[[}{DeepBeliefNet} (x, i) <- value
#' @aliases [[<-
#' @examples
#' # Replace a RBM
#' dbn[[1]] <- RestrictedBolzmannMachine(Layer(10, "binary"), Layer(1000, "binary"))
#' dbn[[2]] <- RestrictedBolzmannMachine(Layer(1000, "binary"), Layer(500, "binary"))
#' dbn[[4]] <- RestrictedBolzmannMachine(Layer(250, "binary"), Layer(2, "gaussian"))
#' \dontrun{
#' # Cannot replace incompatible RestrictedBolzmannMachines
#' dbn[[2]] <- RestrictedBolzmannMachine(1000, 400, input="binary", output="binary")
#' dbn[[2]] <- RestrictedBolzmannMachine(100, 500, input="binary", output="binary")
#' dbn[[2]] <- RestrictedBolzmannMachine(1000, 500, input="binary", output="continuous")
#' dbn[[2]] <- RestrictedBolzmannMachine(1000, 500, input="gaussian", output="binary")
#' }
#' @importFrom methods is
#' @export
"[[<-.DeepBeliefNet" <- function(x, i, value) {
	if (length(i) > 1)
		stop("i must be of length 1")
	if (i > length(x$rbms))
		stop("subscript out of bounds")
	if (!is(value, "RestrictedBolzmannMachine"))
		stop("value must be an object of class RestrictedBolzmannMachine")

	if (i < length(x$rbms))
		ensureCompatibleLayers(x$rbms[[i+1]]$input, value$output)
	if (i > 1)
		ensureCompatibleLayers(x$rbms[[i-1]]$output, value$input)

	weights <- numeric(0) # Update weights
	r <- 1
	while (r < i) { # We have at least one RBM before the insert point. Take its b and W
		weights <- c(weights, getWeightsFromEnv(x$rbms[[r]]$weights.env, "bw", x$rbms[[r]]$weights.breaks))
		r <- r + 1
	}

	weights <- c(weights, getWeightsFromEnv(value$weights.env, "all", value$weights.breaks)) # Weights of the new RBM

	r <- i + 1
	while (r <= length(x$rbms)) { # We have at least one RBM before the insert point. Take its b and W
		weights <- c(weights, getWeightsFromEnv(x$rbms[[r]]$weights.env, "wc", x$rbms[[r]]$weights.breaks))
		r <- r + 1
	}
	
	x$layers[[i]] <- value$input
	x$layers[[i+1]] <- value$output
	
	weights.breaks <- computeDbnBreaks(x$layers)
	weights.env <- packDbnEnv(weights.breaks, weights)
	
	x$rbms[[i]] <- value
	x$weights.env <- weights.env

	for (r in 1:length(x$rbms)) {
		x$rbms[[r]]$weights.env <- weights.env
		x$rbms[[r]]$weights.breaks <- weights.breaks[(2 * r - 1):(2 * r + 2)]
	}
	
	x$pretrained <- all(sapply(x$rbms, function(rbm) rbm$pretrained))
	x$unrolled <- FALSE
	x$finetuned <- FALSE
	
	x
}


#' @rdname Extract
#' @aliases [
#' @examples
#' # Get the first layer as RestrictedBolzmannMachine
#' dbn[[1]]
#' dbn[1, drop=TRUE]
#' # Get the first layer as DeepBeliefNet
#' dbn[1]
#' @export
"[.DeepBeliefNet" <- function(x, i, drop = FALSE) {
	if (missing(i)) # If no i is provided, return the whole dbn
		i <- seq_along(x$rbms)
	if (length(i) < 1) 
		stop("attempt to select less than one layer")
	if (!is.integer(i) && any(abs(i - round(i)) > .Machine$double.eps^0.5))
		stop("i must be an integer")
	if (any(i > length(x$rbms)) || any(i == 0))
		stop("subscript out of bounds")
	
	# Support negative indices: transform as positive
	i_pos <- seq_along(x$rbms)[i]
	
	# drop?
	if (drop && length(i_pos) == 1)
		return(x[[i_pos]])
	
	# Grab the requested RBMs
	rbms <- lapply(i_pos, function(i) x$rbms[[i]])
	dbn <- do.call("c.RestrictedBolzmannMachine", rbms)
	dbn$pretrained <- x$pretrained
	dbn$finetuned <- x$finetuned
	dbn$unrolled <- FALSE
	
	return(dbn)

}

#' @aliases drop.DeepBeliefNet
#' @export
# See generics.R for the rest of the documentation
drop.DeepBeliefNet <- function(x) {
	if (length(x$rbms) == 1)
		return(x[[1]]) # x[[ does clone already
	else
		return(clone(x))
}

#' @title Length (or depth) of a DeepBeliefNet
#' @name length
#' @description Computes the number of layers of the DeepBeliefNet. Note that this is not the same
#' as the number of RestrictedBolzmannMachine contained in the DeepBeliefNet (which is length - 1)
#' @param x DeepBeliefNet object
#' @examples
#' dbn <- DeepBeliefNet(Layers(c(784, 1000, 500, 250, 30), input="continuous", output="gaussian"))
#' length(dbn)
#' @export
length.DeepBeliefNet <- function(x) {
	length(x$layers)
}
