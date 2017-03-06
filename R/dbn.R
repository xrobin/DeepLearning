#' @title Initialize a Deep Belief Net
#' @description Creates a Deep Belief Net (DBN), precisely a \code{DeepBeliefNet} object, with the given specifications. 
#' It consists of a stack of \code{\link{RestrictedBolzmannMachine}} layers that will be created according to the specifications.
#' @param layers a single \code{\link{Layer}} object or a list of layers as returned by \code{\link{Layers}}
#' @param ... same as \code{layers}
#' @param initialize whether to initialize weights and biases with 0 or random uniform values
#' @section Copying/Cloning:
#' #' For performance purposes, the weights are stored in an environment. This means that when you copy the DeepBeliefNet with an assignment, you do not copy the weights
#' and any modification you make to the new object will be propagated to the original one, and reciprocally.
#' Use \code{\link{clone}} to control this and make a copy of the weights whenever you need it. Note that all the functions defined in the package do this by default.
#' 
#' @return an object of class \code{DeepBeliefNet} containing the following elements:
#' \itemize{
#' \item{layers: }{The layers of the RBM}
#' \item{rbms: }{a list of \code{\link{RestrictedBolzmannMachine}} objects making up the network.}
#' \item{pretrained, unrolled, finetuned: }{boolean switches to mark the state of the network.}
#' }
#' 
#' @seealso \code{\link{RestrictedBolzmannMachine}}, \code{\link{Layer}}
#' @examples
#' dbn <- DeepBeliefNet(Layers(c(784, 1000, 500, 250, 30), input="continuous", output="gaussian"))
#' # Identical as
#' dbn2 <- DeepBeliefNet(Layer(784, "continuous"), Layer(1000, "binary"), Layer(500, "binary"), 
#'                       Layer(250, "binary"), Layer(30, "gaussian"))
#' print(dbn)
#' methods(class="DeepBeliefNet")
#' @importFrom methods is
#' @importFrom utils tail
#' @export
DeepBeliefNet <- function(layers, ..., initialize = c("0", "uniform")) {
	initialize <- match.arg(initialize)
	# Merge all input in a single list
	if (is(layers, "Layer")) {
		layers <- list(layers)
	}
	layers <- c(layers, list(...))
	
	if (length(layers) < 2) {
		stop("You must have at least 2 layers to make a DBN.")
	}
	
	return(DeepBeliefNetFromLayersAndOptionalWeights(layers, initialize = initialize))
}

# Private constructor
# Takes a list as returned by Rcpp::wrap(DeepBeliefNet) into a valid R DeepBeliefNet
#DeepBeliefNetFromRcppList <- function(list) {
#	return(DeepBeliefNetFromLayersAndOptionalWeights(list$layers, list$weights))
#}

# Private constructor
# Build a valid DeepBeliefNet from layers and weights
# If weights is not provided, it will be created (see 'packDbnEnv')
DeepBeliefNetFromLayersAndOptionalWeights <- function(layers, weights, initialize) {
	
	# Compute where to break the weights - b, W, c
	weights.breaks <- computeDbnBreaks(layers)
	if (missing(weights)) {
		weights.env <- packDbnEnv(weights.breaks)
	}
	else {
		weights.env <- packDbnEnv(weights.breaks, weights)
		initialize <- "no"
	}
	
	rbms <- lapply(seq(1, length(layers) - 1), function(x) {
		# ifelse: force all internal layers to be binary
		RestrictedBolzmannMachineFromWeightsEnv(input = layers[[x]],
												output = layers[[x + 1]],
												weights.env = weights.env,
												weights.breaks = weights.breaks[(2 * x - 1):(2 * x + 2)],
												initialize = initialize
		)
	})
	dbn <- list(
		weights.env = weights.env,
		layers = layers,
		rbms = rbms,
		# Status
		pretrained = FALSE,
		unrolled = FALSE,
		finetuned = FALSE
	)
	class(dbn) <- "DeepBeliefNet"
	return(dbn)
}


# Private function that computes the weight breaks for a DBN from the layers
computeDbnBreaks <- function(layers) {
	as.integer(cumsum(c(0, sapply(seq(1, length(layers) - 1), function(x) c(layers[[x]]$size, layers[[x]]$size * layers[[x + 1]]$size)), tail(layers, n=1)[[1]]$size)))
}

# Private function that packs weights and breaks into a new environment.
# If no weight is given, creates a new vector of weights initialized at 0
packDbnEnv <- function(weights.breaks, weights = numeric(tail(weights.breaks, n=1))) {
	weights.env <- new.env(size=2)
	assign("weights", weights, pos=weights.env) # Set all weights to 0
	assign("breaks", weights.breaks, pos=weights.env) # Also store all the breaks there
	return(weights.env)
}
