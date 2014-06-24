#' @title Combine DeepBeliefNets, RestrictedBolzmannMachines and Layers into a DeepBeliefNet.
#' @name c
#' @aliases c.DeepBeliefNet c.RestrictedBolzmannMachine, c.Layers
#' 
#' @description This function combines one or more DeepBeliefNets, RestrictedBolzmannMachines and Layers into a single DeepBeliefNets.
#' 
#' The layers must be compatible in order, i.e. the output of the previous DBN/RBM must be the same (both in term of size and type) than the input of the next one.
#' The only exception is with \code{\link{Layer}} objects, where \code{\link{RestrictedBolzmannMachine}}s will be created before and after the layer 
#' (only one \code{\link{RestrictedBolzmannMachine}} is created \code{\link{Layer}} succeeding immediately an other \code{\link{Layer}}).
#' 
#' @param ... objects to be combined
#' @param biases.first whether to use the biases of RBM i (TRUE) or i+1 (FALSE) for shared layers. Defaults to TRUE as we don't care much about the Bs after the pre-training.
#' @examples
#' # DeepBeliefNets only
#' dbn1 <- DeepBeliefNet(Layers(c(784, 1000), input="continuous", output="binary"))
#' dbn2 <- DeepBeliefNet(Layers(c(1000, 500, 250), input="binary", output="binary"))
#' dbn <- c(dbn1, dbn2)
#' 
#' # RestrictedBolzmannMachines only
#' dbn <- c(dbn1[[1]], dbn2[[1]], dbn2[[2]])
#' 
#' # Layers only
#' dbn <- c(Layer(784, "continuous"), Layer(1000, "binary"), Layer(500, "gaussian"))
#' 
#' # Layers only
#' c(Layer(784, "continuous"), Layer(1000, "binary"), Layer(500, "binary"))
#' 
#' # Mixing it all
#' rbm3 <- RestrictedBolzmannMachine(Layer(250, "binary"), Layer(30, "binary"))
#' layer4 <- Layer(2, "gaussian")
#' c(Layer(2, "gaussian"), dbn1, dbn2, rbm3, layer4)
#' 
#' # The following won't work
#' \dontrun{
#' dbn3 <- DeepBeliefNet(Layers(c(250, 500), input="binary", output="binary"))
#' dbn <- c(dbn1, dbn3)
#' dbn4 <- DeepBeliefNet(Layers(c(1000, 500), input="continuous", output="binary"))
#' dbn <- c(dbn1, dbn4)
#' }
#' @return the combined DBN. Note that it is not tagged as unrolled and fine-tuned any more. It is tagged as pre-trained if all individual DBNs/RBMs were pre-trained.
#' @importFrom methods is
#' @export
c.DeepBeliefNet <- function(..., biases.first = TRUE) {
	objects <- list(...)
	
	previous.object.is.layer <- FALSE # Then we need to create an RBM
	previous.layer <- NULL
	
	layers <- list()
	weights <- numeric(0)
	
	#Which weights do we extract (with getWeightsFromEnv) from each Layer
	if (biases.first) {
		which.weights <- c("all", rep("wc", 2))
	}
	else {
		which.weights <- c(rep("bw", 2), "all")
	}
	
	for (nobject in seq_along(objects)) {
		object <- objects[[nobject]]
		if (is(object, "DeepBeliefNet")) {
			if (previous.object.is.layer) {
				rbm <- RestrictedBolzmannMachine(previous.layer, object$layers[[1]])
				weights <- c(weights, getWeightsFromEnv(rbm$weights.env, ifelse(nobject == 2, which.weights[1], which.weights[2]), rbm$weights.breaks))
				layers <- c(layers, list(object$layers[[1]]))
				previous.object.is.layer <- FALSE
			}
			else if (nobject == 1) {
				layers <- c(layers, list(object$layers[[1]]))
			}
			else {
				ensureCompatibleLayers(previous.layer, object$layers[[1]])
			}
			for (nrbm in length(object$rbms)) {
				rbm <- object$rbms[[nrbm]]
				take.which.weights <- if (nobject == 1 && nrbm == 1) 1 else if (nobject == length(objects) && nrbm == length(object$rbms)) 3 else 2
				weights <- c(weights, getWeightsFromEnv(rbm$weights.env, which.weights[take.which.weights], rbm$weights.breaks))
				layers <- c(layers, list(rbm$output))
			}
			previous.layer <- object$layers[[length(object$layers)]]
		}
		else if (is(object, "RestrictedBolzmannMachine")) {
			if (previous.object.is.layer) {
				rbm <- RestrictedBolzmannMachine(previous.layer, object$input)
				weights <- c(weights, getWeightsFromEnv(rbm$weights.env, ifelse(nobject == 2, which.weights[1], which.weights[2]), rbm$weights.breaks))
				layers <- c(layers, list(object$input))
				previous.object.is.layer <- FALSE
			}
			else if (nobject == 1) {
				layers <- c(layers, list(object$input))
			}
			else {
				ensureCompatibleLayers(previous.layer, object$input)
			}
			take.which.weights <- if (nobject == 1) 1 else if (nobject == length(objects)) 3 else 2
			weights <- c(weights, getWeightsFromEnv(object$weights.env, which.weights[take.which.weights], object$weights.breaks))
			layers <- c(layers, list(object$output))
			previous.layer <- object$output
		}
		else if (is(object, "Layer")) {
			layers <- c(layers, list(object))
			if (previous.object.is.layer || nobject > 1) {
				rbm <- RestrictedBolzmannMachine(previous.layer, object)
				take.which.weights <- if (length(objects) == 2) "all" else if (nobject == 2) which.weights[1] else if (nobject == length(objects)) which.weights[3] else which.weights[2]
				weights <- c(weights, getWeightsFromEnv(rbm$weights.env, take.which.weights, rbm$weights.breaks))
			}
			previous.layer <- object
			previous.object.is.layer <- TRUE
		}
		else {
			stop("all objects must be of class DeepBeliefNet, RestrictedBolzmannMachine or Layer")
		}
	}
	
	return(DeepBeliefNetFromLayersAndOptionalWeights(layers, weights))
	
}

#### Aliases for c.DeepBeliefNet
#' @export
c.RestrictedBolzmannMachine <- c.DeepBeliefNet
#' @export
c.Layer <- c.DeepBeliefNet


# Stops with a helpful error message if the layers are incompatible. To be used whenever DBN/RBM objects get stacked
ensureCompatibleLayers <- function(layer1, layer2) {
	if (layer1 != layer2) {
		stop(sprintf('incompatible layers cannot be stacked: (%d-nodes %s, %d-nodes %s)', layer1$size, layer1$type, layer2$size, layer2$type))
	}
}
