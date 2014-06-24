validLayerTypes <- c("binary", "continuous", "gaussian")

#' @title Layer of a DeepBeliefNet
#' 
#' @description This class represents a layer of the DeepBeliefNet.
#' 
#' \code{Layer} creates and returns a single layer of class \code{layer}, while \code{Layers} returns a list of \code{Layer}s of the given sizes, input and output. Hidden (or internal) layers are of type \dQuote{binary} by default.
#' 
#' @param size,sizes the number(s) of nodes of the Layer(s). Will be truncated to integers.
#' @param type,input,output,hidden the type of the Layer, or for Layer the types of input, output and hidden layers. If omitted, \code{hidden}
#' layers will be implicitly assigned a \dQuote{binary} type. \code{input} and \code{output} layer types must be specified explicitly.
#' 
#' @return an object of class \code{layer} containing the following elements:
#' \itemize{
#' \item{size}{number of nodes}
#' \item{type}{types of the nodes. Valid types are \dQuote{binary}, \dQuote{continuous} or \dQuote{gaussian}}
#' }
#' 
#' @examples
#' Layer(10L, "gaussian")
#' @aliases Layer Layers
#' @export
Layer <- function(size, type) {
	type <- match.arg(type, choices=validLayerTypes)
	# Check validity
	if (length(size) != 1) {
		stop(sprintf("Size not of length 1: %d", length(size)))
	}
	else if (size < 1) {
		stop(sprintf("Negative or null size: %d", size))
	}

	#### Code dependency ####
	#### In case the following is ever modified, also update src/RcppConversions -> template <> SEXP wrap(const Layer &layer) 
	Layer <- list(size = as.integer(size), type = type)
	class(Layer) <- "Layer"
	return(Layer)
}


#' @rdname Layer
#' @examples
#' Layers(c(10L, 5L, 2L), "gaussian", "gaussian")
#' @export
Layers <- function(sizes, input, output, hidden = "binary") {
	input <- match.arg(input, choices=validLayerTypes)
	output <- match.arg(output, choices=validLayerTypes)
	hidden <- match.arg(hidden, choices=validLayerTypes)
	sizes <- as.integer(sizes)
	
	if (length(sizes) < 2) {
		stop("sizes must be of length 2 or more")
	}

	layersList <- list(Layer(size = sizes[1], type = input))
	for (i in seq_along(sizes)[-1]) {
		layersList[[i]] <- Layer(size = sizes[i], type = hidden)
	}
	layersList[[length(sizes)]] <- Layer(size = sizes[length(sizes)], type = output)

	return(layersList)
}