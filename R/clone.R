
#' @title Clone a DBN or RBM
#' @description Clones (= makes a deep copy) of a RestrictedBolzmannMachine or DeepBeliefNet. This is necessary because the weights are stored in an environment which is shared,
#' so any modification you make to the new object will be propagated to the original one, and reciprocally. After cloning, the two objects are totally independent.
#' @param x the \code{\link{RestrictedBolzmannMachine}} or \code{\link{DeepBeliefNet}} object
#' @return a clone of \code{x} with weights stored in a new environment
#' @export
clone <- function(x)
	UseMethod("clone")

#' @rdname clone
#' @examples 
#' rbm <- rbm(Layer(784, "continuous"), Layer(1000, "binary"))
#' rbm2 <- clone(rbm)
#' @export
clone.RestrictedBolzmannMachine <- function(x) {
	rbm2 <- x
	new.weights.breaks <- x$weights.breaks - x$weights.breaks[1]
	
	# Create new env - with only the relevant weights
	new.weights.env <- new.env(size=2)

	assign("weights", getWeightsFromEnv(x$weights.env, which="all", breaks=x$weights.breaks), pos=new.weights.env) # Get only the relevant weights
	assign("breaks", new.weights.breaks, pos=new.weights.env) # Also store all the breaks there
	
	rbm2$weights.env <- new.weights.env
	rbm2$weights.breaks <- new.weights.breaks
	
	return(rbm2)
}

#' @rdname clone
#' @examples 
#' dbn <- DeepBeliefNet(Layers(c(784, 1000, 500, 250, 30), input="continuous", output="gaussian"))
#' dbn2 <- clone(dbn)
#' @export
clone.DeepBeliefNet <- function(x) {
	dbn2 <- x
	dbn2$weights.env <- as.environment(as.list(x$weights.env))
	return(dbn2)
}