#' @title Energy of the Deep Belief Net or Restricted Bolzmann Machine.
#' @description Computes the energy of the data points in the DeepBeliefNet or RestrictedBolzmannMachine
#' @param x the \code{\link{DeepBeliefNet}} or \code{\link{RestrictedBolzmannMachine}} object
#' @param data the dataset, either as matrix or data.frame. The number of columns must match the number of nodes of the network input
#' @param drop do not return additional dimensions
#' @param \dots ignored
#' @return a vector or matrix of the same size than the data (rows) giving the energy of each data point
#' @export
energy <- function(x, data, ...)
	UseMethod("energy")

#' @rdname energy
#' @examples
#' library(mnist)
#' data(mnist)
#'
#' # Calculate error per data point on RBM
#' data(pretrained.mnist)
#' rbm <- pretrained.mnist[[1]]
#' en <- energy(rbm, mnist$test$x[1:100,])
#' head(en) # 1 value per data point
#' 
#' @export
energy.RestrictedBolzmannMachine <- function(x, data, drop=TRUE, ...) {
	# Make sure C++/RcppEigen can deal with the data
	ensure.data.validity(data, x$input)
	
	if (drop)
		return(drop(energyRbmCpp(x, data)))
	else
		return(energyRbmCpp(x, data))
}

#' @rdname energy
#' @examples 
#' # Calculate error per data point on DBN
#' data(trained.mnist)
#' en <- energy(trained.mnist, mnist$test$x[1:100,])
#' head(en) # 1 value per data point
#' 
#' # Energy is not related with reconstruction error
#' err <- error(trained.mnist, mnist$test$x[1:100,])
#' plot.mnist(predictions = cbind(err, en))
#' @export
energy.DeepBeliefNet <- function(x, data, drop=TRUE, ...) {
	# Make sure C++/RcppEigen can deal with the data
	ensure.data.validity(data, x[[1]]$input)
	
	if (drop)
		return(drop(energyDbnCpp(x, data)))
	else
		return(energyDbnCpp(x, data))
}
	