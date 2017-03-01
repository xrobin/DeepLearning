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
#' @export
energy.RestrictedBolzmannMachine <- function(x, data, drop=TRUE, ...) {
	if (!(is.matrix(data) || is.data.frame(data))) {
		data <- t(data)
	}
	
	# Make sure C++/RcppEigen can deal with the data
	ensure.data.validity(data, x[[1]]$input)
	
	if (drop)
		return(drop(energyRbmCpp(x, data)))
	else
		return(energyRbmCpp(x, data))
}

#' @rdname energy
#' @export
energy.DeepBeliefNet <- function(x, data, drop=TRUE, ...) {
	if (!(is.matrix(data) || is.data.frame(data))) {
		data <- t(data)
	}
	
	# Make sure C++/RcppEigen can deal with the data
	ensure.data.validity(data, x[[1]]$input)
	
	if (drop)
		return(drop(energyDbnCpp(x, data)))
	else
		return(energyDbnCpp(x, data))
}
	