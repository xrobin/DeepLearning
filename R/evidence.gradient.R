#' @title Computes and returns the gradient of the evidence function
#' @param x the \code{\link{RestrictedBolzmannMachine}} object
#' @param data feature vector
#' @return the reconstruction error of the data
#' @export
evidence.gradient <- function(...)
	UseMethod("evidence.gradient")


#' @rdname error
#' @export
evidence.gradient.RestrictedBolzmannMachine <- function(x, data, ...) {
	stop("The evidence gradient is available only during pre-training for now.")
	ensure.data.validity(data, x$input)
	return(evidenceGradientRbmCpp(x, data))
}