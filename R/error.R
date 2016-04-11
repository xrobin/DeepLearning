#' @title Compute Reconstruction Error
#' @description Computes the reconstruction error (rmse) of the prediction of the data. \code{rmse} is an alias for \code{error}.
#' @param x the \code{\link{RestrictedBolzmannMachine}} or \code{\link{DeepBeliefNet}} object
#' @param data feature vector
#' @param ... further arguments to the \code{plot} function above and to the \code{predict} function.
#' @return the reconstruction error of the data
#' @export
error <- function(...)
	UseMethod("error")


#' @rdname error
#' @export
rmse <- function(...)
	UseMethod("error")


#' @rdname error
#' @export
error.DeepBeliefNet <- function(x, data, ...) {
	ensure.data.validity(data, x[[1]]$input)
	return(errorDbnCpp(x, data))
}


#' @rdname error
#' @export
rmse.DeepBeliefNet <- error.DeepBeliefNet


#' @rdname error
#' @export
errorSum.DeepBeliefNet <- function(x, data, ...) {
	ensure.data.validity(data, x[[1]]$input)
	return(errorSumDbnCpp(x, data))
}


#' @rdname error
#' @export
error.RestrictedBolzmannMachine <- function(x, data, ...) {
	ensure.data.validity(data, x$input)
	return(errorRbmCpp(x, data))
}


#' @rdname error
#' @export
rmse.RestrictedBolzmannMachine <- error.RestrictedBolzmannMachine


	return(errorSumRbmCpp(x, data))
