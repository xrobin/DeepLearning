#' @title Compute Reconstruction Error
#' @description Computes the reconstruction error (rmse) of the prediction of the data. \code{rmse} is an alias for \code{error}. \code{errorSum} sum the error over the data points.
#' @param x the \code{\link{RestrictedBolzmannMachine}} or \code{\link{DeepBeliefNet}} object
#' @param data feature vector
#' @param ... further arguments to the \code{plot} function above and to the \code{predict} function.
#' @return the reconstruction error of the data
#' @examples 
#' library(mnist)
#' data(mnist)
#' data(trained.mnist)
#' 
#' # Calculate error per data point
#' err <- error(trained.mnist, mnist$test$x)
#' length(err) # 1 value per data point
#' # error and rmse are synonymous
#' identical(err, rmse(trained.mnist, mnist$test$x))
#' 
#' # errorSum returns the sum
#' sum <- errorSum(trained.mnist, mnist$test$x)
#' print(sum)
#' all.equal(sum, sum(err)) 
#' # There may be some rounding errors though, so this might not be ==:
#' sum == sum(err)
#' @export
error <- function(...)
	UseMethod("error")


#' @rdname error
#' @export
rmse <- function(...)
	UseMethod("error")


#' @rdname error
#' @export
errorSum <- function(...)
	UseMethod("errorSum")


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


#' @rdname error
#' @export
errorSum.RestrictedBolzmannMachine <- function(x, data, ...) {
	ensure.data.validity(data, x$input)
	return(errorSumRbmCpp(x, data))
}
