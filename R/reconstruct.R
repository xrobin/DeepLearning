#' @title Reconstruct data through a Deep Belief Nets and Restricted Bolzman Machines
#' @description Passes the data all the way through an unrolled DeepBeliefNet (in this case, it is identical to predict).
#' For a RestrictedBolzmannMachine or a DeepBeliefNet that hasn't been unrolled, it will predict, and predict again through the reversed network.
#' In the end, the reconstruction has the same dimension as the input.
#' @param object the \code{\link{RestrictedBolzmannMachine}} or \code{\link{DeepBeliefNet}} object
#' @param newdata a \code{\link{data.frame}} or \code{\link{matrix}} providing the data. Must have the same columns than the input layer of the model.
#' @param drop do not return additional dimensions
#' @param \dots ignored
#' @return the reconstructed data
#' @export
reconstruct <- function(object, newdata, ...)
	UseMethod("reconstruct")


#' @rdname reconstruct
#' @export
reconstruct.DeepBeliefNet <- function(object, newdata, drop=TRUE, ...) {
	# Make sure C++/RcppEigen can deal with the data
	ensure.data.validity(newdata, object[[1]]$input)
	
	if (drop)
		return(drop(reconstructDbnCpp(object, newdata)))
	else
		return(reconstructDbnCpp(object, newdata))
}

#' @rdname reconstruct
#' @export
reconstruct.RestrictedBolzmannMachine <- function(object, newdata, drop=TRUE, ...) {
	# Make sure C++/RcppEigen can deal with the data
	ensure.data.validity(newdata, object$input)
	
	if (drop)
		return(drop(reconstructRbmCpp(object, newdata)))
	else
		return(reconstructRbmCpp(object, newdata))
}