#' @title Sampling Methods for Deep Belief Nets and Restricted Bolzman Machines
#' @description Sample from a \code{\link{DeepBeliefNet}} or \code{\link{RestrictedBolzmannMachine}} object
#' @name resample
#' @aliases resample.DeepBeliefNet
#' @param object the model
#' @param newdata a \code{\link{data.frame}} or \code{\link{matrix}} providing the data to sample. Must have the same columns than the input layer of the model.
#' @param drop do not return additional dimensions
#' @param \dots ignored
#' @examples
#' library(mnist)
#' data(mnist)
#' ## Make predictions on a DBN object
#' data(pretrained.mnist)
#' res <- resample(pretrained.mnist, mnist$test$x)
#' dim(res)
#' head(res)
#' # Contrast with predict
#' pred <- predict(pretrained.mnist, mnist$test$x)
#' dim(pred)
#' head(pred)
#' @export
resample <- function(...)
	UseMethod("resample")

#' @rdname resample
#' @export
resample.DeepBeliefNet <- function(object, newdata, drop=TRUE, ...) {
	# Make sure C++/RcppEigen can deal with the data
	ensure.data.validity(newdata, object[[1]]$input)
	
	if (drop)
		return(drop(sampleDbnCpp(object, newdata)))
	else
		return(sampleDbnCpp(object, newdata))
}


#' @rdname resample
#' @examples
#' 
#' ## Sample an RBM object
#' rbm <- pretrained.mnist[[1]]
#' res <- resample(rbm, mnist$test$x)
#' dim(res)
#' @export
resample.RestrictedBolzmannMachine <- function(object, newdata, drop=TRUE, ...) {
	# Make sure C++/RcppEigen can deal with the data
	ensure.data.validity(newdata, object$input)
	
	if (drop)
		return(drop(sampleRbmCpp(object, newdata)))
	else
		return(sampleRbmCpp(object, newdata))
}