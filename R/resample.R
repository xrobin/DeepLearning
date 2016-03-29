#' @title Sampling Methods for Deep Belief Nets and Restricted Bolzman Machines
#' @name resample
#' @aliases resample.DeepBeliefNet
#' @description Sample from a \code{\link{DeepBeliefNet}} or \code{\link{RestrictedBolzmannMachine}} object
#' @param object the model
#' @param newdata a \code{\link{data.frame}} or \code{\link{matrix}} providing the data to sample. Must have the same columns than the input layer of the model.
#' @param drop do not return additional dimensions
#' @param \dots ignored
#' @examples
#' data(mnist)
#' ## Make predictions on a DBN object
#' dbn <- DeepBeliefNet(Layers(c(784, 1000, 500, 250, 30), input="continuous", output="gaussian"))
#' pretrained <- pretrain(dbn, mnist$train$x, 
#'                        penalization = "l2", lambda=0.0002, epsilon=c(.1, .1, .1, .001), 
#'                        batchsize = 100, maxiters=100000)
#' resample(pretrained, mnist$test$x)
#' @export
resample <- function(...)
	UseMethod("resample")

#' @rdname resample
#' @export
resample.DeepBeliefNet <- function(object, newdata, drop=TRUE, ...) {
	# If user passed a vector f or a data.frame, we convert it to a compatible matrix
	if (!(is.matrix(newdata) || is.data.frame(newdata))) {
		newdata <- t(newdata)
	}

	# Make sure C++/RcppEigen can deal with the data
	ensure.data.validity(newdata, object[[1]]$input)
	
	if (drop)
		return(drop(sampleDbnCpp(object, newdata)))
	else
		return(sampleDbnCpp(object, newdata))
}


#' @rdname resample
#' @examples
#' ## Sample an RBM object
#' rbm <- RestrictedBolzmannMachine(Nv = 784, Nh = 1000, input="continuous", output="binary")
#' pretrained <- pretrain(rbm, mnist$train$x)
#' resample(pretrained, mnist$test$x)
#' @export
resample.RestrictedBolzmannMachine <- function(object, newdata, drop=TRUE, ...) {

	# If user passed a vector f, we convert it to a compatible matrix
	if (!(is.matrix(newdata) || is.data.frame(newdata))) {
		newdata <- t(newdata)
	}

	# Make sure C++/RcppEigen can deal with the data
	ensure.data.validity(newdata, object$input)
	
	if (drop)
		return(drop(sampleRbmCpp(object, newdata)))
	else
		return(sampleRbmCpp(object, newdata))
}