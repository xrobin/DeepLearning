#' @title Predict Methods for Deep Belief Nets and Restricted Bolzman Machines
#' @name predict
#' @aliases predict.DeepBeliefNet
#' @description Obtain predictions from a \code{\link{DeepBeliefNet}} or \code{\link{RestrictedBolzmannMachine}} object
#' @param object the model
#' @param newdata a \code{\link{data.frame}} or \code{\link{matrix}} providing the data. Must have the same columns than the input layer of the model.
#' @param drop do not return additional dimensions
#' @param \dots ignored
#' @examples
#' data(mnist)
#' ## Make predictions on a DBN object
#' dbn <- DeepBeliefNet(Layers(c(784, 1000, 500, 250, 30), input="continuous", output="gaussian"))
#' pretrained <- pretrain(dbn, mnist$train$x, 
#'                        penalization = "l2", lambda=0.0002, epsilon=c(.1, .1, .1, .001), 
#'                        batchsize = 100, maxiters=100000)
#' predict(pretrained, mnist$test$x)
#' @export
predict.DeepBeliefNet <- function(object, newdata, drop=TRUE, ...) {
	
	# If user passed a vector f or a data.frame, we convert it to a compatible matrix
	if (!(is.matrix(newdata) || is.data.frame(newdata))) {
		newdata <- t(newdata)
	}

	# Make sure C++/RcppEigen can deal with the data
	ensure.data.validity(newdata, object[[1]]$input)
	
	if (drop)
		return(drop(predictDbnCpp(object, newdata)))
	else
		return(predictDbnCpp(object, newdata))
}


#' @rdname predict
#' @examples
#' ## Make predictions on a RBM object
#' rbm <- RestrictedBolzmannMachine(Nv = 784, Nh = 1000, input="continuous", output="binary")
#' pretrained <- pretrain(rbm, mnist$train$x)
#' predict(pretrained, mnist$test$x)
#' @export
predict.RestrictedBolzmannMachine <- function(object, newdata, drop=TRUE, ...) {

	# If user passed a vector f, we convert it to a compatible matrix
	if (!(is.matrix(newdata) || is.data.frame(newdata))) {
		newdata <- t(newdata)
	}

	# Make sure C++/RcppEigen can deal with the data
	ensure.data.validity(newdata, object$input)
	
	if (drop)
		return(drop(predictRbmCpp(object, newdata)))
	else
		return(predictRbmCpp(object, newdata))
}