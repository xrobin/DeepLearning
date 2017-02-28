#' @title Predict Methods for Deep Belief Nets and Restricted Bolzman Machines
#' @name predict
#' @aliases predict.DeepBeliefNet
#' @description Obtain predictions from a \code{\link{DeepBeliefNet}} or \code{\link{RestrictedBolzmannMachine}} object
#' @param object the model
#' @param newdata a \code{\link{data.frame}} or \code{\link{matrix}} providing the data. Must have the same columns than the input layer of the model.
#' @param drop do not return additional dimensions
#' @param \dots ignored
#' @examples
#' library(mnist)
#' data(mnist)
#' ## Make predictions on a DBN object
#' data(trained.mnist)
#' predict(trained.mnist, mnist$test$x[1:10,])
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
#' data(pretrained.mnist)
#' rbm <- pretrained.mnist[[1]]
#' predictions <- predict(rbm, mnist$test$x)
#' dim(predictions) # 1000 columns, output size of the rbm
#' ncol(predictions) == rbm$output$size
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