#' @title Fine-tunes the DeepBeliefNet
#' @description Performs fine-tuning on the DBN network with backpropagation.
#' @param x the DBN
#' @param data the training data
#' @param miniters,maxiters minimum and maximum number of iterations to perform
#' @param batchsize the size of the batches on which error & gradients are averaged
#' @param continue.function that can stop the training between miniters and maxiters if it returns \code{FALSE}. 
#' By default, \code{\link{continue.function.exponential}} will be used. An alternative is to use \code{\link{continue.function.always}} that will always return true and thus carry on with the training until maxiters is reached.
#' A user-supplied function must accept \code{(error, iter, batchsize)} as input and return a \code{\link{logical}} of length 1. The training is stopped when it returns \code{FALSE}.
#' @param continue.function.frequency the frequency at which continue.function will be assessed.
#' @param continue.stop.limit the number of consecutive times \code{continue.function} must return \code{FALSE} before the training is stopped. For example, \code{1} will stop as soon as \code{continue.function} returns \code{FALSE}, whereas \code{Inf} will ensure the result of \code{continue.function} is never enforced (but the function is still executed). The default is \code{3} so the training will continue until 3 consecutive calls of \code{continue.function} returned \code{FALSE}, giving more robustness to the decision.
#' @param optim.control control arguments for the optim function that are not typically changed for normal operation. The parameters are:
#' maxit, type, trace, steplength, stepredn, acctol, reltest, abstol, intol, setstep. Their default values are defined in TrainParameters.h.
#' @param diag,diag.rate,diag.data,diag.function diagnmostic specifications. See details.
#' @param n.proc number of cores to be used for Eigen computations
#' @param ... ignored
#' 
#' @section Diagnostic specifications:
#' The specifications can be passed directly in a list with elements \code{rate}, \code{data} and \code{f}, or separately with parameters \code{diag.rate}, \code{diag.data} and \code{diag.function}. The function must be of the following form:
#' \code{function(rbm, batch, data, iter, batchsize, maxiters)}
#' \itemize{
#' \item \code{rbm}: the RBM object after the training iteration.
#' \item \code{batch}: the batch that was used at that iteration.
#' \item \code{data}: the data provided in \code{diag.data} or \code{diag$data}.
#' \item \code{iter}: the training iteration number, starting from 0 (before the first iteration).
#' \item \code{batchsize}: the size of the batch.
#' \item \code{maxiters}: the target number of iterations.
#' }
#' Note the absence of the \code{layer} argument that is available only in \code{\link{pretrain}}.
#' 
#' The following \code{diag.rate} or \code{diag$rate} are supported:
#' \itemize{
#' \item \dQuote{none}: the diag function will never be called.
#' \item \dQuote{each}: the diag function will be called before the first iteration, and at the end of each iteration.
#' \item \dQuote{accelerate}: the diag function will called before the first iteration, at the first 200 iterations, and then with a rate slowing down proportionally with the iteration number.
#' }
#' 
#' Note that diag functions incur a slight overhead as they involve a callback to R and multiple object conversions. Setting \code{diag.rate = "none"} removes any overhead.
#' 
#' @section Progress:
#' \code{train.progress} is a convenient pre-built diagnostic specification that displays a progress bar.
#' 
#' @return the fine-tuned DBN
#' @examples 
#' data(pretrained.mnist)
#' 
#' \dontrun{
#' # Fine-tune the DBN with backpropagation
#' trained.mnist <- train(unroll(pretrained.mnist), mnist$train$x, maxiters = 2000, batchsize = 1000,
#'                        optim.control = list(maxit = 10))
#' }
#' \dontrun{
#' # Train with a progress bar
#' # In this case the overhead is nearly 0
#' diag <- list(rate = "each", data = NULL, f = function(rbm, batch, data, iter, batchsize, maxiters) {
#' 	if (iter == 0) {
#' 		DBNprogressBar <<- txtProgressBar(min = 0, max = maxiters, initial = 0, width = NA, style = 3)
#' 	}
#' 	else if (iter == maxiters) {
#' 		setTxtProgressBar(DBNprogressBar, iter)
#' 		close(DBNprogressBar)
#' 	}
#' 	else {
#' 		setTxtProgressBar(DBNprogressBar, iter)
#' 	}
#' })
#' trained.mnist <- train(unroll(pretrained.mnist), mnist$train$x, maxiters = 1000, batchsize = 100,
#'                        continue.function = continue.function.always, diag = diag)
#' # Equivalent to using train.progress
#' trained.mnist <- train(unroll(pretrained.mnist), mnist$train$x, maxiters = 1000, batchsize = 100,
#'                        continue.function = continue.function.always, diag = train.progress)
#' }
#' @export
train <- function(x, data, 
                  miniters = 100, maxiters = 1000, batchsize = 100,
				  optim.control = list(),
				  continue.function = continue.function.exponential, continue.function.frequency = 100, continue.stop.limit = 3,
				  diag = list(rate = diag.rate, data = diag.data, f = diag.function), diag.rate = c("none", "each", "accelerate"), diag.data = NULL, diag.function = NULL,
				  n.proc = detectCores() - 1, ...) {
	if (!x$unrolled)
		stop("DBN must be unrolled before it can be trained")
	
	# Check for ignored arguments
	ignored.args <- names(list(...))
	if (length(ignored.args) > 0) {
		warning(paste("The following arguments were ignored in train:", paste(ignored.args, collapse=", ")))
	}
	
	ensure.data.validity(data, x[[1]]$input)
	
	# Build diagnostic function
	if (missing(diag) && is.null(diag.data) && is.null(diag.function)) {
		diag$rate <- "none"
	}
	else {
		diag$rate <- match.arg(diag$rate, c("none", "each", "accelerate"))
	}
	
	# Build continue function
	continue.function <- list(
		continue.function = continue.function,
		continue.function.frequency = continue.function.frequency,
		continue.stop.limit = continue.stop.limit
	)
	
	# Check content of optim.control
	optim.names <- c("maxit", "type", "trace", "steplength", "stepredn", "acctol", "reltest", "abstol", "intol", "setstep")
	allowed.names <- names(optim.control) %in% optim.names
	if (any(!allowed.names)) {
		warning(paste("Elements were ignored in optim.control: ", paste(names(optim.control)[!allowed.names], collapse=", ")))
	}
	
	# Training parameters 
	train.control <- list(
		miniters = miniters,
		maxiters = maxiters,
		batchsize = batchsize,
		n.proc = n.proc,
		optim.control = optim.control
	)

	x <- trainDbnCpp(x, data, train.control, diag, continue.function)
	
	x$finetuned <- TRUE
	return(x)
}

#' @rdname train
#' @export
train.progress <- list(rate = "each", data = NULL, f = function(rbm, batch, data, iter, batchsize, maxiters) {
	if (iter == 0) {
		DBNprogressBar <<- txtProgressBar(min = 0, max = maxiters, initial = 0, width = NA, style = 3)
	}
	else if (iter == maxiters) {
		setTxtProgressBar(DBNprogressBar, iter)
		close(DBNprogressBar)
	}
	else {
		setTxtProgressBar(DBNprogressBar, iter)
	}
})