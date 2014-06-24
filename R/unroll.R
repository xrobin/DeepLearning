#' @title Unroll the Deep Belief Net
#' @description Unrolling the DBN means stacking a reversed copy of itself. This create a (pre-trained) auto-encoder that can be 
#' @param x the dbn object
#' @return the unrolled dbn with the \code{unrolled} switch is set to \code{TRUE}.
#' @examples
#' data(mnist)
#' 
#' # Initialize a 784-1000-500-250-30 layers DBN to process the MNIST data set
#' dbn <- dbn(layers=c(784, 1000, 500, 250, 30), input="continuous", output="gaussian")
#' pretrained <- pretrain(dbn, mnist$train$x, 
#'                        penalization = "l2", lambda=0.0002, epsilon=c(.1, .1, .1, .001), 
#'                        batchsize = 100, maxiters=100000)
#' 
#' unrolled <- unroll(pretrained)
#' @importFrom methods is
#' @export
unroll <- function(x) {
	if (is(x, "DeepBeliefNet")) {
		return(.Call('DeepLearning_unrollDbnCpp', PACKAGE = 'DeepLearning', x))
	}
	else {
		stop("Expected a DeepBeliefNet")
	}
}