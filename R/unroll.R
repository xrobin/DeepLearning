#' @title Unroll the Deep Belief Net
#' @description Unrolling the DBN means stacking a reversed copy of itself. This create a (pre-trained) auto-encoder that can be 
#' @param x the dbn object
#' @return the unrolled dbn with the \code{unrolled} switch is set to \code{TRUE}.
#' @examples
#' data(pretrained.mnist)
#' unrolled <- unroll(pretrained.mnist)
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