#' @title Drop a DeepBeliefNet
#' @description Drops the DeepBeliefNet information if it contains only a single RestrictedBolzmannMachine.
#' Provides an alias to \code{\link[base]{drop}} as \code{drop.default}.
#' @param x the DeepBeliefNet
#' @seealso \code{\link{DeepBeliefNet}}, \code{\link[base]{drop}}
#' @examples
#' dbn <- dbn(layers=c(784, 1000), input="continuous", output="binary")
#' print(dbn)
#' rbm <- drop(dbn)
#' print(rbm)
#' @export
drop <- function(x) 
	UseMethod("drop")

#' @export
# Cannot write:
# drop.default <- base::drop
# because it generates a warning upon R CMD check
drop.default <- function(x)
	base::drop(x)