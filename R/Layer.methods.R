
#' @rdname print
#' @examples 
#' layer <- Layer(10L, "gaussian")
#' print(layer)
#' @seealso \code{\link{print}} \code{\link{Layer}}
#' @export
print.Layer <- function(x, ...) {
	cat(sprintf("%d-nodes %s layer\n", x$size, x$type))
}

#' @title Layers equality testing
#' @description Binary operator to test for equality or inequality of two \code{Layer}s. This is equal to a call to \code{\link{identical}}.
#' @name Comparison
#' @rdname Comparison
#' @param e1,e2 Layers to compare
#' @aliases ==.Layer Comparison ==
#' @examples 
#' Layer(10L, "gaussian") == Layer(10L, "gaussian") # TRUE
#' Layer(10L, "gaussian") == Layer(10L, "continuous") # FALSE
#' Layer(10L, "gaussian") == Layer(20L, "gaussian") # FALSE
#' @seealso \code{\link{Comparison}} \code{\link{Layer}}
#' @importFrom methods is
#' @export
`==.Layer` <- function(e1, e2) {
	if (is(e1, "Layer") && is(e2, "Layer"))
		return(identical(e1, e2))
	stop("comparison of these types is not implemented")
}

#' @rdname Comparison
#' @aliases !=.Layer !=
#' @examples 
#' Layer(10L, "gaussian") != Layer(10L, "gaussian") # FALSE
#' Layer(10L, "gaussian") != Layer(10L, "continuous") # TRUE
#' Layer(10L, "gaussian") != Layer(20L, "gaussian") # TRUE
#' @export
`!=.Layer` <- function(e1, e2) {
	if (is(e1, "Layer") && is(e2, "Layer"))
		return(! identical(e1, e2))
	stop("comparison of these types is not implemented")
}