#' @rdname print
#' @examples 
#' rbm <- rbm(784, 1000, input="continuous", output="gaussian")
#' print(rbm)
#' @seealso \code{\link{RestrictedBolzmannMachine}}, \code{\link{print}}
#' @export
print.RestrictedBolzmannMachine <- function(x, ...) {
	# Assess status
	training.state <- "Initialized"
	if (x$pretrained)
		training.state <- "Pre-trained"
	
	cat("Restricted Bolzman Machine\n")
	classes <- c(x$input$type, x$output$type)
	# Pad the layer sizes to match the classes and align properly
	layers <- sprintf(sprintf("%% %ii", nchar(classes)), c(x$input$size, x$output$size))
	# Now cat all this
	cat(paste(layers, collapse = " -> "), "\n", sep="")
	cat(paste(classes, collapse = " -> "), "\n", sep="")
	cat("Status:", training.state)
}

#' @rdname rev
#' @examples 
#' rbm <- RestrictedBolzmannMachine(Layer(784, "continuous"), Layer(1000, "gaussian"))
#' rev(rbm)
#' @export
rev.RestrictedBolzmannMachine <- function(x) {
	return(reverseRbmCpp(x))
}


#' @rdname Extract
#' @aliases $
#' @examples
#' # Get the first layer as RestrictedBolzmannMachine
#' rbm$W
#' rbm$b
#' rbm$c
#' @export
`$.RestrictedBolzmannMachine` <- function (x, name) {
	name <- match.arg(tolower(name), c("w", "b", "c", names(x)))
	if (name == "w") {
		return(extractRbmWCpp(x))
	}
	if (name == "c") {
		return(extractRbmCCpp(x))
	}
	if (name == "b") {
		return(extractRbmBCpp(x))
	}
	return(x[[name]])
}

#' @rdname Extract
#' @aliases $<-
#' @examples
#' rbm$b <- 1:10
#' @export
`$<-.RestrictedBolzmannMachine` <- function(x, name, value) {
	name <- match.arg(tolower(name), c("w", "b", "c", names(x)))
	if (name == "w") {
		if (!is.matrix(value) || storage.mode(value) != "double") {
			stop("'value' must be a numeric matrix")
		}
		if (!identical(dim(value), dim(extractRbmWCpp(x)))) {
			stop("'value' dimensions incompatible with RBM")
		}
		x <- setRbmWCpp(x, value)
	}
	else if (name == "c") {
		if (!is.numeric(value)) {
			stop("'value' must be a numeric vector")
		}
		if (length(value) != x$output$size) {
			stop("'value' length incompatible with RBM")
		}
		x <- setRbmCCpp(x, value)
	}
	else if (name == "b") {
		if (!is.numeric(value)) {
			stop("'value' must be a numeric vector")
		}
		if (length(value) != x$input$size) {
			stop("'value' length incompatible with RBM")
		}
		x <- setRbmBCpp(x, value)
	}
	else {
		x[[name]] <- value
	}
	return(x)
}

