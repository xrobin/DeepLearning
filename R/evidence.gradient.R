# #' @title Computes and returns the gradient of the evidence function
# #' @description This is specifically the contrastive divergence evidence.
# #' @param x the \code{\link{RestrictedBolzmannMachine}} object
# #' @param data feature vector
# #' @param ... ignored
# #' @return the reconstruction error of the data
# #' @export
# evidence.gradient <- function(...)
# 	UseMethod("evidence.gradient")
# 
# 
# #' @rdname evidence.gradient
# #' @export
# evidence.gradient.RestrictedBolzmannMachine <- function(x, data, ...) {
# 	stop("The evidence gradient is available only during pre-training for now.")
# 	ensure.data.validity(data, x$input)
# 	return(evidenceGradientRbmCpp(x, data))
# }