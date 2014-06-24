rbm.update.batch.slow <- function(rbm, data, momentum = 0, penalization = c("l1", "l2", "none"),
							 lambda = 0, lambda.b = lambda, lambda.c = lambda, lambda.W = lambda,
							 epsilon = 0.1, epsilon.b = epsilon, epsilon.c = epsilon, epsilon.W = epsilon,
							 browseOnInfiniteWeights = FALSE) {
	
	penalization = match.arg(penalization)
	
	# Copy the weights locally
	# TODO: check if this is really faster
	b <- rbm$b
	c <- rbm$c
	c.mat = matrix(c, nrow=dim(data)[1], ncol=length(c), byrow=TRUE)
	W <- rbm$W
	
	# Initialize the deltas with 0
	delta.b = b - b
	delta.c = c - c
	delta.W = W - W
	
	# Sample from P(h|f)
	Alpha <- t(c + t(W) %*% t(data))
	#Alpha <- matrix(c, nrow=dim(data)[1], ncol=length(c), byrow=TRUE) + data %*% W
	#dim(matrix(c, nrow=dim(data)[1], ncol=length(c), byrow=TRUE))
	if (rbm$output == "continuous") {
		yh <- matrix(runif(prod(dim(Alpha))), nrow=dim(data)[1])
		h.sampled <- 1 / Alpha * log(yh * (exp(Alpha) - 1) + 1)
		# Replace NaN values with 0.5
		h.sampled[abs(Alpha) < sqrt(.Machine$double.eps) * 2 ] <- yh[abs(Alpha) < sqrt(.Machine$double.eps) * 2 ]
	}
	else if (rbm$output == "gaussian") {
		yh <- matrix(rnorm(prod(dim(Alpha))), nrow=dim(data)[1])
		h.sampled <- Alpha + yh
	}
	else { # Treat gaussian as binary
		yh <- matrix(runif(prod(dim(Alpha))), nrow=dim(data)[1])
		P.h.given.f <- 1 / (1 + exp(-Alpha))
		h.sampled <- yh < P.h.given.f
	}
	
	# P(f|h)
	Beta <- t(b + W %*% t(h.sampled))
	#Beta2 <- b + tcrossprod(h.sampled, W)
	if (rbm$input == "continuous") {
		P.f.given.h <- (exp(Beta) * (1 - 1 / Beta) + 1 / Beta) / (exp(Beta) - 1)
		# Replace NaN values with 0.5
		P.f.given.h[abs(Beta) < sqrt(.Machine$double.eps) * 2 ] <- 0.5
	}
	else if (rbm$input == "gaussian") {
		P.f.given.h <- Beta
	}
	else {
		P.f.given.h <- 1 / (1 + exp(-Beta))
	}
	
	# P(f|g)
	Alpha <- try(t(c + t(W) %*% t(P.f.given.h)))
	if (rbm$output == "continuous") {
		P.h.given.f <- (exp(Alpha) * (1 - 1 / Alpha) + 1 / Alpha) / (exp(Alpha) - 1)
		# Replace NaN values with 0.5
		P.h.given.f[abs(Alpha) < sqrt(.Machine$double.eps) * 2 ] <- 0.5
	}
	else if (rbm$output == "gaussian") {
		P.h.given.f <- Alpha # Actually the expectation
	}
	else {
		P.h.given.f <- 1 / (1 + exp(-Alpha))
	}
	
	# Compute deltas
	delta.c <- colSums(h.sampled - P.h.given.f) 
	delta.b <- colSums(data - P.f.given.h) 
	delta.W <- (t(data) %*% h.sampled - t(P.f.given.h) %*% P.h.given.f)
	
	if (any(!is.finite(delta.W))) {
		if (browseOnInfiniteWeights) {
			cat("Infinite weights computed. Please inspect why.")
			browser()
		}
		stop("Infinite weights computed. Aborting.")
	}
	
	# Penalize large W
	if (penalization == "l1") {
		delta.W = (delta.W / dim(data)[1]) - lambda.W * sign(W)
		delta.b = (delta.b / dim(data)[1]) - lambda.b * sign(b)
		delta.c = (delta.c / dim(data)[1]) - lambda.c * sign(c)
	}
	else if (penalization == "l2") {
		delta.W = (delta.W / dim(data)[1]) - lambda.W * W
		delta.b = (delta.b / dim(data)[1]) - lambda.b * b
		delta.c = (delta.c / dim(data)[1]) - lambda.c * c
	}
	
	# Compute weights increments
	rbm$Winc <- momentum * rbm$Winc + epsilon.W * delta.W
	rbm$bInc <- momentum * rbm$bInc + epsilon.b * delta.b
	rbm$cInc <- momentum * rbm$cInc + epsilon.c * delta.c
	
	# Update and return the RBM object
	rbm$b <- b + rbm$bInc
	rbm$c <- c + rbm$cInc
	rbm$W <- W + rbm$Winc
	return(rbm)
}