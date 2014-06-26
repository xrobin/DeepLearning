#  Fits an exponential (error ~ d * exp(-a*t) + b) with reasonable start values
#  @param x a \code{\link{data.frame}} with columns \code{error} and \code{t} to be fitted.
#  @return the model fitted by \code{\link{nls}}
fit.1exp <- function(x) {
	#print(x)
	#write.csv(x, "data_to_fit.csv")
	sdx <- sd(x$error)
	start <- list(
		a = 0.01, # Just make up the initial alpha - we expect this kind of value, a convergence over about a few 100 steps
		#a = -5, # For a double expoential, exp(-5) =~ 0.01
		b = mean(x$error), # b is the plateau, the mean should be a decent starting point, a bit high because we started higher
		d = 2 * sdx)
		#v = 1)
	#return(nls(error ~ d * t^(-v) * exp(-a*t) + b, # v = 0
	#return(nls(error ~ d * exp(-a*t^v) + b, # v = 1
	#return(nls(error ~ d * t^(-v) * exp(-exp(a) * t) + b, # lower[a] = -Inf, v = 0
	return(nls(error ~ d * exp(-a * t) + b, # lower[a] = -Inf, v = 0
	#return(nls(error ~ d * t^(-v) + b, # v = 1
			   data = x, start = start, lower = 0, algorithm = "port",
			   trace = FALSE, control = nls.control(maxiter = 500, warnOnly = TRUE)))
}

# #  Fits 2 exponentials (error ~ d * exp(-a*t) + d2 * exp(-a2*t) + b) with reasonable start values
# #  @param x a \code{\link{data.frame}} with columns \code{error} and \code{t} to be fitted.
# #  @return the model fitted by \code{\link{nls}}
# fit.2exp <- function(x) {
# 	sdx <- sd(x$error)
# 	start <- list(
# 		a = 0.01, # Just make up the initial alpha - we expect this kind of value, a convergence over about a few 100 steps
# 		a2 = 1,
# 		b = mean(x$error), # b is the plateau, the mean should be a decent starting point, a bit high because we started higher
# 		d = 2 * sdx,
# 		d2 = 5000)
# 	return(nls(error ~ d * exp(-a * t) + d2 * exp(-a2 * t) + b,
# 			   data = x, start = start, lower = 0, algorithm = "port",
# 			   trace = FALSE, control = nls.control(maxiter = 500, warnOnly = TRUE)))
# }
# 
# #  Fits a power law (error ~ d * t ^ (-v) + b) with reasonable start values
# #  @param x a \code{\link{data.frame}} with columns \code{error} and \code{t} to be fitted.
# #  @return the model fitted by \code{\link{nls}}
# fit.power <- function(x) {
# 	sdx <- sd(x$error)
# 	start <- list(
# 		b = mean(x$error), # b is the plateau, the mean should be a decent starting point, a bit high because we started higher
# 		d = 2 * sdx,
# 		v = 1)
# 	return(nls(error ~ d * t ^ (-v) + b,
# 			   data = x, start = start, lower = 0, algorithm = "port",
# 			   trace = FALSE, control = nls.control(maxiter = 500, warnOnly = TRUE)))
# }
# 
# #  Fits a power law and an exponential (error ~ d * t ^ (-v) + b) with reasonable start values
# #  @param x a \code{\link{data.frame}} with columns \code{error} and \code{t} to be fitted.
# #  @return the model fitted by \code{\link{nls}}
# fit.powerexp <- function(x) {
# 	sdx <- sd(x$error)
# 	start <- list(
# 		a = 0.01,
# 		b = mean(x$error), # b is the plateau, the mean should be a decent starting point, a bit high because we started higher
# 		d = 2 * sdx,
# 		v = 0)
# 	return(nls(error ~ d * t ^ (-v) * exp(-a * t) + b,
# 			   data = x, start = start, lower = 0, algorithm = "port",
# 			   trace = FALSE, control = nls.control(maxiter = 500, warnOnly = TRUE)))
# }

#  Fits a linear regression (error ~ a * t + b)
#  @param x a \code{\link{data.frame}} with columns \code{error} and \code{t} to be fitted.
#  @return the model fitted by \code{\link{nls}}
fit.linear <- function(x) {
	sdx <- sd(x$error)
	start <- list(
		a = 0, # Assume it's flat
		b = x$error[1] # Intercept term, close to one of the first points of error
	)
	return(nls(error ~ a * t + b,
			   data = x, start = start, lower = c(-Inf, 0), upper=c(0, Inf), algorithm = "port",
			   trace = FALSE, control = nls.control(maxiter = 500, warnOnly = TRUE)))
}

#  Fits a flat line (error ~ b)
#  @param x a \code{\link{data.frame}} with columns \code{error} and \code{t} to be fitted.
#  @return the model fitted by \code{\link{nls}}
fit.flat <- function(x) {
	sdx <- sd(x$error)
	start <- list(
		b = mean(x$error) # isn't this actually the best fit at all?
	)
	return(nls(error ~ 0 * t + b,
			   data = x, start = start, lower = 0, algorithm = "port",
			   trace = FALSE, control = nls.control(maxiter = 500, warnOnly = TRUE)))
}

# nice formatting with at least 'digits' significant digits, and all significant digits for large numbers
prettyFormat <- function(x, digits) {
	UseMethod("prettyFormat", x)
}
prettyFormat.numeric <- function(x, digits=3) {
	return(ifelse(x > 10 ^ (digits - 1), sprintf("%.0f", x), signif(x, digits)))
}
prettyFormat.matrix <- function(x, digits=3) {
	x[] <- prettyFormat(as.vector(x))
	return(x)
}

#' @title Continue functions
#' @rdname continue.functions
#' @name continue.functions
#' @description Functions to stop (by returning \code{FALSE}) the training if it has converged.
#' 
#' \code{continue.function.exponential} fits an exponential to the error and return \code{TRUE} if the function hasn't converged (or in case of doubt), \code{FALSE} if the function has converged or if the data couldn't be fitted (plateau reached or no exponential fit).
#' As a side effect, this function plots the error and the fit, and prints a summary of the fit on the console
#' 
#' \code{continue.function.always} always returns true so that the training carries on until \code{maxiters} is reached
#' 
#' @param error a vector of the errors along the training
#' @param iter,batchsize current iteration number and batchsize.
#' @param maxiters maximum number of iterations
#' @param layer during RBM pre-training, which layer is being pre-trained. Otherwise, 0.
#' @return boolean (see description)
#' @importFrom plotrix addtable2plot
#' @export
continue.function.exponential <- function(error, iter, batchsize, maxiters, layer = 0, ic = AIC) {

	x <- data.frame(error = error, t = seq_along(error))
	
	# Plot the error
	plot(error~t, x, type="l", main = sprintf("n = %d, i = %d", batchsize, iter))
	
	# Not enough data points... continue
	if (length(error) < 20) {
		print("Not enough data points to fit...")
		legend("bottomleft", legend="Not enough data points to fit", col="black")	
		return(TRUE)
	}
	
	# Get the 3 fits
	exp.fit <- try(fit.1exp(x[-c(1:10),]))
	linear.fit <- try(fit.linear(x[(length(error)/2+1):(length(error)),]))
	flat.fit <- try(fit.flat(x[(length(error)/2+1):(length(error)),]))
	
	fit.error <- FALSE
	if (inherits(exp.fit, "try-error")) {
		print("Could not fit exponential function...")
		legend("bottomleft", legend="Could not fit exponential function", col="black")	
		print(exp.fit)
		fit.error <- TRUE
	}
	if (inherits(linear.fit, "try-error")) {
		print("Could not fit linear function...")
		legend("bottomleft", legend="Could not fit linear function", col="black")	
		print(linear.fit)
		fit.error <- TRUE
	}
	if (inherits(flat.fit, "try-error")) {
		print("Could not fit flat function...")
		legend("bottomleft", legend="Could not fit flat function", col="black")	
		print(flat.fit)
		fit.error <- TRUE
	}
	if (fit.error) {return(TRUE)}
	
	# Calculate the BICs and normalize with degrees of freedom
	ic.exp <- ic(exp.fit)  / df.residual(exp.fit)
	ic.lin <- ic(linear.fit)  /df.residual(linear.fit)
	ic.flat <- ic(flat.fit) / df.residual(flat.fit)
	
	# Find the best fit with BIC
	best.fit <- which.min(c(ic.exp, ic.lin, ic.flat))
	
	# Add the fit to the plot
	lines(predict(exp.fit, x) ~ x$t, type="l", col="red", lwd=1 + 3 * (best.fit == 1), lty=2) # Extrapolate the exponential in dashed line
	lines(predict(exp.fit) ~ x$t[-c(1:10)], type="l", col="red", lwd=1 + 3 * (best.fit == 1), lty=1) # exponential fit only
	lines(predict(linear.fit) ~ x$t[(length(error)/2+1):(length(error))], type="l", col="blue", lwd=1 + 3 * (best.fit == 2), lty=1)
	lines(predict(flat.fit) ~ x$t[(length(error)/2+1):(length(error))], type="l", col="green", lwd=1 + 3 * (best.fit == 3), lty=1)
	
	# Plot the coefficients
	legend("bottomright", legend=do.call(expression, list(
		substitute(error == d %*% e^{- a * t} + b ~~ (ic == bic), c(as.list(prettyFormat(coef(exp.fit))), bic = prettyFormat(ic.exp))),
		substitute(error == a * t + b ~~ (ic == bic), c(as.list(prettyFormat(coef(linear.fit))), bic = prettyFormat(ic.lin))),
		substitute(error == b ~~ (ic == bic), c(as.list(prettyFormat(coef(flat.fit))), bic = prettyFormat(ic.flat))))
	))
	
	# Plot some diagnostics about the fit
	title(sub=sprintf("exp: %s in %d iterations\nlinear: %s in %d iterations\nflat: %s in %d iterations", 
					  exp.fit$convInfo$stopMessage, exp.fit$convInfo$finIter,
					  linear.fit$convInfo$stopMessage, linear.fit$convInfo$finIter,
					  flat.fit$convInfo$stopMessage, flat.fit$convInfo$finIter
					  ))

	# Compute the model summary...
	summary.exp <- try(summary(exp.fit))
	summary.linear <- try(summary(linear.fit))
	summary.flat <- try(summary(flat.fit))

	summary.error <- FALSE
	if (inherits(summary.exp, "try-error")) {
		print("Could not compute exponential fit summary...")
		legend("bottomleft", legend="Could not compute exponential fit summary", col="black")	
		print(exp.fit)
		summary.error <- TRUE
	}
	if (inherits(summary.linear, "try-error")) {
		print("Could not compute linear fit summary...")
		legend("bottomleft", legend="Could not compute linear fit summary", col="black")	
		print(linear.fit)
		summary.error <- TRUE
	}
	if (inherits(summary.flat, "try-error")) {
		print("Could not compute flat fit summary...")
		legend("bottomleft", legend="Could not compute flat fit summary", col="black")	
		print(flat.fit)
		summary.error <- TRUE
	}
	if (summary.error) {return(TRUE)}
	
	# Merge the summaries
	summary.full <- prettyFormat(rbind(summary.exp$param, summary.linear$param, summary.flat$param))
	# Add names
	summary.full <- cbind(c("exp a", "exp b", "exp d", "linear a", "linear b", "flat b"), summary.full)
	
	# Add it to the plot
	addtable2plot("topright", table = summary.full)
	
	if (best.fit == 3) { # Flat - stop
		print("STOP (flat)")
		legend("bottomleft", legend="STOP (flat)", col="darkgreen")	
		return(FALSE)
	}
	else if (best.fit == 2) { # Linear - continue
		print("Continue (linear)")
		legend("bottomleft", legend="Continue (linear)", col="maroon")	
		return(TRUE)
	}
	else { # Exponential - do as before, check significance and exit if a is exceeded
		# Is it significant?]
		signif <- summary.exp$param[,"Pr(>|t|)"]
		# If a or d are non significant, we have a flat plateau and should stop
		# In doubt continue to train, so use very high p-values
		if (signif["a"] > 0.1 && signif["d"] > 0.1) {
			print("Fit not significant...")
			legend("bottomleft", legend="Continue (not significant)", col="blue")
			return(TRUE)
		}
		
		# If the fit is good (both a and d < 0.05), we have a good exponential fit
		# And should exit if we are in the plateau (t >> 1 / alpha (plateau.factor times)),
		# or continue otherwise
		if (signif["a"] < 0.05 && signif["d"] < 0.05) {
			plateau.factor <- 5
			print("Fit significant...")
			a <- coef(exp.fit)["a"]
			must.continue <- length(error) < plateau.factor * (1 / a)
			print(sprintf("%d < %d * (1 / %.3f) = %s", length(error), plateau.factor, a, ifelse(must.continue, "TRUE (continue)", "FALSE (finished)")))
			legend("bottomleft", legend=ifelse(must.continue, "Continue", "STOP"), col=ifelse(must.continue, "red", "green"))
			return(must.continue)
		}
		
		# Here we have an uncertain fit
		# In doubt, continue until we can fit
		legend("bottomleft", legend="Continue (fit uncertain)", col="purple")
		print("Continue (fit uncertain)")
		return(TRUE)
	}
	legend("bottomleft", legend="Continue (reached normally unreachable code, there is a bug there!)", col="orange")
	print("Continue (reached normally unreachable code, there is a bug there!)")
	return(TRUE)
}

#' @rdname continue.functions
#' @export
continue.function.always <- function(error, iter, batchsize, maxiters, layer = 0) {
	return(TRUE)
}

#' @rdname continue.functions
#' @export
continue.function.exponential.aic <- function(error, iter, batchsize, maxiters, layer = 0) {
	continue.function.exponential(error, iter, batchsize, AIC)
}

#' @rdname continue.functions
#' @export
continue.function.exponential.bic <- function(error, iter, batchsize, maxiters, layer = 0) {
	continue.function.exponential(error, iter, batchsize, BIC)
}

# @rdname continue.function
# @export
# continue.function.1exp <- function(error, iter, batchsize) {
# 	continue.function.exponential(error, iter, batchsize)
# }
# legend.1exp <- function(coefs) {
# 	legend("bottomright", legend=substitute(error == d %*% e^{- a * t} + b, as.list(coefs)))
# }
# 
# @rdname continue.function
# @export
# continue.function.2exp <- function(error, iter, batchsize) {
# 	continue.function.exponential(error, iter, batchsize, fit.2exp, legend.2exp)
# }
# legend.2exp <- function(coefs) {
# 	legend("bottomright", legend=substitute(error == d %*% e^{- a * t} + d2 %*% e^{- a2 * t} + b, as.list(coefs)))
# }
# 
# @rdname continue.function
# @export
# continue.function.powerexp <- function(error, iter, batchsize) {
# 	continue.function.exponential(error, iter, batchsize, fit.powerexp)
# }
# legend.powerexp <- function(coefs) {
# 	legend("bottomright", legend=substitute(error == d %*% t^{-v} %*% e^{- a * t} + b, as.list(coefs)))
# }

#' @rdname continue.functions
#' @export
continue.function.random <- function(error, iter, batchsize, maxiters, layer = 0) {
	r <- as.logical(round(runif(1)))
	print(r)
	return(r)
}