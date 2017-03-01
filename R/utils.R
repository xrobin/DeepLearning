ensure.data.validity <- function(data, input) {
	if (! methods::is(data, "matrix") || ! storage.mode(data) == "double") {
		stop("'data' must be a matrix with storage.mode(data) == 'double'.")
	}
	
	# Make sure the data is adapted to the first layer
	if ((datacols <- ncol(data)) != input$size) {
		stop(sprintf("Invalid number of data column (%d) for the input layer.", datacols))
	}
}