#include <Rcpp.h>

#include <thread>

// Detect how many cores are available on the machine. C++11.
unsigned int detectCores();

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
unsigned int detectCores() {
	return std::thread::hardware_concurrency();
}
