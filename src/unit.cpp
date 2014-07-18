/* Exports for unit tests */

#include <Rcpp.h>
using Rcpp::as;
using Rcpp::wrap;
using Rcpp::NumericVector;

#include <RcppEigen.h> 
#include "DeepBeliefNet.h"
#include "RcppConversions.h"

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
SEXP unit_DbnGradient(SEXP& aDBN, SEXP& aDataMatrix) {
	const Eigen::Map<Eigen::MatrixXd> dataAsEigen(as<Eigen::Map<Eigen::MatrixXd>>(aDataMatrix));
	
	DeepBeliefNet myDBN = as<DeepBeliefNet>(aDBN);
	shared_array_ptr<double> df = myDBN.getData().clone();
	myDBN.getGradient(dataAsEigen.transpose(), df); // Set gradient in df

	return wrap(df);
}
