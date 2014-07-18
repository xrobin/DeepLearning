/* Exports for unit tests */

#include <Rcpp.h>
#include <RcppEigen.h> 

#include <DeepLearning/DeepBeliefNet.h>
#include <RcppConversions.h>
using namespace DeepLearning;


SEXP unit_DbnGradient(SEXP& aDBN, SEXP& aDataMatrix);

// [[Rcpp::export]]
SEXP unit_DbnGradient(SEXP& aDBN, SEXP& aDataMatrix) {
	const Eigen::Map<Eigen::MatrixXd> dataAsEigen(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(aDataMatrix));
	
	DeepBeliefNet myDBN = Rcpp::as<DeepBeliefNet>(aDBN);
	shared_array_ptr<double> df = myDBN.getData().clone();
	myDBN.getGradient(dataAsEigen.transpose(), df); // Set gradient in df

	return Rcpp::wrap(df);
}
