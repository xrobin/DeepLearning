#include <Rcpp.h>
using Rcpp::NumericMatrix;
using Rcpp::as;
using Rcpp::wrap;

#include <RcppEigen.h> 

#include <vector>
using std::vector;
#include <memory> // std::unique_ptr
using std::unique_ptr;

#include "Layer.h"
#include "RBM.h"
#include "DeepBeliefNet.h"
#include "RcppConversions.h"

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
DeepBeliefNet unrollDbnCpp(DeepBeliefNet& aDBN) {
	return aDBN.unroll();
}

/* PREDICT */

// [[Rcpp::export]]
Eigen::MatrixXd predictRbmCpp(const RBM& anRBM, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix) {
	return anRBM.predict(aDataMatrix.transpose()).transpose();
}

// [[Rcpp::export]]
Eigen::MatrixXd predictDbnCpp(const DeepBeliefNet& aDBN, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix) {
	return aDBN.predict(aDataMatrix.transpose()).transpose();
}

/* RECONSTRUCT */

// [[Rcpp::export]]
Eigen::MatrixXd reconstructRbmCpp(const RBM& anRBM, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix) {
	return anRBM.reconstruct(aDataMatrix.transpose()).transpose();
}

// [[Rcpp::export]]
Eigen::MatrixXd reconstructDbnCpp(const DeepBeliefNet& aDBN, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix) {
	return aDBN.reconstruct(aDataMatrix.transpose()).transpose();
}

/* PRETRAIN */

// [[Rcpp::export]]
RBM pretrainRbmCpp(RBM& anRBM, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix, const PretrainParameters& params, const std::unique_ptr<PretrainProgress>& diag, const ContinueFunction& cont) {
	anRBM.pretrain(aDataMatrix.transpose(), params, *diag, cont);
	return anRBM;
}

// [[Rcpp::export]]
DeepBeliefNet pretrainDbnCpp(DeepBeliefNet& aDBN, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix, const std::vector<PretrainParameters>& params, const std::unique_ptr<PretrainProgress>& diag, ContinueFunction& cont, const Rcpp::IntegerVector& aSkip) {
	const std::vector<size_t> skip(as<std::vector<size_t>>(aSkip));
	aDBN.pretrain(aDataMatrix.transpose(), params, *diag, cont, skip);
	return aDBN;
}


/* TRAIN */

// [[Rcpp::export]]
DeepBeliefNet trainDbnCpp(DeepBeliefNet& aDBN, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix, const TrainParameters& trainParams, const std::unique_ptr<TrainProgress>& diag, const ContinueFunction& cont) {
	aDBN.train(aDataMatrix.transpose(), trainParams, *diag, cont);
	return aDBN;
}

/* REVERSE */

// [[Rcpp::export]]
RBM reverseRbmCpp(RBM& anRBM) {
	return anRBM.reverse();
}

// [[Rcpp::export]]
DeepBeliefNet reverseDbnCpp(DeepBeliefNet& aDBN) {
	return aDBN.reverse();
}

/* Energy */

// [[Rcpp::export]]
ArrayX1d energyRbmCpp(const RBM& anRBM, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix) {
	return anRBM.energy(aDataMatrix.transpose());
}

// [[Rcpp::export]]
ArrayX1d energyDbnCpp(const DeepBeliefNet& aDBN, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix) {
	return aDBN.energy(aDataMatrix.transpose());
}

/* Error */

// [[Rcpp::export]]
ArrayX1d errorRbmCpp(const RBM& anRBM, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix) {
	return anRBM.error(aDataMatrix.transpose());
}

// [[Rcpp::export]]
ArrayX1d errorDbnCpp(const DeepBeliefNet& aDBN, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix) {
	return aDBN.error(aDataMatrix.transpose());
}
