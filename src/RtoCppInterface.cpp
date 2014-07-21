#include <Rcpp.h>
#include <RcppEigen.h> 

#include <vector>
using std::vector;
#include <memory> // std::unique_ptr
using std::unique_ptr;

#include <DeepLearning/Layer.h>
#include <DeepLearning/RBM.h>
#include <DeepLearning/DeepBeliefNet.h>
#include <RcppConversions.h>
#include "RtoCppInterface.h"
//using namespace DeepLearning;


// [[Rcpp::export]]
DeepLearning::DeepBeliefNet unrollDbnCpp(DeepLearning::DeepBeliefNet& aDBN) {
	return aDBN.unroll();
}

/* PREDICT */

// [[Rcpp::export]]
Eigen::MatrixXd predictRbmCpp(const DeepLearning::RBM& anRBM, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix) {
	return anRBM.predict(aDataMatrix.transpose()).transpose();
}

// [[Rcpp::export]]
Eigen::MatrixXd predictDbnCpp(const DeepLearning::DeepBeliefNet& aDBN, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix) {
	return aDBN.predict(aDataMatrix.transpose()).transpose();
}

/* RECONSTRUCT */

// [[Rcpp::export]]
Eigen::MatrixXd reconstructRbmCpp(const DeepLearning::RBM& anRBM, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix) {
	return anRBM.reconstruct(aDataMatrix.transpose()).transpose();
}

// [[Rcpp::export]]
Eigen::MatrixXd reconstructDbnCpp(const DeepLearning::DeepBeliefNet& aDBN, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix) {
	return aDBN.reconstruct(aDataMatrix.transpose()).transpose();
}

/* PRETRAIN */

// [[Rcpp::export]]
DeepLearning::RBM pretrainRbmCpp(DeepLearning::RBM& anRBM, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix, const DeepLearning::PretrainParameters& params, const std::unique_ptr<DeepLearning::PretrainProgress>& diag, const DeepLearning::ContinueFunction& cont) {
	anRBM.pretrain(aDataMatrix.transpose(), params, *diag, cont);
	return anRBM;
}

// [[Rcpp::export]]
DeepLearning::DeepBeliefNet pretrainDbnCpp(DeepLearning::DeepBeliefNet& aDBN, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix, const std::vector<DeepLearning::PretrainParameters>& params, const std::unique_ptr<DeepLearning::PretrainProgress>& diag, DeepLearning::ContinueFunction& cont, const Rcpp::IntegerVector& aSkip) {
	const std::vector<size_t> skip(Rcpp::as<std::vector<size_t>>(aSkip));
	aDBN.pretrain(aDataMatrix.transpose(), params, *diag, cont, skip);
	return aDBN;
}


/* TRAIN */

// [[Rcpp::export]]
DeepLearning::DeepBeliefNet trainDbnCpp(DeepLearning::DeepBeliefNet& aDBN, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix, const DeepLearning::TrainParameters& trainParams, const std::unique_ptr<DeepLearning::TrainProgress>& diag, const DeepLearning::ContinueFunction& cont) {
	aDBN.train(aDataMatrix.transpose(), trainParams, *diag, cont);
	return aDBN;
}

/* REVERSE */

// [[Rcpp::export]]
DeepLearning::RBM reverseRbmCpp(DeepLearning::RBM& anRBM) {
	return anRBM.reverse();
}

// [[Rcpp::export]]
DeepLearning::DeepBeliefNet reverseDbnCpp(DeepLearning::DeepBeliefNet& aDBN) {
	return aDBN.reverse();
}

/* Energy */

// [[Rcpp::export]]
DeepLearning::ArrayX1d energyRbmCpp(const DeepLearning::RBM& anRBM, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix) {
	return anRBM.energy(aDataMatrix.transpose());
}

// [[Rcpp::export]]
DeepLearning::ArrayX1d energyDbnCpp(const DeepLearning::DeepBeliefNet& aDBN, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix) {
	return aDBN.energy(aDataMatrix.transpose());
}

/* Error */

// [[Rcpp::export]]
DeepLearning::ArrayX1d errorRbmCpp(const DeepLearning::RBM& anRBM, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix) {
	return anRBM.error(aDataMatrix.transpose());
}

// [[Rcpp::export]]
DeepLearning::ArrayX1d errorDbnCpp(const DeepLearning::DeepBeliefNet& aDBN, const Eigen::Map<Eigen::MatrixXd>& aDataMatrix) {
	return aDBN.error(aDataMatrix.transpose());
}
