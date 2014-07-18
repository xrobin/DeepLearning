#pragma once

#include <Rcpp.h>
#include <RcppEigen.h> 

#include <vector>
#include <memory> // std::unique_ptr

#include <DeepLearning.h>


/* UNROLL */
DeepLearning::DeepBeliefNet unrollDbnCpp(DeepLearning::DeepBeliefNet&);

/* PREDICT */
Eigen::MatrixXd predictRbmCpp(const DeepLearning::RBM&, const Eigen::Map<Eigen::MatrixXd>&);
Eigen::MatrixXd predictDbnCpp(const DeepLearning::DeepBeliefNet&, const Eigen::Map<Eigen::MatrixXd>&);

/* RECONSTRUCT */
Eigen::MatrixXd reconstructRbmCpp(const DeepLearning::RBM&, const Eigen::Map<Eigen::MatrixXd>&);
Eigen::MatrixXd reconstructDbnCpp(const DeepLearning::DeepBeliefNet&, const Eigen::Map<Eigen::MatrixXd>&);

/* PRETRAIN */
DeepLearning::RBM pretrainRbmCpp(DeepLearning::RBM&, const Eigen::Map<Eigen::MatrixXd>&, const DeepLearning::PretrainParameters&, const std::unique_ptr<DeepLearning::PretrainProgress>&, const DeepLearning::ContinueFunction&);
DeepLearning::DeepBeliefNet pretrainDbnCpp(DeepLearning::DeepBeliefNet&, const Eigen::Map<Eigen::MatrixXd>&, const std::vector<DeepLearning::PretrainParameters>&, const std::unique_ptr<DeepLearning::PretrainProgress>&, DeepLearning::ContinueFunction&, const Rcpp::IntegerVector&);

/* TRAIN */
DeepLearning::DeepBeliefNet trainDbnCpp(DeepLearning::DeepBeliefNet&, const Eigen::Map<Eigen::MatrixXd>&, const DeepLearning::TrainParameters&, const std::unique_ptr<DeepLearning::TrainProgress>&, const DeepLearning::ContinueFunction&);

/* REVERSE */
DeepLearning::RBM reverseRbmCpp(DeepLearning::RBM&);
DeepLearning::DeepBeliefNet reverseDbnCpp(DeepLearning::DeepBeliefNet&);

/* Energy */
DeepLearning::ArrayX1d energyRbmCpp(const DeepLearning::RBM&, const Eigen::Map<Eigen::MatrixXd>&);
DeepLearning::ArrayX1d energyDbnCpp(const DeepLearning::DeepBeliefNet&, const Eigen::Map<Eigen::MatrixXd>&);

/* Error */
DeepLearning::ArrayX1d errorRbmCpp(const DeepLearning::RBM&, const Eigen::Map<Eigen::MatrixXd>&);
DeepLearning::ArrayX1d errorDbnCpp(const DeepLearning::DeepBeliefNet&, const Eigen::Map<Eigen::MatrixXd>&);
