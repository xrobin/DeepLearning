#pragma once

#include <Rcpp.h>
#include <RcppEigen.h> 

#include <vector>
#include <memory> // std::unique_ptr

#include "Layer.h"
#include "RBM.h"
#include "DeepBeliefNet.h"

/* UNROLL */
DeepBeliefNet unrollDbnCpp(DeepBeliefNet&);

/* PREDICT */
Eigen::MatrixXd predictRbmCpp(const RBM&, const Eigen::Map<Eigen::MatrixXd>&);
Eigen::MatrixXd predictDbnCpp(const DeepBeliefNet&, const Eigen::Map<Eigen::MatrixXd>&);

/* RECONSTRUCT */
Eigen::MatrixXd reconstructRbmCpp(const RBM&, const Eigen::Map<Eigen::MatrixXd>&);
Eigen::MatrixXd reconstructDbnCpp(const DeepBeliefNet&, const Eigen::Map<Eigen::MatrixXd>&);

/* PRETRAIN */
RBM pretrainRbmCpp(RBM&, const Eigen::Map<Eigen::MatrixXd>&, const PretrainParameters&, const std::unique_ptr<PretrainProgress>&, const ContinueFunction&);
DeepBeliefNet pretrainDbnCpp(DeepBeliefNet&, const Eigen::Map<Eigen::MatrixXd>&, const std::vector<PretrainParameters>&, const std::unique_ptr<PretrainProgress>&, ContinueFunction&, const Rcpp::IntegerVector&);

/* TRAIN */
DeepBeliefNet trainDbnCpp(DeepBeliefNet&, const Eigen::Map<Eigen::MatrixXd>&, const TrainParameters&, const std::unique_ptr<TrainProgress>&, const ContinueFunction&);

/* REVERSE */
RBM reverseRbmCpp(RBM&);
DeepBeliefNet reverseDbnCpp(DeepBeliefNet&);

/* Energy */
ArrayX1d energyRbmCpp(const RBM&, const Eigen::Map<Eigen::MatrixXd>&);
ArrayX1d energyDbnCpp(const DeepBeliefNet&, const Eigen::Map<Eigen::MatrixXd>&);

/* Error */
ArrayX1d errorRbmCpp(const RBM&, const Eigen::Map<Eigen::MatrixXd>&);
ArrayX1d errorDbnCpp(const DeepBeliefNet&, const Eigen::Map<Eigen::MatrixXd>&);
