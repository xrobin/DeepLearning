#pragma once

#include <DeepLearning.h>
#include <Rcpp.h>

// detectCores
RcppExport SEXP DeepLearning_detectCores();
// unrollDbnCpp
RcppExport SEXP DeepLearning_unrollDbnCpp(SEXP aDBNSEXP);
// predictRbmCpp
RcppExport SEXP DeepLearning_predictRbmCpp(SEXP anRBMSEXP, SEXP aDataMatrixSEXP);
// predictDbnCpp
RcppExport SEXP DeepLearning_predictDbnCpp(SEXP aDBNSEXP, SEXP aDataMatrixSEXP);
// reconstructRbmCpp
RcppExport SEXP DeepLearning_reconstructRbmCpp(SEXP anRBMSEXP, SEXP aDataMatrixSEXP);
// reconstructDbnCpp
RcppExport SEXP DeepLearning_reconstructDbnCpp(SEXP aDBNSEXP, SEXP aDataMatrixSEXP);
// pretrainRbmCpp
RcppExport SEXP DeepLearning_pretrainRbmCpp(SEXP anRBMSEXP, SEXP aDataMatrixSEXP, SEXP paramsSEXP, SEXP diagSEXP, SEXP contSEXP);
// pretrainDbnCpp
RcppExport SEXP DeepLearning_pretrainDbnCpp(SEXP aDBNSEXP, SEXP aDataMatrixSEXP, SEXP paramsSEXP, SEXP diagSEXP, SEXP contSEXP, SEXP aSkipSEXP);
// trainDbnCpp
RcppExport SEXP DeepLearning_trainDbnCpp(SEXP aDBNSEXP, SEXP aDataMatrixSEXP, SEXP trainParamsSEXP, SEXP diagSEXP, SEXP contSEXP);
// reverseRbmCpp
RcppExport SEXP DeepLearning_reverseRbmCpp(SEXP anRBMSEXP);
// reverseDbnCpp
RcppExport SEXP DeepLearning_reverseDbnCpp(SEXP aDBNSEXP);
// energyRbmCpp
RcppExport SEXP DeepLearning_energyRbmCpp(SEXP anRBMSEXP, SEXP aDataMatrixSEXP);
// energyDbnCpp
RcppExport SEXP DeepLearning_energyDbnCpp(SEXP aDBNSEXP, SEXP aDataMatrixSEXP);
// errorRbmCpp
RcppExport SEXP DeepLearning_errorRbmCpp(SEXP anRBMSEXP, SEXP aDataMatrixSEXP);
// errorDbnCpp
RcppExport SEXP DeepLearning_errorDbnCpp(SEXP aDBNSEXP, SEXP aDataMatrixSEXP);
// unit_DbnGradient
RcppExport SEXP DeepLearning_unit_DbnGradient(SEXP aDBNSEXP, SEXP aDataMatrixSEXP);
