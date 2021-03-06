# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

unrollDbnCpp <- function(aDBN) {
    .Call('_DeepLearning_unrollDbnCpp', PACKAGE = 'DeepLearning', aDBN)
}

predictRbmCpp <- function(anRBM, aDataMatrix) {
    .Call('_DeepLearning_predictRbmCpp', PACKAGE = 'DeepLearning', anRBM, aDataMatrix)
}

predictDbnCpp <- function(aDBN, aDataMatrix) {
    .Call('_DeepLearning_predictDbnCpp', PACKAGE = 'DeepLearning', aDBN, aDataMatrix)
}

sampleRbmCpp <- function(anRBM, aDataMatrix) {
    .Call('_DeepLearning_sampleRbmCpp', PACKAGE = 'DeepLearning', anRBM, aDataMatrix)
}

sampleDbnCpp <- function(aDBN, aDataMatrix) {
    .Call('_DeepLearning_sampleDbnCpp', PACKAGE = 'DeepLearning', aDBN, aDataMatrix)
}

reconstructRbmCpp <- function(anRBM, aDataMatrix) {
    .Call('_DeepLearning_reconstructRbmCpp', PACKAGE = 'DeepLearning', anRBM, aDataMatrix)
}

reconstructDbnCpp <- function(aDBN, aDataMatrix) {
    .Call('_DeepLearning_reconstructDbnCpp', PACKAGE = 'DeepLearning', aDBN, aDataMatrix)
}

pretrainRbmCpp <- function(anRBM, aDataMatrix, params, diag, cont) {
    .Call('_DeepLearning_pretrainRbmCpp', PACKAGE = 'DeepLearning', anRBM, aDataMatrix, params, diag, cont)
}

pretrainDbnCpp <- function(aDBN, aDataMatrix, params, diag, cont, aSkip) {
    .Call('_DeepLearning_pretrainDbnCpp', PACKAGE = 'DeepLearning', aDBN, aDataMatrix, params, diag, cont, aSkip)
}

trainDbnCpp <- function(aDBN, aDataMatrix, trainParams, diag, cont) {
    .Call('_DeepLearning_trainDbnCpp', PACKAGE = 'DeepLearning', aDBN, aDataMatrix, trainParams, diag, cont)
}

reverseRbmCpp <- function(anRBM) {
    .Call('_DeepLearning_reverseRbmCpp', PACKAGE = 'DeepLearning', anRBM)
}

reverseDbnCpp <- function(aDBN) {
    .Call('_DeepLearning_reverseDbnCpp', PACKAGE = 'DeepLearning', aDBN)
}

energyRbmCpp <- function(anRBM, aDataMatrix) {
    .Call('_DeepLearning_energyRbmCpp', PACKAGE = 'DeepLearning', anRBM, aDataMatrix)
}

energyDbnCpp <- function(aDBN, aDataMatrix) {
    .Call('_DeepLearning_energyDbnCpp', PACKAGE = 'DeepLearning', aDBN, aDataMatrix)
}

errorRbmCpp <- function(anRBM, aDataMatrix) {
    .Call('_DeepLearning_errorRbmCpp', PACKAGE = 'DeepLearning', anRBM, aDataMatrix)
}

errorDbnCpp <- function(aDBN, aDataMatrix) {
    .Call('_DeepLearning_errorDbnCpp', PACKAGE = 'DeepLearning', aDBN, aDataMatrix)
}

errorSumRbmCpp <- function(anRBM, aDataMatrix) {
    .Call('_DeepLearning_errorSumRbmCpp', PACKAGE = 'DeepLearning', anRBM, aDataMatrix)
}

errorSumDbnCpp <- function(aDBN, aDataMatrix) {
    .Call('_DeepLearning_errorSumDbnCpp', PACKAGE = 'DeepLearning', aDBN, aDataMatrix)
}

extractRbmWCpp <- function(anRBM) {
    .Call('_DeepLearning_extractRbmWCpp', PACKAGE = 'DeepLearning', anRBM)
}

extractRbmCCpp <- function(anRBM) {
    .Call('_DeepLearning_extractRbmCCpp', PACKAGE = 'DeepLearning', anRBM)
}

extractRbmBCpp <- function(anRBM) {
    .Call('_DeepLearning_extractRbmBCpp', PACKAGE = 'DeepLearning', anRBM)
}

setRbmWCpp <- function(anRBM, aNewW) {
    .Call('_DeepLearning_setRbmWCpp', PACKAGE = 'DeepLearning', anRBM, aNewW)
}

setRbmCCpp <- function(anRBM, aNewC) {
    .Call('_DeepLearning_setRbmCCpp', PACKAGE = 'DeepLearning', anRBM, aNewC)
}

setRbmBCpp <- function(anRBM, aNewB) {
    .Call('_DeepLearning_setRbmBCpp', PACKAGE = 'DeepLearning', anRBM, aNewB)
}

detectCores <- function() {
    .Call('_DeepLearning_detectCores', PACKAGE = 'DeepLearning')
}

unit_DbnGradient <- function(aDBN, aDataMatrix) {
    .Call('_DeepLearning_unit_DbnGradient', PACKAGE = 'DeepLearning', aDBN, aDataMatrix)
}

