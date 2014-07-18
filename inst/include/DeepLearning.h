#pragma once 

// Helper stuff
#include <DeepLearning/typedefs.h>
#include <DeepLearning/utils.h>
#include <shared_array_ptr.h>

// Training stuff
#include <DeepLearning/TrainParameters.h>
#include <DeepLearning/PretrainParameters.h>
#include <DeepLearning/Progress.h> // Progress tracking function

// Core DBN stuff
#include <DeepLearning/Layer.h>
#include <DeepLearning/RBM.h>
#include <DeepLearning/DeepBeliefNet.h>

// Conversions from/to R
#include <RcppEigen.h> // This is used for conversions in RcppExports.cpp
#include <RcppConversions.h>
#include <RcppExports.h>
