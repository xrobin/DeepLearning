#pragma once 

// Helper stuff
#include <typedefs.h>
#include <utils.h>
#include <shared_array_ptr.h>

// Training stuff
#include <TrainParameters.h>
#include <PretrainParameters.h>
#include <Progress.h> // Progress tracking function

// Core DBN stuff
#include <Layer.h>
#include <RBM.h>
#include <DeepBeliefNet.h>

// Conversions from/to R
#include <RcppEigen.h> // This is used for conversions in RcppExports.cpp
#include <RcppConversions.h>
#include <RcppExports.h>
