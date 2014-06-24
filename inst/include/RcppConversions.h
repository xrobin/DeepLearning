#include <Rcpp.h>

#include <memory> // std::unique_ptr

#include "DeepBeliefNet.h" 
#include "RBM.h"
#include "Layer.h"
#include "shared_array_ptr.h"

namespace Rcpp {
	// size_t
	template <> size_t as(SEXP ptr);
	
	// shared_array_ptr
	template <> shared_array_ptr<double> as(SEXP ptr);
	template <> SEXP wrap(const shared_array_ptr<double> &ptr);
	
	// Offsets (weights.breaks)
	// This is computed automaticall on import, so no as
	//template <> TrainParameters as(SEXP params);
	template <> SEXP wrap(const offsets &offsets);
	SEXP wrap_with_additionalOffset(const offsets &someOffsets, size_t additionalOffset);

	// Layer
	template <> Layer as(SEXP layer);
	template <> SEXP wrap(const Layer &layer);

	// RBM
	template <> RBM as(SEXP rbm);
	template <> SEXP wrap(const RBM &rbm);
	/** Special wrap with an environment. Used in DBNs where env must be shared */
	SEXP wrap_with_env(const RBM& rbm, const Rcpp::Environment& env, size_t additionalOffset = 0);
	
	// DeepBeliefNet
	template <> DeepBeliefNet as(SEXP dbn);
	template <> SEXP wrap(const DeepBeliefNet &dbn);
	
	// PretrainParameters
	template <> PretrainParameters as(SEXP params);
	template <> std::vector<PretrainParameters> as(SEXP params);
	// no need to return so no wrap
	// template <> SEXP wrap(const PretrainParameters &params);
	
	// TrainParameters
	template <> TrainParameters as(SEXP params);
	template <> CgMinParams as(SEXP params);
	// no need to return so no wrap
	// template <> SEXP wrap(const TrainParameters &params);
	
	// PretrainProgress
	template <> std::unique_ptr<PretrainProgress> as(SEXP diag);
	// no need to return so no wrap
	// template <> SEXP wrap(const PretrainProgress &diag);
	
	// TrainProgress
	template <> std::unique_ptr<TrainProgress> as(SEXP diag);
	// no need to return so no wrap
	// template <> SEXP wrap(const TrainProgress &diag);
}

