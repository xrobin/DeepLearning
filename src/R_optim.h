#pragma once

#include <TrainParameters.h>
#include <DeepBeliefNet.h>
#include <RBM.h>

/** Parameters passed to the optimization functions */
struct OptimParameters {
	OptimParameters(DeepBeliefNet &aDBN, Eigen::MatrixXd &aMatrix): dbn(aDBN), batch(aMatrix), gradientRBMs() {}
	DeepBeliefNet &dbn;
	Eigen::MatrixXd &batch;
	vector<RBM> gradientRBMs;
};

/** Optimization function typedefs */
typedef double optimfn(OptimParameters&);
typedef void optimgr(double *, OptimParameters&);

/** contrasted divergence minimzer */
void cgmin(size_t n, double *Bvec, double *X, double *Fmin,
           optimfn fminfn, optimgr fmingr, int *fail,
           const CgMinParams& params, OptimParameters& ex,
           unsigned int *fncount, unsigned int *grcount);
	   