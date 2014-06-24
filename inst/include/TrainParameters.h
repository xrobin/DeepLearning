#pragma once 

#include <algorithm> // std::find, std::tolower
#include <functional> // std::function
#include <vector>
#include <string>
#include <stdexcept> // std::invalid_argument
#include <limits>

#include <typedefs.h> // UNUSED(variables)


/** Optimization parameters for R's cgmin. 
 * There are a few parameters and a lot of magic numbers we need to be able to change:
 * 
 * 			params.abstol, // abstol, tolerance on the error, final
			params.intol, // intol, minimal reduction for step (reltol in ?optim)
		
			params.trace, // trace, boolean (verbose)
			params.maxit // maxit 
			
			
 *     - algorithmType: 1 (FR), 2 (PR) or 3 (BS)
 *       Throws an std::invalid_argument if the argument supplied is not in this list. You can also use the availableAlgorithmTypes member to check it yourself.
 *     - trace: integer interpreted as boolean meaning vebose?
 *     - maxCgIters: default 10; number of iterations of the CG (per maxiter)
 *     - abstol: absolute tolerance, on the final error
 *     - intol: relative tolerance (reltol in ?optim), on the improvement at a particular step
 * 
 * All members can be set directly or trough the set* functions.
 * 
 */
struct CgMinParams {
	int type, trace;
	unsigned int maxCgIters;
	double steplength, stepredn, acctol, reltest, abstol, intol, setstep;

	CgMinParams(): type(2), trace(0), maxCgIters(10), steplength(1.0), stepredn(0.2), acctol(0.0001), reltest(10.0), abstol(-std::numeric_limits<double>::infinity()), 
		intol(sqrt(std::numeric_limits<double>::epsilon())), setstep(1.7) {};

	CgMinParams& setAlgorithmType(int newAlgorithmType) {
		if (newAlgorithmType < 1 || newAlgorithmType > 3) {
			throw std::invalid_argument("Unknown algorithm type");
		}
		type = newAlgorithmType;
		return *this;
	}
	CgMinParams& setAlgorithmType(std::string newAlgorithmType) {
		std::transform(newAlgorithmType.begin(), newAlgorithmType.end(), newAlgorithmType.begin(), ::tolower);
		if (newAlgorithmType == "fletcher-reeves" || newAlgorithmType == "fr") {
			setAlgorithmType(1);
		}
		else if (newAlgorithmType == "polak-ribiere" || newAlgorithmType == "pr") {
			setAlgorithmType(2);
		}
		else if (newAlgorithmType == "beale-sorenson" || newAlgorithmType == "bs") {
			setAlgorithmType(3);
		}
		else {
			throw std::invalid_argument("Unknown algorithm type");
		}
		return *this;
	}
	
	CgMinParams& setTrace(int newTrace) {trace = newTrace; return *this;}
	CgMinParams& setMaxCgIters(unsigned int newMaxCgIters) {maxCgIters = newMaxCgIters; return *this;}
	CgMinParams& setStepLength(double newStepLength) {steplength = newStepLength; return *this;}
	CgMinParams& setSteredn(double newSteredn) {stepredn = newSteredn; return *this;}
	CgMinParams& setReltest(double newReltest) {reltest = newReltest; return *this;}
	CgMinParams& setAcctol(double newAcctol) {acctol = newAcctol; return *this;}
	CgMinParams& setAbstol(double newAbstol) {abstol = newAbstol; return *this;}
	CgMinParams& setIntol(double newIntol) {intol = newIntol; return *this;}
	CgMinParams& setSetstep(double newSetstep) {setstep = newSetstep; return *this;}
};

/**
 * Structure defining the parameters for the training
 * Contains the following members:
 *	 - size_t batchsize: default 100; The size of the batches
 *	 - unsigned int minIters: default 100; The minimum number of iterations of the algorithm (number of batches we draw)
 *	 - unsigned int maxIters: default 1000; The maximum number of iterations of the algorithm (number of batches we draw)
 *   - unsigned int nProcs: default 0 (for Eigen, special value = no parallel execution)
 *   - cgMinParams: optimization parameters for the conjugate gradient algorithm. An object of class CgMinParams.
 * 
 * All members can be set directly or trough the set* functions.
 * 
 */

struct TrainParameters {
	typedef std::function<bool(std::vector<double>, unsigned int, size_t)> continueFunctionType;

	CgMinParams myCgMinParams;
	size_t batchSize;
	int nbThreads;
	unsigned int minIters, maxIters, continueFunctionFrequency, continueStopLimit;
	continueFunctionType continueFunction;

	TrainParameters& setCgMinParams(const CgMinParams& newcgMinParams) {myCgMinParams = newcgMinParams; return *this;}
	TrainParameters& setBatchSize(size_t newBatchSize) {batchSize = newBatchSize; return *this;}
	TrainParameters& setNbThreads(int newNbThreads) {nbThreads = newNbThreads; return *this;}
	TrainParameters& setMinIters(unsigned int newMinIters) {minIters = newMinIters; return *this;}
	TrainParameters& setMaxIters(unsigned int newMaxIters) {maxIters = newMaxIters; return *this;}
	TrainParameters& setContinueFunctionFrequency(unsigned int newContinueFunctionFrequency) {continueFunctionFrequency = newContinueFunctionFrequency; return *this;}
	TrainParameters& setContinueStopLimit(unsigned int newContinueStopLimit) {continueStopLimit = newContinueStopLimit; return *this;}
	TrainParameters& setContinueFunction(continueFunctionType newContinueFunction) {continueFunction = newContinueFunction; return *this;}

	TrainParameters() : myCgMinParams(), batchSize(100), nbThreads(0), minIters(100), maxIters(1000), continueFunctionFrequency(100), continueStopLimit(3),
	continueFunction([](std::vector<double> errors, unsigned int iter, size_t batchsize) {
		UNUSED(errors); UNUSED(iter); UNUSED(batchsize);
		return true;
	})
	{};
};

