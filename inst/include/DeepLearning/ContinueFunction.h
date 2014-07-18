#pragma once 

#include <vector>

#include <DeepLearning/typedefs.h> // continueFunctionType


namespace DeepLearning {
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
	
	struct ContinueFunction {
		
		size_t layer;
		unsigned int frequency, limit;
		continueFunctionType continueFunction;
	
		ContinueFunction& setLayer(size_t newLayer) {layer = newLayer; return *this;}
		ContinueFunction& setFrequency(unsigned int newFrequency) {frequency = newFrequency; return *this;}
		ContinueFunction& setLimit(unsigned int newLimit) {limit = newLimit; return *this;}
		ContinueFunction& setContinueFunction(continueFunctionType newContinueFunction) {continueFunction = newContinueFunction; return *this;}
	
		ContinueFunction() : layer(0), frequency(100), limit(3),
		// Default continueFunction: do nothing (ie return true and always continue)!
			continueFunction([](std::vector<double> errors, unsigned int iter, size_t batchsize, unsigned int maxiters, size_t aLayer) {
			UNUSED(errors); UNUSED(iter); UNUSED(batchsize); UNUSED(maxiters); UNUSED(aLayer);
			return true;
		})
		{}
		
		bool operator()(std::vector<double> errors, unsigned int iter, size_t batchsize, unsigned int maxiters) const {
			return continueFunction(errors, iter, batchsize, maxiters, this->layer);
		}
		
	    static ContinueFunction& getInstance() {
	    	static ContinueFunction anInstance; // A default instance that does nothing, see default for continueFunction
	    	return anInstance;
	   	}
	};
}
