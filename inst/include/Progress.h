#pragma once

#include <Eigen/Dense>

#include <fstream>
#include <string>

class RBM;
class DeepBeliefNet;

#include <typedefs.h>

/** PretrainProgress and TrainProgress are abstract base functor classes that are called at the end of each iteration of contrastive divergence
 * (PretrainProgress) and conjugate gradient (TrainProgress).
 * 
 *
 * In PretrainProgress, the following members must be implemented:
 * - virtual void operator()(const RBM&, const Eigen::MatrixXd& b, const unsigned int i); // b = batch; i = batch number
 * - virtual void setLayer(const size_t); // Keep track of the layer when pre-training a DBN
 * - virtual void setBatchSize(const size_t); // Keep track of the batch size
 * - virtual void setMaxIters(const unsigned int); // Keep track of the layer max # iterations
 * - virtual void setData(const Eigen::MatrixXd&); // Test dataset
 * - virtual void propagateData(const RBM&); // When a layer is trained, this function is called and updates the test data so that it can be used in the next layer
 * - virtual void setFunction(const pretrainDiagFunctionType&); // a function to evaluate
 * - virtual void reset(); // restarts the counter if any
 * 
 * In TrainProgress:
 * - virtual void operator()(const DeepBeliefNet&, const Eigen::MatrixXd& b, const unsigned int i); // b = batch; i = batch number
 * - virtual void setBatchSize(const size_t); // Keep track of the batch size
 * - virtual void setMaxIters(const unsigned int); // Keep track of the layer max # iterations
 * - virtual void setData(const Eigen::MatrixXd&); // Test dataset
 * - virtual void setFunction(const trainDiagFunctionType&); // a function to evaluate
 * - virtual void reset(); // restarts the counter if any
 */
                                                 
class PretrainProgress {
	public:
    virtual void operator()(const RBM&, const Eigen::MatrixXd&, const unsigned int) = 0;
    virtual void setLayer(const size_t) = 0;
    virtual void setBatchSize(const size_t) = 0;
    virtual void setMaxIters(const unsigned int) = 0;
    virtual void setData(const Eigen::MatrixXd&) = 0;
    virtual void setFunction(const pretrainDiagFunctionType&) = 0;
    virtual void propagateData(const RBM&) = 0; 
    virtual void reset() = 0; 
};

class TrainProgress {
	public:
    virtual void operator()(const DeepBeliefNet&, const Eigen::MatrixXd&, const unsigned int) = 0;
    virtual void setBatchSize(const size_t) = 0;
    virtual void setMaxIters(const unsigned int) = 0;
    virtual void setData(const Eigen::MatrixXd&) = 0;
    virtual void setFunction(const trainDiagFunctionType&) = 0;
    virtual void reset() = 0; 
};

/** NoOpPretrainProgress and NoOpTrainProgress are no-op implementations of PretrainProgress and TrainProgress.
 * They just do nothing and are used as default.
 */

class NoOpPretrainProgress: public PretrainProgress {
	public:
    void operator()(const RBM&, const Eigen::MatrixXd&, const unsigned int) {return;}
    void setLayer(const size_t) {return;}
    void setBatchSize(const size_t) {return;}
    void setMaxIters(const unsigned int) {return;}
    void setData(const Eigen::MatrixXd&) {return;}
    void propagateData(const RBM&) {return;}
    void setFunction(const pretrainDiagFunctionType&) {return;}
    void reset() {return;}
    static NoOpPretrainProgress& getInstance() {
    	static NoOpPretrainProgress anInstance;
    	return anInstance;
   	}
};

class NoOpTrainProgress: public TrainProgress {
	public:
    void operator()(const DeepBeliefNet&, const Eigen::MatrixXd&, const unsigned int) {return;};
    void setBatchSize(const size_t) {return;}
    void setMaxIters(const unsigned int) {return;}
    void setData(const Eigen::MatrixXd&) {return;}
    void setFunction(const trainDiagFunctionType&) {return;}
    void reset() {return;}
    static NoOpTrainProgress& getInstance() {
    	static NoOpTrainProgress anInstance;
    	return anInstance;
   	}
};

/** AcceleratePretrainProgress and AccelerateTrainProgress make less and less of the output as the (pre-)training progresses
 */

class AcceleratePretrainProgress: public PretrainProgress {
	private:
	double storeImage;
	unsigned int maxIters;
	size_t currentLayer, batchSize;
	Eigen::MatrixXd testData;
	pretrainDiagFunctionType function;
	constexpr static const double InitialStoreImage = 1; // initialize storeImage to 1
	
	public:	
    void operator()(const RBM& anRBM, const Eigen::MatrixXd& aBatch, const unsigned int iter) {
    	if (storeImage >= 1 || iter == maxIters) {
    		function(anRBM, aBatch, testData, iter, batchSize, maxIters, currentLayer);
    		storeImage = 100.0 / iter;
    	}
    	storeImage += 100.0 / iter;
    }
    void setLayer(const size_t aLayer) {currentLayer = aLayer;}
    void setData(const Eigen::MatrixXd& aTestData) {testData = aTestData;}
    void propagateData(const RBM& anRBM);
    void setBatchSize(const size_t aBatchSize) {batchSize = aBatchSize;}
    void setMaxIters(const unsigned int aMaxIters) {maxIters = aMaxIters;}
    void setFunction(const pretrainDiagFunctionType& aFunction) {function = aFunction;}
    void reset() {storeImage = InitialStoreImage;};
};

class AccelerateTrainProgress: public TrainProgress {
	private:
	double storeImage;
	unsigned int maxIters;
	size_t batchSize;
	Eigen::MatrixXd testData;
	trainDiagFunctionType function;
	constexpr static const double InitialStoreImage = 1; // initialize storeImage to 1
	//AccelerateTrainProgress(AccelerateTrainProgress&) = delete;
	
	public:	
    void operator()(const DeepBeliefNet& aDBN, const Eigen::MatrixXd &aBatch, unsigned int iter) {
    	if (storeImage >= 1 || iter == maxIters) {
    		function(aDBN, aBatch, testData, iter, batchSize, maxIters);
    		storeImage = 100.0 / iter;
    	}
    	storeImage += 100.0 / iter;
    }
    void setData(const Eigen::MatrixXd& aTestData) {testData = aTestData;}
    void setBatchSize(const size_t aBatchSize) {batchSize = aBatchSize;}
    void setMaxIters(const unsigned int aMaxIters) {maxIters = aMaxIters;}
    void setFunction(const trainDiagFunctionType& aFunction) {function = aFunction;}
    void reset() {storeImage = InitialStoreImage;};
};


/** EachStepPretrainProgress and EachStepTrainProgress perform the output at each step of the (pre-)training
 */
class EachStepPretrainProgress: public PretrainProgress {
	private:
	unsigned int maxIters;
	size_t currentLayer, batchSize;
	Eigen::MatrixXd testData;
	pretrainDiagFunctionType function;
	
	public:
	void operator()(const RBM& anRBM, const Eigen::MatrixXd& aBatch, const unsigned int iter) {
    	function(anRBM, aBatch, testData, iter, batchSize, maxIters, currentLayer);
    }
    
    void setLayer(const size_t aLayer) {currentLayer = aLayer;}
    void setBatchSize(const size_t aBatchSize) {batchSize = aBatchSize;}
    void setMaxIters(const unsigned int aMaxIters) {maxIters = aMaxIters;}
    void setData(const Eigen::MatrixXd& aTestData) {testData = aTestData;}
    void propagateData(const RBM& anRBM);
    void setFunction(const pretrainDiagFunctionType& aFunction) {function = aFunction;}
    void reset() {return;};
};

class EachStepTrainProgress: public TrainProgress {
	private:
	unsigned int maxIters;
	size_t batchSize;
	Eigen::MatrixXd testData;
	trainDiagFunctionType function;

	public:
    void operator()(const DeepBeliefNet& aDBN, const Eigen::MatrixXd &aBatch, unsigned int iter) {
    	function(aDBN, aBatch, testData, iter, batchSize, maxIters);
    }
    void setBatchSize(const size_t aBatchSize) {batchSize = aBatchSize;}
    void setMaxIters(const unsigned int aMaxIters) {maxIters = aMaxIters;}
    void setData(const Eigen::MatrixXd& aTestData) {testData = aTestData;}
    void setFunction(const trainDiagFunctionType& aFunction) {function = aFunction;}
    void reset() {return;};
};
