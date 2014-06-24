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
 * - virtual void operator()(RBM& x, unsigned int i); // i = batch number
 * - virtual void setLayer(const size_t); // Keep track of the layer when pre-training a DBN
 * - virtual void setData(const Eigen::MatrixXd&); // Test dataset
 * - virtual void propagateData(const RBM&); // When a layer is trained, this function is called and updates the test data so that it can be used in the next layer
 * 
 * In TrainProgress:
 * - virtual void operator()(DeepBeliefNet& x, unsigned int i); // i = batch number
 * - virtual void setData(const Eigen::MatrixXd&); // Test dataset
 */

class PretrainProgress {
	public:
    virtual void operator()(RBM&, unsigned int) = 0;
    virtual void setLayer(const size_t) = 0;
    virtual void setData(const Eigen::MatrixXd&) = 0;
    virtual void propagateData(const RBM&) = 0; 
};

class TrainProgress {
	public:
    virtual void operator()(DeepBeliefNet&, unsigned int) = 0;
    virtual void setData(const Eigen::MatrixXd&) = 0;
};

/** NoOpPretrainProgress and NoOpTrainProgress are no-op implementations of PretrainProgress and TrainProgress.
 * They just do nothing and are used as default.
 */

class NoOpPretrainProgress: public PretrainProgress {
	public:
    void operator()(RBM&, unsigned int) {return;}
    void setLayer(const size_t) {return;}
    void setData(const Eigen::MatrixXd&) {return;}
    virtual void propagateData(const RBM&) {return;}
    static NoOpPretrainProgress& getInstance() {
    	static NoOpPretrainProgress anInstance;
    	return anInstance;
   	}
};

class NoOpTrainProgress: public TrainProgress {
	public:
    void operator()(DeepBeliefNet&, unsigned int) {return;}
    void setData(const Eigen::MatrixXd&) {return;}
    static NoOpTrainProgress& getInstance() {
    	static NoOpTrainProgress anInstance;
    	return anInstance;
   	}
};


/** PretrainProgressDo and TrainProgressDo are the functions that actually do something */
void PretrainProgressDo(RBM& anRBM, Eigen::MatrixXd& aTestData, unsigned int aBatch, size_t currentLayer, std::ostream& anOutputStream);
void TrainProgressDo(DeepBeliefNet& aDBN, Eigen::MatrixXd& aTestData, unsigned int aBatch, std::ostream& anOutputStream);


/** AcceleratePretrainProgress and AccelerateTrainProgress make less and less of the output as the (pre-)training progresses
 */

class AcceleratePretrainProgress: public PretrainProgress {
	private:
	double storeImage;
	unsigned int maxIters;
	size_t currentLayer;
	std::ofstream myOutput;
	Eigen::MatrixXd testData;
	constexpr static const double InitialStoreImage = 1; // initialize storeImage to 1
	//AcceleratePretrainProgress(AcceleratePretrainProgress&) = delete;
	
	public:
	explicit AcceleratePretrainProgress(const char* aFileName = "pretrainProgress.txt"): storeImage(InitialStoreImage), maxIters(0), currentLayer(0), myOutput(aFileName) {};
	explicit AcceleratePretrainProgress(unsigned int aMaxIters, const char* aFileName = "pretrainProgress.txt"): storeImage(InitialStoreImage), maxIters(aMaxIters), currentLayer(0), myOutput(aFileName) {};
	explicit AcceleratePretrainProgress(size_t aLayer, const char* aFileName = "pretrainProgress.txt"): storeImage(InitialStoreImage), maxIters(0), currentLayer(aLayer), myOutput(aFileName) {};
	explicit AcceleratePretrainProgress(size_t aLayer, unsigned int aMaxIters, const char* aFileName = "pretrainProgress.txt"): storeImage(InitialStoreImage), maxIters(aMaxIters), currentLayer(aLayer), myOutput(aFileName) {};
	
    void operator()(RBM& anRBM, unsigned int aBatch) {
    	if (storeImage >= 1 || aBatch == maxIters) {
    		PretrainProgressDo(anRBM, testData, aBatch, currentLayer, myOutput);
    		storeImage = 100.0 / aBatch;
    	}
    	storeImage += 100.0 / aBatch;
    }
    void setLayer(const size_t aLayer) {currentLayer = aLayer; storeImage = InitialStoreImage;}
    void setData(const Eigen::MatrixXd& aTestData) {testData = aTestData;}
    void propagateData(const RBM& anRBM);
};

class AccelerateTrainProgress: public TrainProgress {
	private:
	double storeImage;
	unsigned int maxIters;
	std::ofstream myOutput;
	Eigen::MatrixXd testData;
	constexpr static const double InitialStoreImage = 1; // initialize storeImage to 1
	//AccelerateTrainProgress(AccelerateTrainProgress&) = delete;
	
	public:
	explicit AccelerateTrainProgress(const char* aFileName = "trainProgress.txt"): storeImage(InitialStoreImage), maxIters(0), myOutput(aFileName) {};
	explicit AccelerateTrainProgress(unsigned int aMaxIters, const char* aFileName = "trainProgress.txt"): storeImage(InitialStoreImage), maxIters(aMaxIters), myOutput(aFileName) {};
	
    void operator()(DeepBeliefNet& aDBN, unsigned int aBatch) {
    	if (storeImage >= 1 || aBatch == maxIters) {
    		TrainProgressDo(aDBN, testData, aBatch, myOutput);
    		storeImage = 100.0 / aBatch;
    	}
    	storeImage += 100.0 / aBatch;
    }
    void setData(const Eigen::MatrixXd& aTestData) {testData = aTestData;}
};


/** EachStepPretrainProgress and EachStepTrainProgress perform the output at each step of the (pre-)training
 */
class EachStepPretrainProgress: public PretrainProgress {
	private:
	size_t currentLayer;
	std::ofstream myOutput;
	Eigen::MatrixXd testData;
	//EachStepPretrainProgress(EachStepPretrainProgress&) = delete; // no copy - or we'll run into trouble!
	
	public:
	explicit EachStepPretrainProgress(const char* aFileName = "pretrainProgress.txt"): currentLayer(0), myOutput(aFileName) {};
	explicit EachStepPretrainProgress(size_t aLayer, const char* aFileName = "trainProgress.txt"): currentLayer(aLayer), myOutput(aFileName) {};
	
    void operator()(RBM& anRBM, unsigned int aBatch) {
    	PretrainProgressDo(anRBM, testData, aBatch, currentLayer, myOutput);
    }
    void setLayer(const size_t aLayer) {currentLayer = aLayer;}
    void setData(const Eigen::MatrixXd& aTestData) {testData = aTestData;}
    void propagateData(const RBM& anRBM);
};

class EachStepTrainProgress: public TrainProgress {
	private:
	std::ofstream myOutput;
	Eigen::MatrixXd testData;
	//EachStepTrainProgress(EachStepTrainProgress&) = delete; // no copy - or we'll run into trouble!

	public:
	explicit EachStepTrainProgress(const char* aFileName = "trainProgress.txt"): myOutput(aFileName) {};
    void operator()(DeepBeliefNet& aDBN, unsigned int aBatch) {
    	TrainProgressDo(aDBN, testData, aBatch, myOutput);
    }
    void setData(const Eigen::MatrixXd& aTestData) {testData = aTestData;}
};

template<class T>
void GenericProgressDo(T& aNetwork, Eigen::MatrixXd& aTestData, std::string aDiagString, std::ostream& anOutputStream) {
	// Predict && determine error:
	const Eigen::MatrixXd predictions = aNetwork.predict(aTestData);
	const double error = aNetwork.errorSum(aTestData, aNetwork.reverse_predict(predictions));
	// Get size of predictions
	const Eigen_size_type predictSize = predictions.size();
	
	// Print
	anOutputStream <<  aDiagString << ", " << error;
	if (predictSize < 1000) {
		anOutputStream << ", " << predictSize << ": ";
		const double *predictData = predictions.data();
		for (Eigen_size_type i = 0; i < predictSize; ++i) {
			anOutputStream << predictData[i] << ", ";
		}
	}
	anOutputStream << std::endl;
}