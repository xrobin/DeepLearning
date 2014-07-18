/* This file implements the training (conjugate gradients) part of the DeepBeliefNet.
*/
#undef NDEBUG
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::ArrayXXd;
#include <Rcpp.h> // Rcpp::checkUserInterrupt
#include "boost/numeric/conversion/cast.hpp"

#include <iostream>
#include <stdexcept> // std::runtime_error
#include <string>
#include <utility> // std::size_t
#include <vector>

using std::vector;
using std::cout;
using std::endl;
using std::size_t;
using std::string;

#include "R_optim.h" // cgmin
#include "Random.h"
#include "shared_array_ptr.h"
#include "DeepBeliefNet.h"
#include "typedefs.h"

double my_f (OptimParameters&);
double my_f (OptimParameters& params) {
	DeepBeliefNet& dbn = params.dbn;
	Eigen::MatrixXd& batch = params.batch;
	
	// Pass the data to compute error
	double f = dbn.errorSum(batch);
	return f;
}


/** The gradient of f, df = (df/dx, df/dy). 
 * *df: the gradients
 * *rawParams: additional OptimParameters object passad as void pointer
 */
void my_df (double *df, OptimParameters&);
void my_df (double *df, OptimParameters& params) {
	DeepBeliefNet& dbn = params.dbn;
	Eigen::MatrixXd& batch = params.batch;
	vector<RBM>& gradientRBMs = params.gradientRBMs;
	
	// Also apply the data to the gradientRBMs vector if needed
	if (gradientRBMs.empty() || df != gradientRBMs[0].getData().data()) {
		std::cout << "Applying gradient to gradientRBMs" << std::endl;
		shared_array_ptr<double> newData(df, dbn.getData().size(), false);
		DeepBeliefNet::constructRBMs(gradientRBMs, dbn.getLayers(), newData);
	}

	dbn.getGradient(batch, gradientRBMs);
}


/** Returns the gradient of the DeepBeliefNet related with the provided data in a vector<RBM>
 * This gradient can be used for backpropagation or other puroposes
 */
vector<RBM> DeepBeliefNet::getGradient(const MatrixXd& data, shared_array_ptr<double> df) {
	vector<RBM> gradientRBMs; // the Bolzman Machines
	constructRBMs(gradientRBMs, myLayers, df);
	getGradient(data, gradientRBMs);
	return gradientRBMs;
}


/** Derivative activation function of a binary layer */
MatrixXd binaryActivationDerivative(MatrixXd&);
MatrixXd binaryActivationDerivative(MatrixXd& activations) {
	ArrayXXd minusActivationsExp = (-(activations.array())).exp();
	return (minusActivationsExp / (minusActivationsExp + 1).square()).matrix();
}

/** Derivative activation function of a unit continuous layer */
MatrixXd continuousActivationDerivative(MatrixXd&);
MatrixXd continuousActivationDerivative(MatrixXd& activations) {
	auto activationsArray = activations.array();
	return (activationsArray.abs() < 10e-3).select(
		1 / 12 - activationsArray.square() / 240,
		1 / activationsArray.square() - (1 / (activationsArray.exp() + (-activationsArray).exp() - 2))
	).matrix();
}


/** Computes the gradient of the DeepBeliefNet related with the provided data into gradientRBMs (vector<RBM>)
 * This gradient can be used for backpropagation or other puroposes.
 */
void DeepBeliefNet::getGradient(const MatrixXd& data, vector<RBM>& gradientRBMs, double* f) {
	if (!unrolled) {
		throw std::runtime_error("You must unroll the DBN before calling getGradient.");
	}
	
	// compute the gradient and put it in *df
	// Pass up and compute activations & activities
	size_t L = myRBMs.size();
	vector<MatrixXd> activations(L + 1), activities(L + 1), deltas(L + 1);
	activities[0] = data; // activities of layer 0 is the data... not sure it makes sense or will be convenient later on...
	
	// Compute activations and activities for all layers
	for (size_t l = 0; l < L; ++l) {
		RBM& layer = myRBMs[l];
		activations[l + 1] = layer.forwardsDataToActivations(activities[l]);
		activities[l + 1] = layer.forwardsActivationsToActivities(activations[l + 1]);
	}
	const MatrixXd& reconstructions = activities[L];
	
	// Compute error if a pointer was supplied
	if (f != nullptr) {
		*f = errorSum(data, reconstructions); 
	}

	// Error gradient on last layer
	Layer::Type outputType = myLayers[L].getType();
	if (outputType == Layer::binary) {
		deltas[L] = (reconstructions.array() - data.array()) *  binaryActivationDerivative(activations[L]).array();
	}
	else if (outputType == Layer::gaussian) {
		deltas[L] = (reconstructions.array() - data.array());
	}
	else { // implicit Layer::continuous
		deltas[L] = (reconstructions.array() - data.array()) *  continuousActivationDerivative(activations[L]).array();
	}
	
	// Now back-propagate this gradient to the previous layers
	for (size_t l = L - 1; l > 0; --l) {
		const RBM& currentRBM = myRBMs[l];
		outputType = currentRBM.getInput().getType();
		const MatrixXdMap& currentW = currentRBM.getW();
		
		if (outputType == Layer::binary) {
			deltas[l] = binaryActivationDerivative(activations[l]).array() * (currentW.transpose() * deltas[l + 1]).array();
		}
		else if (outputType == Layer::gaussian) {
			deltas[l] = (currentW.transpose() * deltas[l + 1]);
		}
		else { // implicit Layer::continuous
			deltas[l] = continuousActivationDerivative(activations[l]).array() * (currentW.transpose() * deltas[l + 1]).array();
		}
	}

	// Compute the weight gradients
	for (size_t l = L ; l-- > 0 ; ) { // loop l = L-1 .. 0, see http://stackoverflow.com/questions/665745/whats-the-best-way-to-do-a-reverse-for-loop-with-an-unsigned-index
		gradientRBMs[l].setC(deltas[l + 1].rowwise().sum());
		gradientRBMs[l].setW(deltas[l + 1] * activities[l].transpose());
	}
}

DeepBeliefNet& DeepBeliefNet::train(const MatrixXd& data, const TrainParameters& params, TrainProgress& aProgressFunctor, const ContinueFunction& aContinueFunction) {
	/* Running eigen threaded? */
	Eigen::setNbThreads(params.nbThreads);
	
	if (!unrolled) throw std::runtime_error("Only unrolled networks can be trained");
	
	DeepBeliefNet trainingDBN = this->clone(); // work on a copy
	
	Eigen_size_type batchSizeEigen = boost::numeric_cast<Eigen_size_type>(params.batchSize);
	MatrixXd batch = MatrixXd::Zero(myLayers[0].getSize(), batchSizeEigen);
	Random batchRand("uniform_int", boost::numeric_cast<size_t>(data.cols()));

	// Input and output pointers for/from cgmin:
	OptimParameters OptimParams(trainingDBN, batch);
	std::unique_ptr<unsigned int> fncount(new unsigned int {0}), grcount(new unsigned int {0});
	std::unique_ptr<int> fail(new int {0});
	std::unique_ptr<double> Fmin(new double {0.0});
	shared_array_ptr<double> trainingData = trainingDBN.getData();
	shared_array_ptr<double> X = trainingDBN.getData().clone(); // Working copy of weights
	
	// Store error in a vector
	vector<double> errors;
	errors.reserve(params.maxIters);
	
	applyDataIfNeeded(trainingData); // Apply the best weights to the DBN
	
	// Get random batch
	aProgressFunctor.setBatchSize(params.batchSize);
	aProgressFunctor.setMaxIters(params.maxIters);
	batchRand.setBatch(data, batch);
		
	//bool continueTraining = true;
	unsigned int stopCounter = 0;
	unsigned int iter = 0;

	// Report progress
	aProgressFunctor(*this, batch, iter);
	
	cout << "Training until stopCounter reaches " << aContinueFunction.limit << endl;
	while (stopCounter < aContinueFunction.limit && iter < params.maxIters) {
		++iter;
        Rcpp::checkUserInterrupt();
		//cout << "Backprop iteration " << iter << " / " << params.maxIters << " (batchsize " << params.batchSize << ")" << endl;

		cgmin(
			trainingDBN.getData().size(), // n, nb arguments
			trainingDBN.getData().data(), // Bvec, vector of working & start parameters, length n
			X.data(), // X, vector of temporary parameters, length n
			Fmin.get(), // Fmin, Minimum of the function
			my_f, // fn, Error function
			my_df, // gr, Gradient function
			fail.get(), // fail, Output
			params.myCgMinParams, // Additional minimization parameters
			OptimParams, // ex, parameters probably passed to optimfn and optimgr
			fncount.get(), // fncount, Output
			grcount.get() // grcount, Output
		);
		
		// Store error
		errors.push_back(*Fmin);

		// Report progress
		aProgressFunctor(*this, batch, iter);
		
		// Do we continue?
		if (iter >= params.minIters && iter % aContinueFunction.frequency == 0) {
			aContinueFunction(errors, iter, params.batchSize, params.maxIters) ? stopCounter = 0 : ++stopCounter;
		}
		
		if (stopCounter < aContinueFunction.limit && iter < params.maxIters) {
			// Get random batch
			batchRand.setBatch(data, batch);
		}
	}

	std::cout << "Final error: " << errorSum(batch) / double(params.batchSize) << std::endl;
	finetuned = true;
	
	return *this;
}
