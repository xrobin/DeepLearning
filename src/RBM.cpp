#undef NDEBUG
#include <Rcpp.h>

#include <Eigen/Dense>
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
#include <boost/numeric/conversion/cast.hpp>

#include <vector>
#include <fstream> // std::ofstream
#include <cassert> // assert
using std::vector;

#include <DeepLearning/Progress.h>
#include <DeepLearning/RBM.h>
#include <DeepLearning/utils.h> // tanhInPlace
#include "Random.h"


namespace DeepLearning {
	RBM RBM::clone() const { // return a deep copy of the object - but the shared_array_ptr is cloned only between offset and over totalSize(), effectively only cloning the weights of the RBM
		RBM newRBM(*this);
		newRBM.myData = newRBM.myData.clone();
		return newRBM;
	}
	
	RBM RBM::reverse() const { // returns a reversed clone of the RBM
		RBM newRBM(this->output, this->input);
		newRBM.setW(this->getW().transpose());
		newRBM.setB(this->getC());
		newRBM.setC(this->getB());
		return newRBM;
	}
	
	offsets RBM::computeOffsets(const Layer& aInput, const Layer& aOutput) {
		return std::make_tuple(0, aInput.getSize(), aInput.getSize() + aInput.getSize() * aOutput.getSize(), aInput.getSize() + aInput.getSize() * aOutput.getSize() + aOutput.getSize());
	}
	
	void RBM::forwardsDataToActivationsInPlace(const MatrixXd& data, MatrixXd& activations) const {
		activations = ((W * data).array().colwise() + c);
	}
	
	void RBM::forwardsDataToActivationsInPlace(MatrixXd& data) const {
		data = ((W * data).array().colwise() + c).eval();
	}
	
	MatrixXd RBM::forwardsDataToActivations(MatrixXd data) const {
		forwardsDataToActivationsInPlace(data);
		return data;
	}
	
	void RBM::forwardsActivationsToActivitiesInPlace(MatrixXd& act) const {
		genericActivationsToActivitiesInPlace(act, output.getType());
	}
	
	
	MatrixXd RBM::forwardsActivationsToActivities(MatrixXd activations) const {
		forwardsActivationsToActivitiesInPlace(activations);
		return activations.matrix();
	}
	
	void RBM::forwardsDataToActivitiesInPlace(const MatrixXd& data, MatrixXd& act) const {
		forwardsDataToActivationsInPlace(data, act);
		forwardsActivationsToActivitiesInPlace(act);
	}
	
	void RBM::forwardsDataToActivitiesInPlace(MatrixXd& data) const {
		forwardsDataToActivationsInPlace(data);
		forwardsActivationsToActivitiesInPlace(data);
	}
	
	MatrixXd RBM::forwardsDataToActivities(MatrixXd data) const {
		forwardsDataToActivationsInPlace(data);
		forwardsActivationsToActivitiesInPlace(data);
		return data;	
	}
	
	/* Backward pass functions */
	void RBM::backwardsHiddenToActivationsInPlace(const MatrixXd& hidden, MatrixXd& activations) const {
		activations = ((W.transpose() * hidden).array().colwise() + b);
	}
	
	void RBM::backwardsHiddenToActivationsInPlace(MatrixXd& hidden) const {
		hidden = ((W.transpose() * hidden).array().colwise() + b).eval();
	}
	
	MatrixXd RBM::backwardsHiddenToActivations(MatrixXd hidden) const {
		backwardsHiddenToActivationsInPlace(hidden);
		return hidden;
	}
	
	//void RBM::backwardsActivationsToActivitiesInPlace(ArrayXXd& act) const {
	//	genericActivationsToActivitiesInPlace(act, input.getType());
	//}
	
	void RBM::backwardsActivationsToActivitiesInPlace(MatrixXd& act) const {
		genericActivationsToActivitiesInPlace(act, input.getType());
	}
	
	MatrixXd RBM::backwardsActivationsToActivities(MatrixXd activations) const {
		backwardsActivationsToActivitiesInPlace(activations);
		return activations;
	}
	
	void RBM::backwardsHiddenToActivitiesInPlace(const MatrixXd& hidden, MatrixXd& act) const {
		backwardsHiddenToActivationsInPlace(hidden, act);
		backwardsActivationsToActivitiesInPlace(act);
	}
	
	void RBM::backwardsHiddenToActivitiesInPlace(MatrixXd& hidden) const {
		backwardsHiddenToActivationsInPlace(hidden);
		backwardsActivationsToActivitiesInPlace(hidden);
	}
	
	MatrixXd RBM::backwardsHiddenToActivities(MatrixXd hidden) const {
		backwardsHiddenToActivationsInPlace(hidden);
		backwardsActivationsToActivitiesInPlace(hidden);
		return hidden;
	}
	
	void RBM::genericActivationsToActivitiesInPlace(MatrixXd& act, const Layer::Type& target) const {
		if (target == Layer::binary) {
			act.array() = 1 / ((act.array() * (-1)).exp() + 1);
		}
		else if (target == Layer::gaussian) {
			//do nothing
		}
		else {
			act.array() = (act.array().abs() < 1e-5).select(0.5, (act.array().exp() * (1 - 1 / act.array()) + 1 / act.array()) / (act.array().exp() - 1));
		}
	}
	
	// tests if SampleAlpha < Alpha and assigns 1 or 0 into Alpha depending on the result (1 = TRUE, 0 = FALSE)
	// Modifies Alpha in place
	void biggerThanInPlace(MatrixXd&, const ArrayXXd&);
	void biggerThanInPlace(MatrixXd& Alpha, const ArrayXXd& SampleAlpha) {
		for (int i = 0; i < Alpha.size(); i++) {
			*(Alpha.data() + i) = *(SampleAlpha.data() + i) < *(Alpha.data() + i) ? 1 : 0;
		}
	}
	
	void RBM::forwardsActivationsToActivitiesSampleInPlace(MatrixXd& act, ArrayXXd sample) const {
		if (output.getType() == Layer::Type::binary) {
			act = (1 / ((act.array() * (-1)).exp() + 1)).eval();
			biggerThanInPlace(act, sample);
		}
		else if (output.getType() == Layer::Type::gaussian) {
			act = (act.array() + sample).eval();
		}
		else {
			act = ((act.array().abs() < 1e-6).select(sample, 1 / act.array() * (sample * (act.array().exp() - 1) + 1).log())).eval();
		}
	}
	
	RBM& RBM::pretrain(const MatrixXd& data, const PretrainParameters& params, PretrainProgress& aProgressFunctor, const ContinueFunction& aContinueFunction) {
		// assert(1 == 2); // check whether we run in debug mode
		/* Running eigen threaded? */
		Eigen::setNbThreads(params.nbThreads);
		
		/* data size ? */
		size_t samplesize = boost::numeric_cast<size_t>(data.cols());
		
		/* get pretraining parameters from params */
		const unsigned int maxIters = params.maxIters;
		const size_t batchSize = params.batchSize;
		const Eigen_size_type batchSizeAsEigen = boost::numeric_cast<Eigen_size_type>(batchSize);
		const double batchSizeAsDouble = boost::numeric_cast<double>(batchSize);
		const PretrainParameters::PenalizationType penalization = params.penalization;
		const vector<double> momentums = params.getValidMomentums();
		const ArrayX1d lambdaBvec = ArrayX1d::Constant(b.size(), params.lambdaB);
		const ArrayX1d lambdaCvec = ArrayX1d::Constant(c.size(), params.lambdaC);
		const ArrayXXd lambdaWarr = ArrayXXd::Constant(W.rows(), W.cols(), params.lambdaW);
		// Dummy arrays of 0
		const ArrayX1d zeroBvec = ArrayX1d::Constant(b.size(), 0);
		const ArrayX1d zeroCvec = ArrayX1d::Constant(c.size(), 0);
		const ArrayXXd zeroWarr = ArrayXXd::Constant(W.rows(), W.cols(), 0);
		const double epsilonB = params.epsilonB;
		const double epsilonC = params.epsilonC;
		const double epsilonW = params.epsilonW;
		const bool trainB = params.trainB;
		const bool trainC = params.trainC;
		
		// Print some output to let the user know we're doing something
		Rcpp::Rcout << "Pre-training " << input.getSize() << "-" << input.getTypeAsString() << " x " << output.getSize() << "-" << output.getTypeAsString() << " RBM "
		      << "with " << maxIters << " x " << batchSize << " out of " << samplesize << std::endl
		      << "learning rate (b, W, c) = " << epsilonB << ", " << epsilonW << ", " << epsilonC << "; "
		      << "penalization (b, W, c) = " << PretrainParameters::PenalizationTypeToString(penalization) 
		      << " * (" << params.lambdaB << ", " << params.lambdaW << ", " << params.lambdaC << "); "
		      << "updating (b, c) = (" << trainB << ", " << trainC << ")" << std::endl;
		
		// Pre allocate variables that will be used multiple times
		MatrixXd batch = MatrixXd::Zero(input.getSize(), batchSizeAsEigen);
		ArrayXXd Winc = MatrixXd::Zero(W.rows(), W.cols());
		ArrayX1d bInc = ArrayX1d::Zero(b.size());
		ArrayX1d cInc = ArrayX1d::Zero(c.size());
		ArrayXXd SampleAlpha = ArrayXXd::Zero(output.getSize(), batchSizeAsEigen); // sample variable for h
		MatrixXd Alpha = ArrayXXd::Zero(output.getSize(), batchSizeAsEigen); // h.sampled
		MatrixXd Beta = ArrayXXd::Zero(input.getSize(), batchSizeAsEigen); // P.f.given.h
		MatrixXd Alpha2 = ArrayXXd::Zero(output.getSize(), batchSizeAsEigen); // P.h.given.f
		ArrayX1d deltaB, deltaC, penalizedDeltaB, penalizedDeltaC;
		ArrayXXd deltaW, penalizedDeltaW;
		
		// Prepare the random number generator
		Random sampleRand(output.getType());
		Random batchRand("uniform_int", samplesize);
	
		// Store error in a vector
		vector<double> errors;
		errors.reserve(maxIters);
		
		// This is the RcppProgress that will display a progress bar / handle user interrupts
		//Progress p(maxIters, true);
	
		// Loop over batches
		unsigned int stopCounter = 0;
		unsigned int i = 0;
		
		// Modify the batch in place
		batchRand.setBatch(data, batch);	
		
		// Start with a null batch progress
		aProgressFunctor.setBatchSize(batchSize);
		aProgressFunctor.setMaxIters(maxIters);
		aProgressFunctor(*this, batch, i);
		
		Rcpp::Rcout << "Pre-training until stopCounter reaches " << aContinueFunction.limit << std::endl;
		while (stopCounter < aContinueFunction.limit && i < maxIters) {
			++i;
			//Rcpp::Rcout << "Pretrain iteration " << i << " / " << params.maxIters << " (batchsize " << params.batchSize << ")" << std::endl;
	        Rcpp::checkUserInterrupt();
			
			// Set Alpha (in-place modification)
			forwardsDataToActivationsInPlace(batch, Alpha);
			sampleRand.setRandom(SampleAlpha);
			forwardsActivationsToActivitiesSampleInPlace(Alpha, SampleAlpha); 
			
			// Set Beta (in-place modification)
			backwardsHiddenToActivitiesInPlace(Alpha, Beta);
			
			// Set Alpha2 (in-place modification)
			forwardsDataToActivitiesInPlace(Beta, Alpha2);
	
			// Compute deltas
			if (trainB) deltaB = ((batch.array() - Beta.array()).rowwise().sum()) / batchSizeAsDouble;
			if (trainC) deltaC = ((Alpha.array() - Alpha2.array()).rowwise().sum()) / batchSizeAsDouble;
			deltaW = ((Alpha * batch.transpose()).array() - (Alpha2 * Beta.transpose()).array()) / batchSizeAsDouble;
			
			if (trainB) bInc = epsilonB * deltaB;
			if (trainC) cInc = epsilonC * deltaC;
			Winc.array() = epsilonW * deltaW.array();
			
			if (penalization == PretrainParameters::PenalizationType::l1) {			
				// According to Tsuruoka, Tsujii and Ananiadou, 2009 "Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty"
				// in Proceedings of the 47th Annual Meeting of the ACL and the 4th IJCNLP of the AFNLP, pages 477–485
				// Equation page 479
				if (trainB) {
					const ArrayX1d tempB = b + bInc;
					b = (tempB != 0).select((tempB > 0).select(
					zeroBvec.max(tempB - epsilonB * lambdaBvec), // b_i > 0
					zeroBvec.min(tempB + epsilonB * lambdaBvec)), // b_i < 0
					0); // b_i == 0
				}
				
				if (trainC) {
					const ArrayX1d tempC = c + cInc;
					c = (tempC != 0).select((tempC > 0).select(
					zeroCvec.max(tempC - epsilonC * lambdaCvec), // c_i > 0
					zeroCvec.min(tempC + epsilonC * lambdaCvec)), // c_i < 0
					0); // c_i == 0
				}
				
				const ArrayXXd tempW = W.array() + Winc.array();
				W.array()= (tempW != 0).select((tempW > 0).select(
					zeroWarr.max(tempW - epsilonW * lambdaWarr), // w_i > 0
					zeroWarr.min(tempW + epsilonW * lambdaWarr)), // w_i < 0
					0); // w_i == 0
			}
			else if (penalization == PretrainParameters::PenalizationType::l2) {
				if (trainB) deltaB -= lambdaBvec * b;
				if (trainC) deltaC -= lambdaCvec * c;
				deltaW.array() -= lambdaWarr * W.array();
			}
			
			if (penalization != PretrainParameters::PenalizationType::l1) {
				//Update the RBM object
				if (trainB) b += tanhInPlace(bInc);
				//b = Beta.rowwise().sum();
				if (trainC) c += tanhInPlace(cInc);
				//c = Alpha.rowwise().sum();
				W.array() += tanhInPlace(Winc);
			}
			
			// Store error
			errors.push_back(errorSum(deltaB, deltaC, deltaW));
			
			// Report progress
			aProgressFunctor(*this, batch, i);
			
			// Do we continue?
			if (i >= params.minIters && i % aContinueFunction.frequency == 0) {
				aContinueFunction(errors, i, params.batchSize, maxIters) ? stopCounter = 0 : ++stopCounter;
			}
			
			if (stopCounter < aContinueFunction.limit && i < maxIters) {
				// Modify the batch in place
				batchRand.setBatch(data, batch);	
			}
		}
		this->pretrained = true;
		return *this;
	}
	
	
	double RBM::errorSum(const ArrayX1d& deltaB, const ArrayX1d& deltaC, const ArrayXXd& deltaW) const {
		double error = 0;
		error += deltaB.square().sum();
		error += deltaC.square().sum();
		error += deltaW.square().sum();
		error /= (nInput() + nOutput() + nWeights());
		return sqrt(error);
	}
	
	ArrayX1d RBM::energy(const MatrixXd& data) const {
		ArrayXXd predictions = predict(data);
		return - (data.array().colwise() + b).colwise().sum() - (predictions.colwise() + c).colwise().sum() - ((W * data).array() * predictions).colwise().sum();
	}
	
	double RBM::energySum(const MatrixXd& data) const {
		return energy(data).sum();
	}
	
	/* Setting values */
	RBM& RBM::setB(ArrayX1d aNewB) {
		assert(b.rows() == aNewB.rows() && b.cols() == aNewB.cols());
		b = aNewB;
		return *this;
	}
	RBM& RBM::setC(ArrayX1d aNewC) {
		assert(c.rows() == aNewC.rows() && c.cols() == aNewC.cols());
		c = aNewC;
		return *this;
	}
	RBM& RBM::setW(Eigen::MatrixXd aNewW) {
		assert(W.rows() == aNewW.rows() && W.cols() == aNewW.cols());
		W = aNewW;
		return *this;
	}
	RBM& RBM::setB(ArrayX1dMap aNewB) {
		assert(b.rows() == aNewB.rows() && b.cols() == aNewB.cols());
		b = aNewB;
		return *this;
	}
	RBM& RBM::setC(ArrayX1dMap aNewC) {
		assert(c.rows() == aNewC.rows() && c.cols() == aNewC.cols());
		c = aNewC;
		return *this;
	}
	RBM& RBM::setW(MatrixXdMap aNewW) {
		assert(W.rows() == aNewW.rows() && W.cols() == aNewW.cols());
		W = aNewW;
		return *this;
	}
}
