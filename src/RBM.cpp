#undef NDEBUG
#include <Eigen/Dense>
#include <Rcpp.h>

#include <vector>
#include <fstream> // std::ofstream
#include <cassert> // assert
using std::vector;

#include "Random.h"
#include "RBM.h"
#include "utils.h" // tanhInPlace
#include "Progress.h"

using Eigen::ArrayXXd;
using Eigen::MatrixXd;

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

RBM& RBM::pretrain(const MatrixXd& data, const PretrainParameters& params, PretrainProgress& aProgressFunctor) {
	// assert(1 == 2); // check whether we run in debug mode
	/* Running eigen threaded? */
	Eigen::setNbThreads(params.nbThreads);
	
	/* data size ? */
	size_t samplesize = data.cols();
	
	/* get pretraining parameters from params */
	const size_t maxiters = params.maxiters;
	const size_t batchsize = params.batchsize;
	const double batchsizeAsDouble = static_cast<double>(batchsize);
	const PretrainParameters::PenalizationType penalization = params.penalization;
	const vector<double> momentums = params.getValidMomentums();
	const ArrayX1d lambdaBvec = ArrayX1d::Constant(b.size(), params.lambdaB);
	const ArrayX1d lambdaCvec = ArrayX1d::Constant(c.size(), params.lambdaC);
	const ArrayXXd lambdaWarr = ArrayXXd::Constant(W.rows(), W.cols(), params.lambdaW);
	const double epsilonB = params.epsilonB;
	const double epsilonC = params.epsilonC;
	const double epsilonW = params.epsilonW;
	const bool trainB = params.trainB;
	const bool trainC = params.trainC;
	
	// Print some output to let the user know we're doing something
	Rcpp::Rcout << "Pre-training " << input.getSize() << "-" << input.getTypeAsString() << " x " << output.getSize() << "-" << output.getTypeAsString() << " RBM "
	      << "with " << maxiters << " x " << batchsize << " out of " << samplesize << std::endl
	      << "learning rate (b, W, c) = " << epsilonB << ", " << epsilonW << ", " << epsilonC << "; "
	      << "penalization (b, W, c) = " << PretrainParameters::PenalizationTypeToString(penalization) 
	      << " * (" << params.lambdaB << ", " << params.lambdaW << ", " << params.lambdaC << "); "
	      << "updating (b, c) = (" << trainB << ", " << trainC << ")" << std::endl;
	
	// Pre allocate variables that will be used multiple times
	MatrixXd batch = MatrixXd::Zero(input.getSize(), batchsize);
	ArrayXXd Winc = MatrixXd::Zero(W.rows(), W.cols());
	ArrayX1d bInc = ArrayX1d::Zero(b.size());
	ArrayX1d cInc = ArrayX1d::Zero(c.size());
	ArrayXXd SampleAlpha = ArrayXXd::Zero(output.getSize(), batchsize); // sample variable for h
	MatrixXd Alpha = ArrayXXd::Zero(output.getSize(), batchsize); // h.sampled
	MatrixXd Beta = ArrayXXd::Zero(input.getSize(), batchsize); // P.f.given.h
	MatrixXd Alpha2 = ArrayXXd::Zero(output.getSize(), batchsize); // P.h.given.f
	ArrayX1d deltaB, deltaC, penalizedDeltaB, penalizedDeltaC;
	ArrayXXd deltaW, penalizedDeltaW;
	
	// Prepare the random number generator
	Random sampleRand(output.getType());
	Random batchRand("uniform_int", samplesize);
	
	// Start with a null batch progress
	aProgressFunctor(*this, 0);
	
	// This is the RcppProgress that will display a progress bar / handle user interrupts
	//Progress p(maxiters, true);

	// Loop over batches
	for (unsigned int i = 1; i <= maxiters; ++i) {
        //if (Progress::check_abort()) throw std::runtime_error("Aborted by user");
        Rcpp::checkUserInterrupt();

		// Modify the batch in place
		batchRand.setBatch(data, batch);
		
		// Set Alpha (in-place modification)
		forwardsDataToActivationsInPlace(batch, Alpha);
		sampleRand.setRandom(SampleAlpha);
		forwardsActivationsToActivitiesSampleInPlace(Alpha, SampleAlpha); 
		
		// Set Beta (in-place modification)
		backwardsHiddenToActivitiesInPlace(Alpha, Beta);
		
		// Set Alpha2 (in-place modification)
		forwardsDataToActivitiesInPlace(Beta, Alpha2);

		// Compute deltas
		if (trainB) deltaB = (batch.array() - Beta.array()).rowwise().sum();
		if (trainC) deltaC = (Alpha.array() - Alpha2.array()).rowwise().sum();
		deltaW = (Alpha * batch.transpose()).array() - (Alpha2 * Beta.transpose()).array();
		
		if (penalization == PretrainParameters::PenalizationType::l1) {
			if (trainB) deltaB = ((deltaB / batchsizeAsDouble) - (b > 0).select(lambdaBvec, (-1) * lambdaBvec)).eval();
			if (trainC) deltaC = ((deltaC / batchsizeAsDouble) - (c > 0).select(lambdaCvec, (-1) * lambdaCvec)).eval();
			deltaW = ((deltaW / batchsizeAsDouble) - (W.array() > 0).select(lambdaWarr, (-1) * lambdaWarr)).eval();
		}
		else if (penalization == PretrainParameters::PenalizationType::l2) {
			if (trainB) deltaB = ((deltaB / batchsizeAsDouble) - (lambdaBvec * b)).eval();
			if (trainC) deltaC = ((deltaC / batchsizeAsDouble) - (lambdaCvec * c)).eval();
			deltaW = ((deltaW / batchsizeAsDouble) - (lambdaWarr * W.array())).eval();
		}
		
		// Compute weights increments
		if (trainB) bInc = epsilonB * deltaB;
		if (trainC) cInc = epsilonC * deltaC;
		Winc = epsilonW * deltaW;
	
		//Update the RBM object
		if (trainB) b += tanhInPlace(bInc);
		//b = Beta.rowwise().sum();
		if (trainC) c += tanhInPlace(cInc);
		//c = Alpha.rowwise().sum();
		W.array() += tanhInPlace(Winc);
		
		// Report progress
		aProgressFunctor(*this, i);
		
		// And increment counter
		//p.increment();
	}
	this->pretrained = true;
	return *this;
}


ArrayX1d RBM::error(const Eigen::MatrixXd& data, const Eigen::MatrixXd& reconstructions) const {
	return (reconstructions.array() - data.array()).square().colwise().sum();
}

double RBM::errorSum(const Eigen::MatrixXd& data, const Eigen::MatrixXd& reconstructions) const {
	return error(data, reconstructions).sum();
}

ArrayX1d RBM::error(const MatrixXd& data) const {
	//MatrixXd reconstructions = reconstruct(data);
	return error(data, reconstruct(data));
}

double RBM::errorSum(const MatrixXd& data) const {
	return error(data).sum();
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