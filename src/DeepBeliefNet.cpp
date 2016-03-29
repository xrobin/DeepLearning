#include <Rcpp.h>

#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::ArrayXXd;
#include <boost/range/adaptor/reversed.hpp> // boost::adaptors::reverse

#include <vector>
using std::vector;

#include <DeepLearning.h>
using namespace DeepLearning;


namespace DeepLearning {
/** Creates a deep copy of the DeepBeliefNet and returns it.
 * for efficiency reasons, copying a DBN does **NOT** copy the weights. Those are stored in a copy-counted pointer
 * and the normal copy semantics makes only a superficial copy. This means that modifying weights in the copy
 * of the DeepBeliefNet also modifies those in the originall one. Use clone() for a deep copy.
 * Should be called as:
 * DeepBeliefNet cloned = original.clone();
 * The cloned version is identical to the original one, except that a deep copy of the weights is performed.
 * Note: this can be relatively slow if the function is called repeatedly on a large network.
 */
DeepBeliefNet DeepBeliefNet::clone() const { // return a deep copy of the object - but without the offset
	DeepBeliefNet newDBN(*this);
	newDBN.myData = newDBN.myData.clone();
	newDBN.constructRBMs();
	return newDBN;
}

/** Compute the size needed to hold all weights from the layers */
size_t DeepBeliefNet::computeDataSize(const vector<Layer>& layers) {
	size_t dataSize = layers[0].getSize();
	// Get data size
	for (size_t i = 0; i < layers.size() - 1; ++i) {
		dataSize += layers[i].getSize() * layers[i+1].getSize() + layers[i+1].getSize();
	}
	return dataSize;
}

/** Constructs the RBMs. Intended to be used just after initialization */
void DeepBeliefNet::constructRBMs() {
	constructRBMs(myRBMs, myLayers, myData);
}

/** Constructs someRBMs and binds them to someData as required by the layout of someLayers.
 * Static method that modifies someRBMs in place. Does not modifies the layer or someData (will take pointers on it though)
 */
void DeepBeliefNet::constructRBMs(vector<RBM>& someRBMs, const vector<Layer>& someLayers, const shared_array_ptr<double>& someData) {
	someRBMs.clear();
	size_t nRBMs = someLayers.size() - 1;
	someRBMs.reserve(nRBMs);

	size_t nextLayerOffset	= 0;
	for (size_t i = 0; i < nRBMs; ++i) {
		someRBMs.push_back(RBM(someLayers[i], someLayers[i+1], someData + nextLayerOffset));
		nextLayerOffset += someRBMs[i].getRelativeOffsetC(); // B of the next layer is C of this one
	}
}


DeepBeliefNet DeepBeliefNet::unroll() const {
	/* Unroll layers */
	vector<Layer> newLayers(myLayers);
	newLayers.insert(newLayers.end(), myLayers.rbegin() + 1, myLayers.rend());

	/* Unrolled data size */
	size_t newDataSize = computeDataSize(newLayers);
	
	/* Create the new DeepBeliefNet object */
	shared_array_ptr<double> newData(new double[newDataSize], newDataSize, true); // true: let this DBN manage the data
	                                                   //, because the pointer will be lost as soon as we return anyway.
	DeepBeliefNet newDBN(newLayers, newData);
	
	// Copy the first B into the new vector
	newDBN.myRBMs[0].setB(this->myRBMs[0].getB());

	size_t newRBMLast = newDBN.myRBMs.size() - 1;
	for (size_t i = 0; i < myRBMs.size(); i++) {
		// For clarity, take references to old, new and reversed RBM
		const RBM& oldRBM = this->myRBMs[i];
		RBM& newRBM = newDBN.myRBMs[i];
		RBM& reversedRBM = newDBN.myRBMs[newRBMLast - i];
		
		// Set new weights
		newRBM.setW(oldRBM.getW());
		reversedRBM.setW(oldRBM.getW().transpose());
		
		// Set new Cs
		newRBM.setC(oldRBM.getC());
		reversedRBM.setC(oldRBM.getB());
	}

	newDBN.pretrained = pretrained; // Keep the pretrained status
	newDBN.unrolled = true;
	
	return newDBN;
}

DeepBeliefNet DeepBeliefNet::reverse() const { // returns a reversed clone of the RBM
	DeepBeliefNet newDBN = this->clone();
	
	/* Reverse layers */
	std::reverse(newDBN.myLayers.begin(), newDBN.myLayers.end());
	
	/* Construct RBMs */
	newDBN.constructRBMs();
	
	/* Assign */
	for (size_t i = 0; i < this->myRBMs.size(); i++) {
		size_t j = this->myRBMs.size() - 1 - i;
		newDBN.myRBMs[i].setB(this->myRBMs[j].getC());
		newDBN.myRBMs[i].setC(this->myRBMs[j].getB());
		newDBN.myRBMs[i].setW(this->myRBMs[j].getW().transpose());
	}
	
	return newDBN;
}

/*DeepBeliefNet& DeepBeliefNet::pretrain(const MatrixXdMap& data, const PretrainParameters& params) {
	MatrixXd tmpdata(data);
	return pretrainModifyingData(tmpdata, params);
}*/

DeepBeliefNet& DeepBeliefNet::pretrain(MatrixXd data, const vector<PretrainParameters>& params, PretrainProgress& aProgressFunctor, ContinueFunction& aContinueFunction, const vector<size_t>& skip) {
	return pretrainModifyingData(data, params, aProgressFunctor, aContinueFunction, skip);
}

DeepBeliefNet& DeepBeliefNet::pretrainModifyingData(MatrixXd& data, const vector<PretrainParameters>& params, PretrainProgress& aProgressFunctor, ContinueFunction& aContinueFunction, const vector<size_t>& skip) {
	// Print some output to let the user know we're doing something
	Rcpp::Rcout << "Pre-training " << myLayers.front().getSize() << " - " << myLayers.back().getSize() << " network with " << nLayers() << " layers" << std::endl;

	if (skip.size() > 0) {
		Rcpp::Rcout << "Ignoring the following layers: ";
		for (size_t layer: skip) {
			Rcpp::Rcout << layer << ", ";
		}
		Rcpp::Rcout << std::endl;
	}

	for (size_t i = 0; i < myRBMs.size(); ++i) {
		if (isIn(skip, i + 1)) {
			Rcpp::Rcout << "Skipping " << myRBMs[i].getInput().getSize() << "-" << myRBMs[i].getInput().getTypeAsString() << " x "
			            << myRBMs[i].getOutput().getSize() << "-" << myRBMs[i].getOutput().getTypeAsString() << " RBM " << std::endl;
		}
		else {
			aProgressFunctor.setBatchSize(params[i].batchSize);
			aProgressFunctor.setMaxIters(params[i].maxIters);
			aProgressFunctor.setLayer(i + 1);
			aContinueFunction.setLayer(i + 1);
			// Pretrain each layer
			myRBMs[i].pretrain(data, params[i], aProgressFunctor, aContinueFunction);	
		}
		// Pass the data through the layer
		if (i < myRBMs.size() - 1) {
			myRBMs[i].predictInPlace(data);
			aProgressFunctor.propagateData(myRBMs[i]);
		}
	}
	pretrained = true;
	return *this;
}

MatrixXd DeepBeliefNet::predict(MatrixXd data) const { // work on a copy of data
	predictInPlace(data);
	return data;
}

void DeepBeliefNet::predictInPlace(MatrixXd& data) const {
	size_t lastLayerToPredict = unrolled ? myRBMs.size() / 2 : myRBMs.size();
	for (size_t i = 0; i < lastLayerToPredict; ++i) {
		data = myRBMs[i].predict(data);
	}
}

MatrixXd DeepBeliefNet::reverse_predict(MatrixXd hidden) const {
	reverse_predictInPlace(hidden);
	return hidden;
}

void DeepBeliefNet::reverse_predictInPlace(MatrixXd& hidden) const {
	if (unrolled) {
		size_t firstLayerToPredict = myRBMs.size() / 2;
		for (size_t i = firstLayerToPredict; i < myRBMs.size(); ++i) {
			hidden = myRBMs[i].predict(hidden);
		}
	}
	else {
		for (RBM rbm: boost::adaptors::reverse(myRBMs)) { // boost::adaptors::reverse make the loop start from the last element
			hidden = rbm.reverse_predict(hidden);
		}
	}
}

MatrixXd DeepBeliefNet::reconstruct(MatrixXd data) const { // work on a copy of data
	reconstructInPlace(data);
	return data;
}

void DeepBeliefNet::reconstructInPlace(MatrixXd& data) const {
	predictInPlace(data);
	reverse_predictInPlace(data);
}

DeepBeliefNet& DeepBeliefNet::applyData(double* newWeights) {
	size_t oldSize = myData.size();
	shared_array_ptr<double> newPtr(newWeights, oldSize, false); // false: the pointer is managed by the client
	applyData(newPtr); // calls the method for a shared_array_ptr. That one will call constructRBM().
	return *this;
}

DeepBeliefNet& DeepBeliefNet::applyData(shared_array_ptr<double>& newWeights) {
	myData = newWeights;
	constructRBMs();
	return *this;
}
	
DeepBeliefNet& DeepBeliefNet::applyDataIfNeeded(double* newWeights) {
	if (getData().data() != newWeights) {
		applyData(newWeights);
	}
	return *this;
}
	
DeepBeliefNet& DeepBeliefNet::applyDataIfNeeded(shared_array_ptr<double>& newWeights) {
	if (getData() != newWeights) {
		applyData(newWeights);
	}
	return *this;
}

ArrayX1d DeepBeliefNet::error(const MatrixXd& data) const {
	MatrixXd reconstructions = reconstruct(data);
	return error(data, reconstructions);
}

ArrayX1d DeepBeliefNet::error(const MatrixXd& data, const MatrixXd& reconstructions) const {
	return (reconstructions.array() - data.array()).square().colwise().mean().sqrt();
}

double DeepBeliefNet::errorSum(const MatrixXd& data) const {
	return error(data).sum();
}

double DeepBeliefNet::errorSum(const MatrixXd& data, const MatrixXd& reconstructions) const {
	return error(data, reconstructions).sum();
}

ArrayX1d DeepBeliefNet::energy(MatrixXd data) const {
	ArrayX1d theEnergy = myRBMs[0].energy(data);
	for (size_t layer = 1; layer < myRBMs.size(); ++layer) {
		data = myRBMs[layer - 1].predict(data);
		theEnergy += myRBMs[layer].predict(data).array();
	}
	return theEnergy;
}

double DeepBeliefNet::energySum(const MatrixXd& data) const {
	return energy(data).sum();
}

MatrixXd DeepBeliefNet::sample(MatrixXd data) const { // work on a copy of data
	sampleInPlace(data);
	return data;
}

void DeepBeliefNet::sampleInPlace(MatrixXd& data) const {
	size_t lastLayerToPredict = myRBMs.size();
	for (size_t i = 0; i < lastLayerToPredict; ++i) {
		data = myRBMs[i].sample(data);
	}
}


}
