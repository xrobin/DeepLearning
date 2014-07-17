#pragma once

#include <Eigen/Dense>

#include <vector>

#include "RBM.h"
#include "Layer.h"
#include "shared_array_ptr.h"
#include "PretrainParameters.h"
#include "ContinueFunction.h"
#include "TrainParameters.h"
#include "Progress.h"

#include "typedefs.h"

/** Class DeepBeliefNet
 * Encodes a Deep Belief Network composed of a pointer to a the data (weights & biases) and a vector of RestrictedBolzmanMachines
 * 
 * Constructors:
 *   - DeepBeliefNet(const std::vector<Layer> &layers)
 *   - DeepBeliefNet(const std::vector<Layer> &layers, std::vector<double>& aData)
 *   - DeepBeliefNet(const DeepBeliefNet& anOtherDeepBeliefNet) // copy constructor
 */

class DeepBeliefNet {
	private:
		std::vector<Layer> myLayers;
		//size_t myDataSize;
		shared_array_ptr<double> myData;
		std::vector<RBM> myRBMs; // the Bolzman Machines
		bool pretrained, unrolled, finetuned;

		static size_t computeDataSize(const std::vector<Layer>&);
		/*size_t computeDataSize() {
			return computeDataSize(myLayers);
		}*/
		void constructRBMs();
		//bool cleanUp = false; // CleanUp if we created *myData ourself - not by default, if it was provided at construction time

	public:
		/** Constructs the RBMs as given by the Layers with the given data. Does not return the RBM but modifies it in place */
		static void constructRBMs(std::vector<RBM>& someRBMs, const std::vector<Layer>& someLayers, const shared_array_ptr<double>& someData);
		/* Consructors */
		explicit DeepBeliefNet(const std::vector<Layer> &layers): myLayers(layers), myData(computeDataSize(layers)), myRBMs(),
		pretrained(false), unrolled(false), finetuned(false) {
			//cleanUp = true;
			constructRBMs();
		}
		DeepBeliefNet(const std::vector<Layer> &layers, std::vector<double>& aData, bool isAlreadyPretrained = false, 
		              bool isAlreadyUnrolled = false, bool isAlreadyFinetuned = false):
			myLayers(layers), myData(aData), myRBMs(), pretrained(isAlreadyPretrained), unrolled(isAlreadyUnrolled),
			finetuned(isAlreadyFinetuned) {
			//cleanUp = false; // aData vector will do it anyway
			constructRBMs();
		}
		DeepBeliefNet(const std::vector<Layer> &layers, shared_array_ptr<double>& aData, bool isAlreadyPretrained = false, 
		              bool isAlreadyUnrolled = false, bool isAlreadyFinetuned = false):
			myLayers(layers), myData(aData), myRBMs(), pretrained(isAlreadyPretrained), unrolled(isAlreadyUnrolled), 
			finetuned(isAlreadyFinetuned) {
			constructRBMs();
		}
		// Copy constructor not needed here!
		//DeepBeliefNet(const DeepBeliefNet& anOtherDeepBeliefNet): myLayers(anOtherDeepBeliefNet.myLayers), 
		//	myDataSize(anOtherDeepBeliefNet.myDataSize), myData(anOtherDeepBeliefNet.myData) {
		//	constructRBMs();
		//}
		/* Accessors */
		size_t nLayers() const {return myLayers.size();}
		size_t nRBMs() const {return myRBMs.size();}
		//size_t nData() const {return myDataSize;}
		std::vector<Layer> getLayers() const {return myLayers;}
		Layer getLayer(size_t aLayer) const {return myLayers[aLayer];}
		std::vector<RBM> getRBMs() const {return myRBMs;}
		RBM getRBM(size_t anRBM) const {return myRBMs[anRBM];}
		shared_array_ptr<double> getData() const {return myData;}
		bool isPretrained() const {return pretrained;}
		bool isUnrolled() const {return unrolled;}
		bool isFinetuned() const {return finetuned;}
		
		/* Setters */
		/** Apply the given data to the DBN. Assumes that the data is of the proper length - it cannot be specified here */
		DeepBeliefNet& applyData(double*);
		DeepBeliefNet& applyData(shared_array_ptr<double>& newData);
		DeepBeliefNet& applyDataIfNeeded(double*);
		DeepBeliefNet& applyDataIfNeeded(shared_array_ptr<double>& newData);
	
		/* Training the net */
		/** pretrain and train the DBN
		 * 
		 * Modify the DBN in place and return a reference to it (so you can chain dbn.pretrain(...).train(...).)
		 * @param someData an Eigen MatrixXd or Map<MatrixXd> representing the data
		 * @param someParameters a PretrainParameters object
		 * 
		 */
		//DeepBeliefNet& pretrain(const MatrixXdMap& someData, const PretrainParameters& someParameters);
		DeepBeliefNet& pretrain(Eigen::MatrixXd someData, const std::vector<PretrainParameters>& someParameters, PretrainProgress& aProgressFunctor = NoOpPretrainProgress::getInstance(), ContinueFunction& aContinueFunction = ContinueFunction::getInstance(), const std::vector<size_t>& skip = std::vector<size_t>());
		DeepBeliefNet& pretrainModifyingData(Eigen::MatrixXd& someData, const std::vector<PretrainParameters>& params, PretrainProgress& aProgressFunctor = NoOpPretrainProgress::getInstance(), ContinueFunction& aContinueFunction = ContinueFunction::getInstance(), const std::vector<size_t>& skip = std::vector<size_t>());
		DeepBeliefNet& train(const Eigen::MatrixXd& someData, const TrainParameters&, TrainProgress& aProgressFunctor = NoOpTrainProgress::getInstance(), const ContinueFunction& aContinueFunction = ContinueFunction::getInstance());
		
		/** Returns the gradient of the DeepBeliefNet related with the provided data in a vector<RBM>
		 * This gradient can be used for backpropagation or other puroposes
		 */
		std::vector<RBM> getGradient(const Eigen::MatrixXd& data, shared_array_ptr<double> df);

		/** Computes the gradient of the DeepBeliefNet related with the provided data into gradientRBMs (vector<RBM>)
		 * This gradient can be used for backpropagation or other puroposes.
		 */
		void getGradient(const Eigen::MatrixXd& data, std::vector<RBM>& gradientRBMs, double* f = nullptr);

		/* Predictions & cie */
		/** Computes the squared error of the reconstruction, per data point, and return it in a vector.
		 *  errorSum computes the sum of error over all data points and returns a single double.
		 *  The behaviour is different on unrolled networks: the hidden layer *is* the reconstruction, whereas on non-unrolled networks reverse_predict is used
		 *  to get the reconstructions. This is done through the reconstruct() function.
		 *  Note that if reconstructions is not supplied, it will be computed with the reconstruct() function.
		 */
		ArrayX1d error(const Eigen::MatrixXd&) const;
		double errorSum(const Eigen::MatrixXd&) const;
		ArrayX1d error(const Eigen::MatrixXd& data, const Eigen::MatrixXd& reconstructions) const;
		double errorSum(const Eigen::MatrixXd& data, const Eigen::MatrixXd& reconstructions) const;
		/** Computes the enery of the network, per data point, and return it in a vector 
		 * energySum computes the sum of error over all data points and returns a single double.
		 * energy() takes a copy of the argument and thus does not modify it
		 */
		ArrayX1d energy(Eigen::MatrixXd) const;
		double energySum(const Eigen::MatrixXd&) const;
		/** Predicts the data points given in the matrix (in column).
		* The versions that takes a matrix makes a copy of it first, and returns this copy. The *InPlace versions take operate on a reference to the object
		* The reverse_* versions take a hidden layer and predicts the visible layer. 
		* For an unrolled network, predict propagates through half of the network, and reverse_predict propagates through the other half.
		*/
		Eigen::MatrixXd predict(Eigen::MatrixXd) const;
		void predictInPlace(Eigen::MatrixXd&) const;
		Eigen::MatrixXd reverse_predict(Eigen::MatrixXd) const;
		void reverse_predictInPlace(Eigen::MatrixXd&) const;
		/** Returns a reconstruction of the input data.
		 * On an unrolled network this is exactly the same as predict() because the hidden layer is the reconstruction by definition.
		 * On networks that haven't been unrolled, it is the result of predict() followed by reverse_predict.
		 */
		Eigen::MatrixXd reconstruct(Eigen::MatrixXd) const;
		void reconstructInPlace(Eigen::MatrixXd&) const;
		/* Architecture */
		//void push_back(const RBM&);
		//void push_back(const Layer&);
		std::vector<Layer> getUnrolledLayers() const;
		DeepBeliefNet reverse() const;
		DeepBeliefNet unroll() const;
		DeepBeliefNet clone() const;
		/* Destructor */
		//~DeepBeliefNet() {if (cleanUp) {delete myData;}}
		// Export to R
};

