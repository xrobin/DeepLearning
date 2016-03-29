#pragma once

#include <Eigen/Dense>
#include "boost/numeric/conversion/cast.hpp"

#include <fstream>
#include <memory>
#include <string>

#include <DeepLearning/Layer.h>
#include <DeepLearning/PretrainParameters.h>
#include <DeepLearning/ContinueFunction.h>
#include <shared_array_ptr.h>
#include <DeepLearning/typedefs.h>
#include <DeepLearning/Progress.h>


namespace DeepLearning {
	/** Class RBM
	 * Encodes a Restricted Bolzman Machine. Contains an input and an output layer.
	 * Does not make much sense (i.e probably not usable) outside the context of a DeepBeliefNet.
	 * 
	 * The constructor needs pointers to b, c and W which are used to build Eigen::Map's.
	 * These pointers should be allocated in a DeepBeliefNet, the RBM class doesn't do it itself.
	 * 
	 * Copy and assignment are forbidden - RBM contains pointer and I don't want to deal with potential invalid pointers floating around.
	 * If any of those is required at some point, just move 'RBM(const RBM&)' or 'RBM& operator=(const RBM&);'  into the public space
	 * and define an implementation (or perhaps delete them and hope the default thing will work - it would probably anyway).
	 */
	class RBM {
		private:
			/* Members */
			const Layer input, output;
			const offsets myOffsets; // quick access to offsets to get b (<0>), W (<1>) and c (<2>). <3> is total size of myData
			shared_array_ptr<double> myData;
			ArrayX1dMap b, c;
			MatrixXdMap W;
			bool pretrained;
	
		public:
			/** Forward pass functions */
			/** Takes a visible layer data as MatrixXd and returns the hidden layer activities as an ArrayXXd. Takes a copy of the object first so it will not modify the input */
			Eigen::MatrixXd forwardsDataToActivities(Eigen::MatrixXd) const;
			/** Same as forwardsDataToActivities(), but instead of returning an ArrayXXd, takes it as second argument and modify it in place.
			 * The data is left unmodified.
			 */
			void forwardsDataToActivitiesInPlace(const Eigen::MatrixXd&, Eigen::MatrixXd&) const;
			/** As forwardsDataToActivitiesInPlace(const MatrixXd, ArrayXXd), but modifies the data itself instead. */
			void forwardsDataToActivitiesInPlace(Eigen::MatrixXd&) const;
			
			/** forwardsDataToActivations is as forwardsDataToActivities, only it goes only to the activations, and does not compute the activities.
			 * Use the forwardsActivationsToActivities if you need the activities later on
			 */
			Eigen::MatrixXd forwardsDataToActivations(Eigen::MatrixXd) const;
			void forwardsDataToActivationsInPlace(const Eigen::MatrixXd&, Eigen::MatrixXd&) const;
			void forwardsDataToActivationsInPlace(Eigen::MatrixXd&) const;
			
			/** Converts activations to activities, either in place or returning an ArrayXXd. 
			 * The version that is not InPlace will make a copy of the object first so it will not modify the input
			*/
			Eigen::MatrixXd forwardsActivationsToActivities(Eigen::MatrixXd) const;
			void forwardsActivationsToActivitiesInPlace(Eigen::MatrixXd&) const;
			//void forwardsActivationsToActivitiesInPlace(Eigen::MatrixXd&) const;
			
	
			/* Backward pass functions */
			Eigen::MatrixXd backwardsHiddenToActivations(Eigen::MatrixXd) const;
			void backwardsHiddenToActivationsInPlace(const Eigen::MatrixXd&, Eigen::MatrixXd&) const;
			void backwardsHiddenToActivationsInPlace(Eigen::MatrixXd&) const;
			
			Eigen::MatrixXd backwardsActivationsToActivities(Eigen::MatrixXd) const;
			void backwardsActivationsToActivitiesInPlace(Eigen::MatrixXd&) const;
			
			Eigen::MatrixXd backwardsHiddenToActivities(Eigen::MatrixXd) const;
			void backwardsHiddenToActivitiesInPlace(const Eigen::MatrixXd&, Eigen::MatrixXd&) const;
			void backwardsHiddenToActivitiesInPlace(Eigen::MatrixXd&) const;
			
			/* generic pass functions */
			void genericActivationsToActivitiesInPlace(Eigen::MatrixXd&, const Layer::Type&) const;
			
			/* Sample pass functions */
			void forwardsActivationsToActivitiesSampleInPlace(Eigen::MatrixXd&, Eigen::ArrayXXd) const;
			
			/* Some statics for the constructors */
			static offsets computeOffsets(const Layer&, const Layer&);
	
			// Pass b, c and W explicitly
			//RBM(Layer aInput,  Layer aOutput, double *ab, double *ac, double *aW): input(aInput), output(aOutput), b(ab, aInput.getSize()),
			//	c(ac, aOutput.getSize()), W(aW, aOutput.getSize(), aInput.getSize()) {}
			
			// Pass no data
			RBM(Layer aInput,  Layer aOutput): input(aInput), output(aOutput), myOffsets(computeOffsets(aInput, aOutput)), myData(new double[std::get<3>(myOffsets)], std::get<3>(myOffsets), true),
				b(myData.data(), nInput()), c(myData.data() + std::get<2>(myOffsets), nOutput()), W(myData.data() + std::get<1>(myOffsets), nOutput(), nInput()), pretrained(false) {}
			
			// Pass b, c and W as a single pointer - the others are computed from aInput and aOutput sizes
			RBM(Layer aInput,  Layer aOutput, double* abcW, bool isAlreadyPretrained = false): input(aInput), output(aOutput), myOffsets(computeOffsets(aInput, aOutput)), myData(abcW, std::get<3>(myOffsets), false),
				b(abcW, nInput()), c(abcW + std::get<2>(myOffsets), nOutput()), W(abcW + std::get<1>(myOffsets), nOutput(), nInput()), pretrained(isAlreadyPretrained) {
	//				std::cout << "RBM offsets: " << getRelativeOffsetB() << ", " << getRelativeOffsetW() << ", " << getRelativeOffsetC() << ", " << std::get<3>(myOffsets) << std::endl;
				}
			
			// Pass a shared_array_ptr
			RBM(Layer aInput,  Layer aOutput, shared_array_ptr<double> aData, bool isAlreadyPretrained = false): input(aInput), output(aOutput), myOffsets(computeOffsets(aInput, aOutput)), 
				myData(aData > std::get<3>(myOffsets)),
				b(aData.getOffsetData(), nInput()), c(aData.getOffsetData() + std::get<2>(myOffsets), nOutput()), W(aData.getOffsetData() + std::get<1>(myOffsets), nOutput(), nInput()),
				pretrained(isAlreadyPretrained) {
	//				std::cout << "RBM offsets: " << std::get<0>(myOffsets) << ", " << std::get<1>(myOffsets) << ", " << std::get<2>(myOffsets) << ", " << std::get<3>(myOffsets) << std::endl;
				}
	
			/* Accessors */
			Eigen_size_type nInput() const {return input.getSize();}
			Eigen_size_type nOutput() const {return output.getSize();}
			Eigen_size_type nWeights() const {return nInput() * nOutput();}
			size_t totalSize() const {return std::get<3>(myOffsets);}
			Layer::Type tInput() const {return input.getType();}
			Layer::Type tOutput() const {return output.getType();}
			std::string sInput() const {return input.getTypeAsString();}
			std::string sOutput() const {return output.getTypeAsString();}
			Layer getInput() const {return input;}
			Layer getOutput() const {return output;}
			shared_array_ptr<double> getData() const {return myData;}
			/** Get the data. The following functions provide various ways to get it, as a raw pointer, shared_array_ptr or Eigen array/matrix */
			double* getBAsPtr() const {return myData.getOffsetData() /* + std::get<0>(myOffsets) always 0 */;}
			double* getWAsPtr() const {return myData.getOffsetData() + std::get<1>(myOffsets);}
			double* getCAsPtr() const {return myData.getOffsetData() + std::get<2>(myOffsets);}
			shared_array_ptr<double> getBAsSharedArrayPtr() const {return myData > boost::numeric_cast<std::size_t>(nInput());}
			shared_array_ptr<double> getWAsSharedArrayPtr() const {return myData + std::get<1>(myOffsets) > boost::numeric_cast<std::size_t>(nWeights());}
			shared_array_ptr<double> getCAsSharedArrayPtr() const {return myData + std::get<2>(myOffsets) > boost::numeric_cast<std::size_t>(nOutput());}
			ArrayX1dMap getB() const {return b;}
			ArrayX1dMap getC() const {return c;}
			MatrixXdMap getW() const {return W;}
			/** Setting weights and biases directly with Eigen matrices with setB, setC and setW. */
			RBM& setB(ArrayX1d aNewB);
			RBM& setC(ArrayX1d aNewC);
			RBM& setW(Eigen::MatrixXd aNewW);
			RBM& setB(ArrayX1dMap aNewB);
			RBM& setC(ArrayX1dMap aNewC);
			RBM& setW(MatrixXdMap aNewW);
			
			size_t getRelativeOffsetB() const {return 0;} // Get offset in shared_array_ptr<double> or double* relative to the beginning of the shared_array_ptr of this RBM
			size_t getRelativeOffsetW() const {return std::get<1>(myOffsets);} 
			size_t getRelativeOffsetC() const {return std::get<2>(myOffsets);} 
			offsets getOffsets() const {return myOffsets;}
			bool isPretrained() const {return pretrained;}
			
			/* Training the net */
			RBM& pretrain(const Eigen::MatrixXd&, const PretrainParameters&, PretrainProgress& aProgressFunctor = NoOpPretrainProgress::getInstance(), const ContinueFunction& aContinueFunction = ContinueFunction::getInstance());
			
			/* Predictions & cie */
			Eigen::MatrixXd predict(Eigen::MatrixXd data) const {forwardsDataToActivitiesInPlace(data);return data;}
			void predictInPlace(Eigen::MatrixXd& data) const {forwardsDataToActivitiesInPlace(data);}
			Eigen::MatrixXd reverse_predict(Eigen::MatrixXd data) const {backwardsHiddenToActivitiesInPlace(data); return data;}
			void reverse_predictInPlace(Eigen::MatrixXd& data) const {backwardsHiddenToActivitiesInPlace(data);}
			Eigen::MatrixXd reconstruct(Eigen::MatrixXd data) const {predictInPlace(data); reverse_predictInPlace(data); return data;}
			void reconstructInPlace(Eigen::MatrixXd& data) const {predictInPlace(data); reverse_predictInPlace(data);}
			/* Sampling */
			Eigen::MatrixXd sample(const Eigen::MatrixXd& data) const;
			//Eigen::MatrixXd sampleInPlace(Eigen::MatrixXd& data) const;

			/** Computes the squared error of the reconstruction, per data point, and return it in a vector.
			 *  errorSum computes the sum of error over all data points and returns a single double.
			 *  The behaviour is different on unrolled networks: the hidden layer *is* the reconstruction, whereas on non-unrolled networks reverse_predict is used
			 *  to get the reconstructions. This is done through the reconstruct() function.
			 */
			/* ArrayX1d error(const Eigen::MatrixXd&) const;
			double errorSum(const Eigen::MatrixXd&) const;
			ArrayX1d error(const Eigen::MatrixXd& data, const Eigen::MatrixXd& reconstructions) const;
			double errorSum(const Eigen::MatrixXd& data, const Eigen::MatrixXd& reconstructions) const; 
			*/
			/** The new error simply calculates the root of the squared gradient vectors (after penalization),
			 * but without the training rate, per data point.
			 * errorSum computes the sum of error over all data points and returns a single double.
			 * There is no 'error' function any longer as the gradient is already an average
			 * over all data points.
			 * In addition, it is not available outside the pre-training so it is not a public method.
			 */
		private:
			double errorSum(const ArrayX1d&, const ArrayX1d& deltaC, const Eigen::ArrayXXd& deltaW) const;

			/** Computes the enery of the network, per data point, and return it in a vector 
			 * energySum computes the sum of error over all data points and returns a single double
			 */
		public:
			ArrayX1d energy(const Eigen::MatrixXd&) const;
			double energySum(const Eigen::MatrixXd&) const;
		
			/* clone */
			RBM clone() const;
			RBM reverse() const;
			/* Destructor */ 
			//~RBM() { if (cleanUp) {delete b; delete c; delete W;}} // pointers now handled by shared_array_ptr
	};
}
