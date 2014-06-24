#pragma once 

#include <vector>
#include <string> 
#include <stdexcept>

/**
 * Structure defining the parameters for the pretraining
 * Contains the following members:
 * 	 - double lambdaB, lambdaC, lambdaW: default 0;
 *	 - double epsilonB, epsilonC, epsilonW: default 0.001;
 *	 - vector<double> momentums: default 0;
 *	 - size_t maxiters: default 100;
 *	 - size_t batchsize: default 100; special value 0: data size / maxiters
 *   - unsigned int nProcs: default 0 (special Eigen value = no parallel execution)
 *	 - enum penalization {l1, l2}: default l1;
 *   - bool trainB, trainC: default TRUE;
 * 
 * All members can be set directly or trough the set* functions.
 * Note the convenience functions setLambda and setEpsilon that will set all 
 * lambda* and epsilon*, respectively.
 * All set* functions return the object so you can stack them:
 *     myPretrainParameters.setLambda(0.1).setMaxiters(1000)
 * 
 * Momentums are special:
 *   - momentums must be a vector<double> of length 1, 2 or maxiters.
 *   - if length is 1, momentum is constant
 *   - if length is maxiters, defines the momentum at each iteration
 *   - if length is 2, defines a sequence as in R's seq(from = momentums[1], to = momentums[2], by = diff(momentums) / (maxiters - 1)).
 *   - a valid momentums vector of size maxiters can be accessed with getValidMomentums()
 *   - invalid momentums might be set, and could prevent getValidMomentums() from working later (this is because momentums / maxiters might be set separately)
 *   - use validate() to ensure the structure is correct, or better ensureValidity() just after setting momentums or maxiters.
 */

struct PretrainParameters {
	double lambdaB, lambdaC, lambdaW;
	double epsilonB, epsilonC, epsilonW;
	std::vector<double> momentums;
	size_t maxiters, batchsize;
	int nbThreads;
	enum PenalizationType {l1, l2};
	PenalizationType penalization;
	static std::string PenalizationTypeToString(PenalizationType);
	static PenalizationType PenalizationTypeFromString(std::string aString);
	bool trainB, trainC;
	
	PretrainParameters& setLambda(double newLambda) {lambdaB = lambdaC = lambdaW = newLambda; return *this;}
	PretrainParameters& setLambdaB(double newLambdaB) {lambdaB = newLambdaB; return *this;}
	PretrainParameters& setLambdaC(double newLambdaC) {lambdaC = newLambdaC; return *this;}
	PretrainParameters& setLambdaW(double newLambdaW) {lambdaW = newLambdaW; return *this;}
	PretrainParameters& setEpsilon(double newEpsilon) {epsilonB = epsilonC = epsilonW = newEpsilon; return *this;}
	PretrainParameters& setEpsilonB(double newEpsilonB) {epsilonB = newEpsilonB; return *this;}
	PretrainParameters& setEpsilonC(double newEpsilonC) {epsilonC = newEpsilonC; return *this;}
	PretrainParameters& setEpsilonW(double newEpsilonW) {epsilonW = newEpsilonW; return *this;}
	PretrainParameters& setMaxiters(unsigned int newMaxiters) {maxiters = newMaxiters; return *this;}
	PretrainParameters& setBatchsize(size_t newBatchsize) {batchsize = newBatchsize; return *this;}
	PretrainParameters& setTrainB(bool newTrainB) {trainB = newTrainB; return *this;}
	PretrainParameters& setTrainC(bool newTrainC) {trainC = newTrainC; return *this;}
	PretrainParameters& setNbThreads(int newNbThreads) {nbThreads = newNbThreads; return *this;}
	PretrainParameters& setPenalization(PenalizationType newPenalization) {penalization = newPenalization; return *this;}
	PretrainParameters& setPenalization(std::string newPenalization) {
		penalization = PenalizationTypeFromString(newPenalization);
		return *this;
	}
	PretrainParameters& setMomentum(double newMomentum) {momentums.clear(); momentums.push_back(newMomentum); return *this;}
	PretrainParameters& setMomentum(std::vector<double> newMomentums) {momentums =newMomentums; return *this;}
	void ensureValidity() const {
		if (!validate()) {
			throw std::length_error(invalid_momentums); 
		}
	}
	bool validate() const {
		if (momentums.size() != 1 && momentums.size() != 2 && momentums.size() != maxiters) {
			return false;
		}
		return true;
	}
	
	std::vector<double> getValidMomentums() const {
		if (momentums.size() == 1) return std::vector<double>(maxiters, momentums[0]);
		if (momentums.size() == 2) return getMomentumsFromLengthTwo(momentums);
		if (momentums.size() == maxiters) return momentums;
		// If we're still here it's an error:
		throw std::length_error(invalid_momentums);
	}
	
	PretrainParameters() : lambdaB(0.0), lambdaC(0.0), lambdaW(0.0), 
						epsilonB(0.001)	, epsilonC(0.001), epsilonW(0.001), momentums(1, 0.0),
						maxiters(100), batchsize(100), nbThreads(0), penalization(l1),
						trainB(true), trainC(true) {}
	
	private:
		std::vector<double> getMomentumsFromLengthTwo(const std::vector<double>& someMomentums) const {
			double increment = (someMomentums[1] - someMomentums[0]) / (double(maxiters) - 1);
			double newMomentum = someMomentums[0];
			std::vector<double> momentumsLengthMaxiters;
			momentumsLengthMaxiters.reserve(maxiters);
			for (size_t i = 0; i < maxiters; i++) {
				momentumsLengthMaxiters.push_back(newMomentum);
				newMomentum += increment;
			}
			return momentumsLengthMaxiters;
		}
		const static std::string invalid_momentums;
		const static std::string invalid_penalization;
};

