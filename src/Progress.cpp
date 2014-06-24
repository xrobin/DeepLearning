#include <string>
using std::string;

#include "Progress.h"
#include "RBM.h"
#include "DeepBeliefNet.h"


/** PretrainProgressDo and TrainProgressDo are the functions that actually do something */
void PretrainProgressDo(RBM& anRBM, Eigen::MatrixXd& aTestData, unsigned int aBatch, size_t currentLayer, std::ostream& anOutputStream) {
	string diag = std::to_string(currentLayer) + ", " + std::to_string(aBatch);
	GenericProgressDo(anRBM, aTestData, diag, anOutputStream);
}

void TrainProgressDo(DeepBeliefNet& aDBN, Eigen::MatrixXd& aTestData, unsigned int aBatch, std::ostream& anOutputStream) {
	string diag = std::to_string(aBatch);
	GenericProgressDo(aDBN, aTestData, diag, anOutputStream);
}

void AcceleratePretrainProgress::propagateData(const RBM& anRBM) {testData = anRBM.predict(testData);}
void EachStepPretrainProgress::propagateData(const RBM& anRBM) {testData = anRBM.predict(testData);}