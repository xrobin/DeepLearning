#include <string>
using std::string;

#include <DeepLearning/Progress.h>
#include <DeepLearning/RBM.h>


namespace DeepLearning {
	void AcceleratePretrainProgress::propagateData(const RBM& anRBM) {if (testData.size() > 0) testData = anRBM.predict(testData);} // do nothing if no data...
	void EachStepPretrainProgress::propagateData(const RBM& anRBM) {if (testData.size() > 0) testData = anRBM.predict(testData);}
}
