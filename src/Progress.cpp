#include <string>
using std::string;

#include "Progress.h"
#include "RBM.h"


void AcceleratePretrainProgress::propagateData(const RBM& anRBM) {if (testData.size() > 0) testData = anRBM.predict(testData);} // do nothing if no data...
void EachStepPretrainProgress::propagateData(const RBM& anRBM) {if (testData.size() > 0) testData = anRBM.predict(testData);}