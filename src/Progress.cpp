#include <string>
using std::string;

#include "Progress.h"
#include "RBM.h"


void AcceleratePretrainProgress::propagateData(const RBM& anRBM) {testData = anRBM.predict(testData);}
void EachStepPretrainProgress::propagateData(const RBM& anRBM) {testData = anRBM.predict(testData);}