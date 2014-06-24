#include <string>
#include <algorithm> // std::transform
#include <stdexcept> // std::invalid_argument
#include <iostream>

#include "PretrainParameters.h"

const std::string PretrainParameters::invalid_momentums = "momentums of wrong size: should be 1, 2 or maxiters";
const std::string PretrainParameters::invalid_penalization = "newPenalization not l1 or l2";

std::string PretrainParameters::PenalizationTypeToString(PenalizationType aPT) {
	return aPT == l1 ? "l1" : "l2";
}

PretrainParameters::PenalizationType PretrainParameters::PenalizationTypeFromString(std::string aString) {	
	std::transform(aString.begin(), aString.end(), aString.begin(), ::tolower);
	if (aString == "l1") {
		return l1;
	}
	else if (aString == "l2") {
		return l2;
	}
	else {
		throw std::invalid_argument(invalid_penalization);
	}
}