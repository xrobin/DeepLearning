#include <stdexcept>
#include <string>
using std::string;

#include "Layer.h"

Layer::Type Layer::getType() const {
	return type;
}

string Layer::getTypeAsString() const {
	switch(type) {
		case(gaussian): return "gaussian";
		case(binary): return "binary";
		case(continuous): return "continuous";
		default: throw std::runtime_error("Undefined type!");
	};
}

unsigned int Layer::getSize() const {
	return size;
}

Layer::Type Layer::typeFromString(const string& typeStr) {
	if (typeStr == "binary") {
		return Layer::binary; 
	}
	else if (typeStr == "gaussian") {
		return Layer::gaussian;
	}
	else if (typeStr == "continuous") {
		return Layer::continuous;
	}
	throw std::invalid_argument("Unknown type string: " + typeStr + "!");
}
