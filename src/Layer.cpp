#include <stdexcept>
#include <string>
using std::string;

#include <DeepLearning/Layer.h>
using namespace DeepLearning;


namespace DeepLearning {
	Layer::Type Layer::getType() const {
		return type;
	}
	
	string Layer::getTypeAsString() const {
		switch(type) {
			case(gaussian): return "gaussian";
			case(binary): return "binary";
			case(continuous): return "continuous";
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
}
