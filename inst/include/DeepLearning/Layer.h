#pragma once

#include <string>


namespace DeepLearning {
	/**
	 * A layer in the network. Two layers make up a RBM. Is defined by a size (int) and type (of Enum type, either binary, gaussian or continuous)
	 */
	class Layer {
		public:
			enum Type {binary, gaussian, continuous};
			Layer(unsigned int aSize, Type aType) : size(aSize), type(aType) {}
			Layer(unsigned int aSize, std::string aType) : size(aSize), type(typeFromString(aType)) {}
			
			Type getType() const;
			std::string getTypeAsString() const;
			unsigned int getSize() const;
		
		private:
			unsigned int size;
			Type type;
			static Type typeFromString(const std::string&);
			
	};
}
