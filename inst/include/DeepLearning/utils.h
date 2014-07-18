#pragma once 

#include <cmath> // std::tanh
#include <vector>


namespace DeepLearning {
	/** Element-wise tanh (tangent hyperbolic)
	 * 
	 * Takes an object and executes std::tanh on all its elements.
	 * Designed for Eigen matrices, it accepts any object as long as they implement .data() (a pointer to a double) and .size() (length of the data)
	 */
	template <typename T> T& tanhInPlace(T& anEigenObject) {
		auto ncomponenents = anEigenObject.size();
		double* theData = anEigenObject.data();
		for (auto i = 0; i < ncomponenents; ++i) {
			theData[i] = std::tanh(theData[i]);
		}
		return anEigenObject;
	}
	
	/** Checks if element is present in the container */
	template<typename T> bool isIn(const std::vector<T>& container, const T element) {
		return std::find(container.begin(), container.end(), element) != container.end();
	}
}
