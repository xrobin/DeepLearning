#pragma once

#include <Eigen/Dense>

#include <string>

#include <DeepLearning/Layer.h>
#include <DeepLearning/typedefs.h>


namespace DeepLearning {
	class Random  {
		std::function<int()> rngInt;
		std::function<double()> rngDouble;
		void build(const std::string& type, size_t max);
		void build(const std::string& type);
	
		public:
			Random(const std::string type, size_t max): rngInt(), rngDouble() {build(type, max);}
			Random(const std::string type): rngInt(), rngDouble() {build(type);}
			Random(Layer::Type type): rngInt(), rngDouble() {
				if (type == Layer::Type::gaussian) {
					build("gaussian");
				}
				else {
					build("");
				}
			}
			Random(Layer::Type type, size_t max): rngInt(), rngDouble() {
				if (type == Layer::Type::gaussian) {
					build("gaussian", max);
				}
				else {
					build("", max);
				}
			}
			/** Creates a batch by extracting random columns of data */
			void setBatch(const Eigen::MatrixXd& data, Eigen::MatrixXd& batch);
			/** Fill an entire array with random values */
			void setRandom(Eigen::ArrayXXd& array);
			/** Replace missing values in array with random values */
			void fillMissing(Eigen::ArrayXXd& array);
	};
}
