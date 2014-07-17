#pragma once

#include <Eigen/Dense>

#include <string>

#include "Layer.h"

#include "typedefs.h"

class Random  {
	std::function<size_t()> rngInt;
	std::function<double()> rngDouble;
	void build(const std::string& type, size_t max = -1);

	public:
		Random(const std::string type, size_t max = -1): rngInt(), rngDouble() {build(type, max);}
		Random(Layer::Type type): rngInt(), rngDouble() {
			if (type == Layer::Type::gaussian) {
				build("gaussian");
			}
			else {
				build("");
			}
		}
		void setBatch(const Eigen::MatrixXd&, Eigen::MatrixXd&);
		void setRandom(Eigen::ArrayXXd&);
};
