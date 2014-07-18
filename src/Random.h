#pragma once

#include <Eigen/Dense>

#include <string>

#include "Layer.h"

#include "typedefs.h"

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
		void setBatch(const Eigen::MatrixXd&, Eigen::MatrixXd&);
		void setRandom(Eigen::ArrayXXd&);
};
