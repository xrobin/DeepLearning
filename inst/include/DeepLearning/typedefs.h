#pragma once

#define UNUSED(x) (void)(x) // hide unused parameters warnings

#include <Eigen/Dense>

#include <tuple>
#include <functional> // std::function
#include <vector>


namespace DeepLearning {
	typedef Eigen::Array<double, Eigen::Dynamic, 1> ArrayX1d;
	typedef Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 1>> ArrayX1dMap;
	typedef Eigen::Map<Eigen::MatrixXd> MatrixXdMap;
	typedef Eigen::MatrixXd::Index Eigen_size_type;
	
	typedef std::tuple<size_t, size_t, size_t, size_t> offsets;
	
	typedef std::function<bool(std::vector<double>, unsigned int, size_t, unsigned int maxiters, size_t layer)> continueFunctionType;
	
	class RBM; class DeepBeliefNet;
	typedef std::function<void(const RBM& anRBM, const Eigen::MatrixXd& batch, const Eigen::MatrixXd& data, const unsigned int iter, const size_t batchsize, const unsigned int maxiters, const size_t layer)> pretrainDiagFunctionType;
	typedef std::function<void(const DeepBeliefNet& aDBN, const Eigen::MatrixXd& batch, const Eigen::MatrixXd& data, const unsigned int iter, const size_t batchsize, const unsigned int maxiters)> trainDiagFunctionType;
}
