#include <Eigen/Dense>

#include <string>
#include <random> // std::mt19937, std::random_device, std::uniform_int_distribution, std::uniform_real_distribution, std::normal_distribution
#include <functional> // std::bind
using std::bind;
#include <stdexcept> // throw std::invalid_argument
#include "boost/numeric/conversion/cast.hpp"

#include "Random.h"

using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::Map;

/*std::mt19937 warmup_mt19937() {
	std::array<int, std::mt19937::state_size> seed_data;
	std::random_device r;
	std::generate_n(seed_data.data(), seed_data.size(), std::ref(r));
	std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
	return std::mt19937(seq);
}*/

void Random::build(const std::string& type) {
	std::random_device rd;
	std::mt19937 rng_engine(rd());
	if (type == "gaussian") {
        std::normal_distribution<double> dist(0.0, 1.0);
        rngDouble = bind(dist, rng_engine);
	}
	else if (type == "uniform_int") {
		throw std::invalid_argument("'max' is required with type = 'uniform_int'");
	}
	else {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        rngDouble = bind(dist, rng_engine);
	}
}

void Random::build(const std::string& type, size_t max) {
	std::random_device rd;
	std::mt19937 rng_engine(rd());
	if (type == "uniform_int") {
        std::uniform_int_distribution<int> dist(0.0, boost::numeric_cast<int>(max - 1));
        rngInt = bind(dist, rng_engine);
	}
	else {
		throw std::invalid_argument("'max' is ignored with type != 'uniform_int'");
	}
}
		
void Random::setBatch(const MatrixXd& data, MatrixXd& batch) {
	auto batchsize = batch.cols();
	for (auto i = 0; i < batchsize; i++) {
		batch.col(i) = data.col(rngInt());
	}
}
		
void Random::setRandom(ArrayXXd& array) {
	for (int i = 0; i < array.size(); i++) {
		*(array.data() + i) = rngDouble();
	}
}
