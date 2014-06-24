#pragma once

#define UNUSED(x) (void)(x) // hide unused parameters warnings

#include <Eigen/Dense>

#include <tuple>

typedef Eigen::Array<double, Eigen::Dynamic, 1> ArrayX1d;
typedef Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 1>> ArrayX1dMap;
typedef Eigen::Map<Eigen::MatrixXd> MatrixXdMap;
typedef Eigen::MatrixXd::Index Eigen_size_type;

typedef std::tuple<size_t, size_t, size_t, size_t> offsets;
