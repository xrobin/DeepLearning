#include <Rcpp.h>
using Rcpp::List;
using Rcpp::NumericVector;
using Rcpp::Environment;
using Rcpp::as;
#include <RcppEigen.h> 

#include <string>
using std::string;
#include <stdexcept> // throw std::runtime_error, std::invalid_argument
using std::runtime_error;
#include <tuple>
#include <vector>
using std::vector;
#include <iostream>
#include <memory> // std::unique_ptr
using std::unique_ptr;

#include "boost/numeric/conversion/cast.hpp" // safe numeric_cast

#include "RcppConversions.h"
#include "DeepBeliefNet.h" 
#include "RBM.h"
#include "Layer.h"

// define template specialisations for as and wrap
namespace Rcpp {
	// size_t
	template <> size_t as(SEXP ptr) {
		int i(as<int>(ptr));
		return boost::numeric_cast<size_t>(i);
	}
	
	// shared_array_ptr<double>
	template <> shared_array_ptr<double> as(SEXP ptr) {
		NumericVector NumericVectorPtr(as<NumericVector>(ptr));
		return shared_array_ptr<double>(NumericVectorPtr.begin(), NumericVectorPtr.end(), true); // that's a copy, so clean-up
	}

	template <> SEXP wrap(const shared_array_ptr<double> &ptr) {
		return wrap(ptr.toVector());
	}
	
	SEXP wrap_with_additionalOffset(const offsets &someOffsets, size_t additionalOffset) {
		vector<size_t> offsetsVec = {additionalOffset + std::get<0>(someOffsets), additionalOffset + std::get<1>(someOffsets), 
			additionalOffset + std::get<2>(someOffsets), additionalOffset + std::get<3>(someOffsets)};
		return wrap(offsetsVec);
	}

	// Offsets (weights.breaks)
	template <> SEXP wrap(const offsets &someOffsets) {
		return(wrap_with_additionalOffset(someOffsets, 0));
	}

	// Layer
	template <> Layer as(SEXP layer) {
		List layerList = as<List>(layer);
		if (as<string>(layerList.attr("class")) != "Layer") {
			throw runtime_error("Expected a Layer object, not " + as<string>(layerList.attr("class")));
		}
		return Layer(as<unsigned int>(layerList["size"]), as<string>(layerList["type"]));	
	}

	template <> SEXP wrap(const Layer &layer) {
		/* See R/Layer.R where the Layer are created in R */
		List layerList = List::create(
			Named("size") = wrap(boost::numeric_cast<int>(layer.getSize())),
			Named("type") = wrap(layer.getTypeAsString())
		);
		layerList.attr("class") = "Layer";
		return wrap(layerList);
	}

	// RBM
	template <> RBM as(SEXP rbm) {
		// Make input a List
		List rbmList = as<List>(rbm);
		if (as<string>(rbmList.attr("class")) != "RestrictedBolzmannMachine") {
			throw runtime_error("Expected a RestrictedBolzmannMachine object, not " + as<string>(rbmList.attr("class")));
		}

		// Grab weights in env
		Environment env = as<Environment>(rbmList["weights.env"]);
		NumericVector weights = as<NumericVector>(env["weights"]);
		
		// Make shared_array_ptr
		shared_array_ptr<double> data(as<shared_array_ptr<double>>(weights));
	
		// Build the RBM
		Layer input(as<Layer>(rbmList["input"]));
		Layer output(as<Layer>(rbmList["output"]));
		bool isAlreadyPretrained(as<bool>(rbmList["pretrained"]));

		return RBM(input,  output, data, isAlreadyPretrained);	
	}
	
	SEXP wrap_with_env(const RBM& rbm, const Environment& env, size_t additionalOffset) {	
		List rbmList = List::create(
			Named("input") = rbm.getInput(),
			Named("output") = rbm.getOutput(),
			Named("weights.env") = env,
			Named("weights.breaks") = wrap_with_additionalOffset(rbm.getOffsets(), additionalOffset),
			Named("pretrained") = rbm.isPretrained()
		);
		rbmList.attr("class") = "RestrictedBolzmannMachine";

		return wrap(rbmList);
	}

	template <> SEXP wrap(const RBM& rbm) {		
		Environment env = Rcpp::Environment::namespace_env("DeepLearning").new_child(true);
		env["weights"] = rbm.getData();
		env["breaks"] = rbm.getOffsets();
		return wrap_with_env(rbm, env);
	}

	// DeepBeliefNet
	template <> DeepBeliefNet as(SEXP dbn) {
		// Make input a List
		List dbnList = as<List>(dbn);
		if (as<string>(dbnList.attr("class")) != "DeepBeliefNet") {
			throw runtime_error("Expected a DeepBeliefNet object, not " + as<string>(dbnList.attr("class")));
		}

		// Grab weights in env
		Environment env = as<Environment>(dbnList["weights.env"]);
		NumericVector weights = as<NumericVector>(env["weights"]);
		
		// Make shared_array_ptr
		shared_array_ptr<double> dataPtr = as<shared_array_ptr<double>>(weights);
		
		// Build the list of layers
		List LayersList = as<List>(dbnList["layers"]);

		// Build the output Layers list
		std::vector<Layer> LayersVector;
		LayersVector.reserve(boost::numeric_cast<size_t>(LayersList.size()));
		for (auto aLayer : LayersList) {
			LayersVector.push_back(as<Layer>(aLayer));
		}
		
		bool pretrained = as<bool>(dbnList["pretrained"]);
		bool unrolled = as<bool>(dbnList["unrolled"]);
		bool finetuned = as<bool>(dbnList["finetuned"]);
		
		return DeepBeliefNet(LayersVector, dataPtr, pretrained, unrolled, finetuned);
	}
	
	template <> SEXP wrap(const DeepBeliefNet &dbn) {
		Environment env = Rcpp::Environment::namespace_env("DeepLearning").new_child(true);
		env["weights"] = dbn.getData();
		
		// Push Layers into a List
		List layersList;
		for (Layer layer: dbn.getLayers()) {
			layersList.push_back(layer);
		}
		
		// Push RBMs into a List
		List rbmsList;
		vector<size_t> allOffsets;
		allOffsets.reserve(dbn.getRBMs().size() * 2 + 2);
		allOffsets.push_back(std::get<0>(dbn.getRBM(0).getOffsets()));
		allOffsets.push_back(std::get<1>(dbn.getRBM(0).getOffsets()));
		size_t additionalOffset = 0;
		for (RBM rbm: dbn.getRBMs()) {
			rbmsList.push_back(wrap_with_env(rbm, env, additionalOffset));
			allOffsets.push_back(additionalOffset + std::get<2>(rbm.getOffsets()));
			allOffsets.push_back(additionalOffset + std::get<3>(rbm.getOffsets()));
			additionalOffset += std::get<2>(rbm.getOffsets());
		}
		
		// Push breaks into env
		env["breaks"] = allOffsets;
		
		// Build return object
		List dbnList = List::create(
			Named("layers") = wrap(layersList),
			Named("weights.env") = wrap(env),
			Named("rbms") = wrap(rbmsList),
			Named("pretrained") = wrap(dbn.isPretrained()),
			Named("unrolled") = wrap(dbn.isUnrolled()),
			Named("finetuned") = wrap(dbn.isFinetuned())
		);
		dbnList.attr("class") = "DeepBeliefNet";
		return wrap(dbnList);
	}
	
	// PretrainParameters
	template <> PretrainParameters as(SEXP someParams) {
		List paramList(as<List>(someParams));
		PretrainParameters params;
		try {
			if (paramList.containsElementNamed("penalization")) params.setPenalization(as<std::string>(paramList["penalization"]));
		} catch(std::invalid_argument) {} // Don't do anything - we'll get invalid penalizations with skip...
		if (paramList.containsElementNamed("lambda.b")) params.setLambdaB(as<double>(paramList["lambda.b"]));
		if (paramList.containsElementNamed("lambda.c")) params.setLambdaC(as<double>(paramList["lambda.c"]));
		if (paramList.containsElementNamed("lambda.W")) params.setLambdaW(as<double>(paramList["lambda.W"]));
		if (paramList.containsElementNamed("epsilon.b")) params.setEpsilonB(as<double>(paramList["epsilon.b"]));
		if (paramList.containsElementNamed("epsilon.c")) params.setEpsilonC(as<double>(paramList["epsilon.c"]));
		if (paramList.containsElementNamed("epsilon.W")) params.setEpsilonW(as<double>(paramList["epsilon.W"]));
		if (paramList.containsElementNamed("train.b")) params.setTrainB(as<bool>(paramList["train.b"]));
		if (paramList.containsElementNamed("train.c")) params.setTrainC(as<bool>(paramList["train.c"]));

		if (paramList.containsElementNamed("momentum")) { // we have a momentum
			const SEXP parasexp = paramList["momentum"];
			if (!Rf_isNull(parasexp)) { // and it's not null
				if (Rf_isReal(parasexp)) { // it's a vector!
					params.setMomentum(as<std::vector<double>>(parasexp));
				}
				else { // defaults to list
					List tmpList(as<List>(parasexp));
					try {
						params.setMomentum(as<std::vector<double>>(tmpList[0]));
					} catch(std::exception e) {}
				}
			}
		}
		
		if (paramList.containsElementNamed("miniters")) params.setMinIters(boost::numeric_cast<unsigned int>(as<int>(paramList["miniters"])));
		if (paramList.containsElementNamed("maxiters")) params.setMaxIters(boost::numeric_cast<unsigned int>(as<int>(paramList["maxiters"])));
		if (paramList.containsElementNamed("batchsize")) params.setBatchSize(boost::numeric_cast<size_t>(as<int>(paramList["batchsize"])));
		if (paramList.containsElementNamed("n.proc")) params.setNbThreads(as<int>(paramList["n.proc"]));
		params.ensureValidity();
		return params;
	}
	
	template <> vector<PretrainParameters> as(SEXP someParams) {
		List paramForAllLayers(as<List>(someParams));
		vector<PretrainParameters> out;
		out.reserve(boost::numeric_cast<size_t>(paramForAllLayers.size()));
		for (auto layerMomentum : paramForAllLayers) {
			out.push_back(as<PretrainParameters>(layerMomentum));
		}
		return out;
	}
	
	// TrainParameters
	template <> TrainParameters as(SEXP someParams) {
		List paramList(as<List>(someParams));
		TrainParameters params;
		
		if (paramList.containsElementNamed("batchsize")) params.setBatchSize(as<size_t>(paramList["batchsize"]));
		if (paramList.containsElementNamed("n.proc")) params.setNbThreads(as<int>(paramList["n.proc"]));
		if (paramList.containsElementNamed("miniters")) params.setMinIters(as<unsigned int>(paramList["miniters"]));
		if (paramList.containsElementNamed("maxiters")) params.setMaxIters(as<unsigned int>(paramList["maxiters"]));

		if (paramList.containsElementNamed("optim.control")) {
			params.setCgMinParams(as<CgMinParams>(paramList["optim.control"]));
		}
		
		return params;
	}
	
	template <> CgMinParams as(SEXP someParams) {
		List paramList(as<List>(someParams));
		CgMinParams params;
		
		if (paramList.containsElementNamed("type")) params.setAlgorithmType(as<int>(paramList["type"]));
		if (paramList.containsElementNamed("trace")) params.setAlgorithmType(as<int>(paramList["trace"]));
		if (paramList.containsElementNamed("maxit")) params.setMaxCgIters(as<unsigned int>(paramList["maxit"]));
		if (paramList.containsElementNamed("steplength")) params.setStepLength(as<double>(paramList["steplength"]));
		if (paramList.containsElementNamed("stepredn")) params.setSteredn(as<double>(paramList["stepredn"]));
		if (paramList.containsElementNamed("reltest")) params.setReltest(as<double>(paramList["reltest"]));
		if (paramList.containsElementNamed("acctol")) params.setAcctol(as<double>(paramList["acctol"]));
		if (paramList.containsElementNamed("abstol")) params.setAbstol(as<double>(paramList["abstol"]));
		if (paramList.containsElementNamed("intol")) params.setIntol(as<double>(paramList["intol"]));
		if (paramList.containsElementNamed("setstep")) params.setSetstep(as<double>(paramList["setstep"]));

		return params;
	}
	
	// TrainProgress
	template <> unique_ptr<TrainProgress> as(SEXP aDiag) {
		const List aDiagList(as<List>(aDiag));
		const string aDiagRate(as<string>(aDiagList["rate"]));
		
		if (aDiagRate == "none") {
			return unique_ptr<TrainProgress>(new NoOpTrainProgress());
		}		
		
		unique_ptr<TrainProgress> ptr;
		if (aDiagRate == "each") {
			ptr.reset(new EachStepTrainProgress());
		}
		else if (aDiagRate == "accelerate") {
			ptr.reset(new AccelerateTrainProgress());
		}
		else {
			stop("invalid value for diag.rate");
		}

		if (aDiagList.containsElementNamed("data") && !Rf_isNull(aDiagList["data"])) {
			const Eigen::Map<Eigen::MatrixXd> aTestData(as<Eigen::Map<Eigen::MatrixXd>>(aDiagList["data"]));
			ptr->setData(aTestData.transpose());
		}
		if (aDiagList.containsElementNamed("f")) {
			const Rcpp::Function myRFunction = as<Rcpp::Function>(aDiagList["f"]);
			ptr->setFunction(
				[myRFunction](const DeepBeliefNet& aDBN, const Eigen::MatrixXd& batch, const Eigen::MatrixXd& data, const unsigned int iter, const size_t batchsize, const unsigned int maxiters) -> void {
					myRFunction(aDBN, batch.transpose(), data.transpose(), iter, batchsize, maxiters);
				}
			);
		}
		return ptr;
	}
	
	// PretrainProgress
	template <> unique_ptr<PretrainProgress> as(SEXP aDiag) {
		const List aDiagList(as<List>(aDiag));
		const string aDiagRate(as<string>(aDiagList["rate"]));
		
		if (aDiagRate == "none") {
			return unique_ptr<PretrainProgress>(new NoOpPretrainProgress());
		}
		
		unique_ptr<PretrainProgress> ptr;
		if (aDiagRate == "each") {
			ptr.reset(new EachStepPretrainProgress());
		}
		else if (aDiagRate == "accelerate") {
			ptr.reset(new AcceleratePretrainProgress());
		}
		else {
			stop("invalid value for diag.rate");
		}

		if (aDiagList.containsElementNamed("data") && !Rf_isNull(aDiagList["data"])) {
			const Eigen::Map<Eigen::MatrixXd> aTestData(as<Eigen::Map<Eigen::MatrixXd>>(aDiagList["data"]));
			ptr->setData(aTestData.transpose());
		}
		if (aDiagList.containsElementNamed("f")) {
			const Rcpp::Function myRFunction = as<Rcpp::Function>(aDiagList["f"]);
			ptr->setFunction(
				[myRFunction](const RBM& anRBM, const Eigen::MatrixXd& batch, const Eigen::MatrixXd& data, const unsigned int iter, const size_t batchsize, const unsigned int maxiters, const size_t layer) -> void {
					myRFunction(anRBM, batch.transpose(), data.transpose(), iter, batchsize, maxiters, layer);
				}
			);
		}
		
		return ptr;
	}

	// ContinueFunction
	template <> ContinueFunction as(SEXP aCont) {
		const List aContList(as<List>(aCont));
		ContinueFunction cf;
		if (aContList.containsElementNamed("continue.function.frequency")) cf.setFrequency(as<unsigned int>(aContList["continue.function.frequency"]));
		if (aContList.containsElementNamed("continue.stop.limit")) cf.setLimit(as<unsigned int>(aContList["continue.stop.limit"]));
		if (aContList.containsElementNamed("continue.function")) {
			Rcpp::Function myRFunction = as<Rcpp::Function>(aContList["continue.function"]);
			cf.setContinueFunction(
				[myRFunction](std::vector<double> error, unsigned int iter, size_t batchsize, unsigned int maxiters, size_t layer) -> bool {
					bool ret = as<bool>(myRFunction(error, iter, batchsize, maxiters, layer));
					return ret;
				}
			);
		}
		return cf;
	}
}
