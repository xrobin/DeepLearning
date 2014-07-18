#include <Rcpp.h>
using Rcpp::Rcout;

#include <boost/format.hpp>
#include <boost/numeric/conversion/cast.hpp> // safe numeric_cast

#include <cmath> // std::isfinite
using std::isfinite;
#include <iostream>
#include <vector>
using std::vector;
#include <stdexcept> // runtime_error

#include "R_optim.h"


namespace DeepLearning {
	/* Conjugate gradients, based on R's  src/appl/optim.c, re-crafted to be stand-alone c++ code
	 * Originally based on Pascal code
	 * in J.C. Nash, `Compact Numerical Methods for Computers', 2nd edition,
	 * converted by p2c then re-crafted by B.D. Ripley
	 */
	void cgmin(size_t n, double *Bvec, double *X, double *Fmin,
			   optimfn fminfn, optimgr fmingr, int *fail,
			   const CgMinParams& params, OptimParameters& ex,
			   unsigned int *fncount, unsigned int *grcount)
	{
		bool accpoint;
		vector<double> c(n), g(n), t(n);
		size_t count, cycle, i;
		double f;
		double G1, G2, G3, gradproj;
		unsigned int funcount = 0, gradcount = 0;
		double newstep, oldstep;
		size_t cyclimit = n;
	
		// Get params from CgMinParams
		double stepredn = params.stepredn; // #defines in appl/optim.c transferred here
		double acctol = params.acctol;
		double reltest = params.reltest; 
		double abstol = params.abstol;
		double intol = params.intol;
		double setstep = params.setstep;
		double steplength = params.steplength;
		int type = params.type; // // 1 (FR), 2 (PR) or 3 (BS)
		int trace = params.trace; // trace, boolean (verbose)
		unsigned int maxit = params.maxCgIters; // params.maxCgIters is stored as unsigned int
	
		if (maxit <= 0) {
			*Fmin = fminfn(ex);
			*fncount = *grcount = 0;
			*fail = false;
			return;
		}
		if (trace) {
			Rcout << "  Conjugate gradients function minimizer" << std::endl;
			switch (type) {
				case 1:		Rcout << "Method: Fletcher Reeves" << std::endl;	break;
				case 2:		Rcout << "Method: Polak Ribiere" << std::endl;		break;
				case 3:		Rcout << "Method: Beale Sorenson" << std::endl;	break;
				default:
					throw std::runtime_error("unknown 'type' in \"CG\" method of 'optim'");
			}
		}
		//c = std::vector(n); g = vect(n); t = vect(n);
		
		*fail = 0;
		double tol = intol * double(n) * sqrt(intol);
		
		if (trace) Rcout << "tolerance used in gradient test=" << tol << std::endl;
		f = fminfn(ex);
		if (!std::isfinite(f)) {
			throw std::runtime_error("Function cannot be evaluated at initial parameters");
		} else {
			*Fmin = f;
			funcount = 1;
			gradcount = 0;
			do {
				for (i = 0; i < n; i++) {
					t[i] = 0.0;
					c[i] = 0.0;
				}
				cycle = 0;
				oldstep = 1.0;
				count = 0;
				do {
					cycle++;
					if (trace) {
						Rcout << gradcount << " " << funcount << " " << *Fmin << std::endl;
						Rcout << "parameters ";
						for (i = 1; i <= n; i++) {
							Rcout << boost::format("%10.5d ") % Bvec[i - 1];
							if (i / 7 * 7 == i && i < n)
								Rcout << std::endl;
						}
						Rcout << std::endl;
					}
					gradcount++;
					if (gradcount > maxit) {
						*fncount = funcount;
						*grcount = gradcount;
						*fail = 1;
						return;
					}
					
					fmingr(g.data(), ex);
					
					G1 = 0.0;
					G2 = 0.0;
					
					for (i = 0; i < n; i++) {
						X[i] = Bvec[i];
						switch (type) {
								
							case 1: /* Fletcher-Reeves */
								G1 += g[i] * g[i];
								G2 += c[i] * c[i];
								break;
								
							case 2: /* Polak-Ribiere */
								G1 += g[i] * (g[i] - c[i]);
								G2 += c[i] * c[i];
								break;
								
							case 3: /* Beale-Sorenson */
								G1 += g[i] * (g[i] - c[i]);
								G2 += t[i] * (g[i] - c[i]);
								break;
								
							default:
								throw std::runtime_error("unknown type in \"CG\" method of 'optim'");
						}
						// DEBUG ONLY:
						if (!isfinite(G1)) {
							Rcout << "G1 =" << G1 << std::endl;
							Rcout << "g[" << i << "] = " << g[i] << std::endl;
							Rcout << "c[" << i << "] = " << c[i] << std::endl;
							throw std::runtime_error("Not a number anymore");
						}
						c[i] = g[i];
					}
					if (G1 > tol) {
						if (G2 > 0.0)
							G3 = G1 / G2;
						else
							G3 = 1.0;
						gradproj = 0.0;
						for (i = 0; i < n; i++) {
							t[i] = t[i] * G3 - g[i];
							gradproj += t[i] * g[i];
						}
						steplength = oldstep;
						
						accpoint = false;
						do {
							count = 0;
							for (i = 0; i < n; i++) {
								Bvec[i] = X[i] + steplength * t[i];
								#pragma GCC diagnostic push
								#pragma GCC diagnostic ignored "-Wfloat-equal"
								if (reltest + X[i] == reltest + Bvec[i]) { // == comparison is safe here, taken from robust R code and using the same algorithm
								#pragma GCC diagnostic pop
									count++;
								}
									
							}
							if (count < n) { /* point is different */
								f = fminfn(ex);
								funcount++;
								accpoint = (std::isfinite(f) &&
											f <= *Fmin + gradproj * steplength * acctol);
											//f <= *Fmin + steplength * acctol);
								
								if (!accpoint) {
									steplength *= stepredn;
									if (trace) Rcout << "*" << std::endl;
								} else {
									*Fmin = f; 
								} /* we improved, so update value */
							}
						} while (!(count == n || accpoint));
						if (count < n) {
							newstep = 2 * (f - *Fmin - gradproj * steplength);
							if (newstep > 0) {
								newstep = -(gradproj * steplength * steplength / newstep);
								for (i = 0; i < n; i++)
									Bvec[i] = X[i] + newstep * t[i];
								*Fmin = f;
								f = fminfn(ex);
								funcount++;
								if (f < *Fmin) {
									*Fmin = f;
									if (trace) Rcout << " i< " << std::endl;
								} else { /* reset Bvec to match lowest point */
									if (trace) Rcout << " i> " << std::endl;
									for (i = 0; i < n; i++)
										Bvec[i] = X[i] + steplength * t[i];
								}
							}
						}
					}
					oldstep = setstep * steplength;
					if (oldstep > 1.0)
						oldstep = 1.0;
				} while ((count != n) && (G1 > tol) && (cycle != cyclimit));
				
			} while ((cycle != 1) ||
					  ((count != n) && (G1 > tol) && *Fmin > abstol));
			
		}
		if (trace) {
			Rcout << "Exiting from conjugate gradients minimizer" << std::endl;
			Rcout << "	" << funcount << " function evaluations used" << std::endl;
			Rcout << "	" << gradcount << " gradient evaluations used" << std::endl;
		}
		*fncount = funcount;
		*grcount = gradcount;
	}
}
