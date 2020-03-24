#pragma once

#include <iostream>		// For console i/o
#include <vector>		// For vectors, not including linear algebra
#include <random>		// For random number generation
#include <algorithm>	// For min/max functions

// Additional libraries that you will have to download.
// First is Eigen, which we use for linear algebra: http://eigen.tuxfamily.org/index.php?title=Main_Page
#include <Eigen/Dense>
// Second is Boost, which we use for ibeta_inv: https://www.boost.org/
#include <boost/math/special_functions/beta.hpp>

// Typically these shouldn't be in a .hpp file.
using namespace std;
using namespace Eigen;
using namespace boost::math;

// This function returns the inverse of Student's t CDF using the degrees of
// freedom in nu for the corresponding probabilities in p. That is, it is
// a C++ implementation of Matlab's tinv function: https://www.mathworks.com/help/stats/tinv.html
// To see how this was created, see the "quantile" block here: https://www.boost.org/doc/libs/1_58_0/libs/math/doc/html/math_toolkit/dist_ref/dists/students_t_dist.html
double tinv(double p, unsigned int nu) {
	double x = ibeta_inv((double)nu / 2.0, 0.5, 2.0 * min(p, 1.0 - p));
	return sign(p - 0.5) * sqrt((double)nu * (1.0 - x) / x);
}

// Get the sample standard deviation of the vector v (an Eigen::VectorXd)
double stddev(const VectorXd& v) {
	double mu = v.mean(), result = 0;			// Get the sample mean
	for (unsigned int i = 0; i < v.size(); i++)
		result += (v[i] - mu) * (v[i] - mu);	// Compute the sum of the squared differences between samples and the sample mean
	result = sqrt(result / (v.size() - 1.0));	// Get the sample variance by dividing by the number of samples minus one, and then the sample standard deviation by taking the square root.
	return result;								// Return the value that we've computed.
}

// Assuming v holds i.i.d. samples of a random variable, compute
// a (1-delta)-confidence upper bound on the expected value of the random
// variable using Student's t-test. That is:
// sampleMean + sampleStandardDeviation/sqrt(n) * tinv(1-delta, n-1),
// where n is the number of points in v.
//
// If numPoints is provided, then ttestUpperBound predicts what its output would be if it were to
// be run using a new vector, v, containing numPoints values sampled from the same distribution as
// the points in v.
double ttestUpperBound(const VectorXd& v, const double& delta, const int numPoints = -1) {
	unsigned int n = (numPoints == -1 ? (unsigned int)v.size() : (unsigned int)numPoints);	// Set n to be numPoints if it was provided, and the number of points in v otherwise.
	return v.mean() + (numPoints != -1 ? 2.0 : 1.0) * stddev(v) / sqrt((double)n) * tinv(1.0 - delta, n - 1u);
}

/*
This function implements CMA-ES (http://en.wikipedia.org/wiki/CMA-ES). Return
value is the minimizer / maximizer. This code is written for brevity, not clarity.
See the link above for a description of what this code is doing.
*/
VectorXd CMAES(
	const VectorXd& initialMean,											// Starting point of the search
	const double& initialSigma,												// Initial standard deviation of the search around initialMean
	const unsigned int& numIterations,										// Number of iterations to run before stopping
	// f, below, is the function to be optimized. Its first argument is the solution, the middle arguments are variables required by f (listed below), and the last is a random number generator.
	double(*f)(const VectorXd& theta, const void* params[], mt19937_64& generator),
	const void* params[],													// Parrameters of f other than theta
	const bool& minimize,													// If true, we will try to minimize f. Otherwise we will try to maximize f
	mt19937_64& generator)													// The random number generator to use
{
	// Define all of the terms that we will use in the iterations
	unsigned int N = (unsigned int)initialMean.size(), lambda = 4 + (unsigned int)floor(3.0 * log(N)), hsig;
	double sigma = initialSigma, mu = lambda / 2.0, eigeneval = 0, chiN = pow(N, 0.5) * (1.0 - 1.0 / (4.0 * N) + 1.0 / (21.0 * N * N));
	VectorXd xmean = initialMean, weights((unsigned int)mu);
	for (unsigned int i = 0; i < (unsigned int)mu; i++)
		weights[i] = i + 1;
	weights = log(mu + 1.0 / 2.0) - weights.array().log();
	mu = floor(mu);
	weights = weights / weights.sum();
	double mueff = weights.sum() * weights.sum() / weights.dot(weights), cc = (4.0 + mueff / N) / (N + 4.0 + 2.0 * mueff / N), cs = (mueff + 2.0) / (N + mueff + 5.0), c1 = 2.0 / ((N + 1.3) * (N + 1.3) + mueff), cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((N + 2.0) * (N + 2.0) + mueff)), damps = 1.0 + 2.0 * max(0.0, sqrt((mueff - 1.0) / (N + 1.0)) - 1.0) + cs;
	VectorXd pc = VectorXd::Zero(N), ps = VectorXd::Zero(N), D = VectorXd::Ones(N), DSquared = D, DInv = 1.0 / D.array(), xold, oneOverD;
	for (unsigned int i = 0; i < DSquared.size(); i++)
		DSquared[i] *= DSquared[i];
	MatrixXd B = MatrixXd::Identity(N, N), C = B * DSquared.asDiagonal() * B.transpose(), invsqrtC = B * DInv.asDiagonal() * B.transpose(), arx(N, (int)lambda), repmat(xmean.size(), (int)(mu + .1)), artmp, arxSubMatrix(N, (int)(mu + .1));
	vector<double> arfitness(lambda);
	vector<unsigned int> arindex(lambda);
	// Perform the iterations
	for (unsigned int counteval = 0; counteval < numIterations;) {
		// Sample the population
		for (unsigned int k = 0; k < lambda; k++) {
			normal_distribution<double> distribution(0, 1);
			VectorXd randomVector(N);
			for (unsigned int i = 0; i < N; i++)
				randomVector[i] = D[i] * distribution(generator);
			arx.col(k) = xmean + sigma * B * randomVector;
		}
		// Evaluate the population
		vector<VectorXd> fInputs(lambda);
		for (unsigned int i = 0; i < lambda; i++) {
			fInputs[i] = arx.col(i);
			arfitness[i] = (minimize ? 1 : -1) * f(fInputs[i], params, generator);
		}
		// Update the population distribution
		counteval += lambda;
		xold = xmean;
		for (unsigned int i = 0; i < lambda; ++i)
			arindex[i] = i;
		std::sort(arindex.begin(), arindex.end(), [&arfitness](unsigned int i1, unsigned int i2) {return arfitness[i1] < arfitness[i2]; });
		for (unsigned int col = 0; col < (unsigned int)mu; col++)
			arxSubMatrix.col(col) = arx.col(arindex[col]);
		xmean = arxSubMatrix * weights;
		ps = (1.0 - cs) * ps + sqrt(cs * (2.0 - cs) * mueff) * invsqrtC * (xmean - xold) / sigma;
		hsig = (ps.norm() / sqrt(1.0 - pow(1.0 - cs, 2.0 * counteval / lambda)) / (double)chiN < 1.4 + 2.0 / (N + 1.0) ? 1 : 0);
		pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma;
		for (unsigned int i = 0; i < repmat.cols(); i++)
			repmat.col(i) = xold;
		artmp = (1.0 / sigma) * (arxSubMatrix - repmat);
		C = (1 - c1 - cmu) * C + c1 * (pc * pc.transpose() + (1u - hsig) * cc * (2 - cc) * C) + cmu * artmp * weights.asDiagonal() * artmp.transpose();
		sigma = sigma * exp((cs / damps) * (ps.norm() / (double)chiN - 1.0));
		if ((double)counteval - eigeneval > (double)lambda / (c1 + cmu) / (double)N / 10.0) {
			eigeneval = counteval;
			for (unsigned int r = 0; r < C.rows(); r++)
				for (unsigned int c = r + 1; c < C.cols(); c++)
					C(r, c) = C(c, r);
			EigenSolver<MatrixXd> es(C);
			D = C.eigenvalues().real();
			B = es.eigenvectors().real();
			D = D.array().sqrt();
			for (unsigned int i = 0; i < B.cols(); i++)
				B.col(i) = B.col(i).normalized();
			oneOverD = 1.0 / D.array();
			invsqrtC = B * oneOverD.asDiagonal() * B.transpose();
		}
	} // End loop over iterations
	return arx.col(arindex[0]);
}