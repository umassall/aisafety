// Standard includes that should come with your compiler
#include <iostream>		// For console i/o
#include <vector>		// For vectors, not including linear algebra
#include <random>		// For random number generation

// Additional library that you should have downloaded.
#include <Eigen/Dense>

#include "HelperFunctions.hpp"					// General functions not specific to our Seldonian algorithm implementation

using namespace std;	// To avoid writing std::vector all the time
using namespace Eigen;	// To avoid writing Eigen::VectorXd all the time

// We will store individual data points in this object
struct Point {
	double x;									// Input value
	double y;									// Output value
};

// Generate numPoints data points. Here generator is the random number generator to use
vector<Point> generateData(int numPoints, mt19937_64& generator)
{
	vector<Point> result(numPoints);			// Create the vector of data points that we will return, of length numPoints
	normal_distribution<double> d(0, 1);		// Create a standard normal distribution (mean 0, variance 1)
	for (Point& p : result)	{					// Loop over each point p in result
		p.x = d(generator);						// Sample x from a standard normal distribution
		p.y = p.x + d(generator);				// Set y to be x, plus noise from a standard normal distribution
	}
	return result;
}

// Using the weights in theta, predict the y-value associated with the provided x.
// This function assumes we are performing linear regression, so that theta has
// two elements, the slope (first parameter) and y-intercept (second parameter)
double predict(const VectorXd & theta, const double & x) {
	return theta[0] * x + theta[1];
}

// Estimator of the primary objective, in this case, the negative sample mean squared error
double fHat(const VectorXd& theta, const vector<Point> & Data) {
	double result = 0, prediction;				// We will store the sample MSE in result. Prediction will store the prediction for each point in the data set
	for (const Point& p : Data) {				// Loop over points p in the data set (Data)
		prediction = predict(theta, p.x);		// Get the prediction using theta
		result += (prediction - p.y) * (prediction - p.y);	// Add the squared error to result
	}
	result /= (double)Data.size();				// We want the sample mean squared error, not the sum of squared errors, so divide by the number of samples
	return -result;								// Return the value that we have computed
}

// Returns unbiased estimates of g_1(theta), computed using the provided data
VectorXd gHat1(const VectorXd& theta, const vector<Point> & Data) {
	VectorXd result(Data.size());						// We will get one estimate per data point, so initialize the result to have length equal to the number of data points
	for (unsigned int i = 0; i < Data.size(); i++) {	// Loop over the data points
		double prediction = predict(theta, Data[i].x);	// Compute the prediction for the i'th data point
		result[i] = (Data[i].y - prediction) * (Data[i].y - prediction);	// Compute the squared error for the i'th data point, and store in the i'th element of result.
	}
	result.array() -= 2.0;								// We want the MSE to be less than 2.0, so g(theta) = MSE-2.0.
	return result;										// Return the result that we have computed
}

// Returns unbiased estimates of g_2(theta), computed using the provided data
VectorXd gHat2(const VectorXd& theta, const vector<Point>& Data) {
	VectorXd result(Data.size());						// We will get one estimate per data point, so initialize the result to have length equal to the number of data points
	for (unsigned int i = 0; i < Data.size(); i++) {	// Loop over the data points
		double prediction = predict(theta, Data[i].x);	// Compute the prediction for the i'th data point
		result[i] = (Data[i].y - prediction) * (Data[i].y - prediction);	// Compute the squared error for the i'th data point, and store in the i'th element of result.
	}
	result.array() = 1.25 - result.array();				// We want the MSE to be at least 1.25, so g(theta) = 1.25-MSE.
	return result;										// Return the result that we have computed
}

// Run ordinary least squares linear regression
VectorXd leastSquares(const vector<Point>& Data) {
	// Put data into an input matrix X (Data.size() rows, 2 cols), and vector y (of Data.size()).
	MatrixXd X = MatrixXd::Ones(Data.size(), 2);							// Initialize X to be a matrix of Data.size() rows and 2 cols, filled with ones.
	VectorXd y(Data.size());												// Initialize y to be a vector of length Data.size()
	for (unsigned int i = 0; i < Data.size(); i++) {						// Loop over data points
		X(i, 0) = Data[i].x;												// Copy the x-value over the entry in the first colum of the i'th row.
		y[i] = Data[i].y;													// Copy the target value into the y-vector
	}
	return X.jacobiSvd(ComputeThinU | ComputeThinV).solve(y);				// Return the least squares solution using Eigen's Jacobi SVD function.
}

// Run the safety test on the solution theta. Returns true if the test is passed
bool safetyTest(
	const VectorXd& theta,													// The solution to test
	const vector<Point>& Data,												// The data to use in the safety test
	const vector<VectorXd(*)(const VectorXd&, const vector<Point>&)>& gHats,// Unbiased estimators of g(theta) for each of the behavioral constraints
	const vector<double>& deltas)											// Confidence levels for the behavioral constraints	
{
	for (unsigned int i = 0; i < gHats.size(); i++) {						// Loop over behavioral constraints, checking each
		if (ttestUpperBound(gHats[i](theta, Data), deltas[i]) > 0)			// Check if the i'th behavioral constraint is satisfied
			return false;													// It wasn't - the safety test failed.
	}
	return true;															// If we get here, all of the behavioral constraints were satisfied
}

// The objective function maximized by getCandidateSolution.
double candidateObjective(
	const VectorXd& theta,						// The solution to evaluate
	const void * params[],
	mt19937_64& generator)						// The random number generator to use
{
	// Unpack the variables in params into their different types. See how they were packed in getCandidateSolution
	const vector<Point>* p_Data = (const vector<Point>*)params[0];
	const vector<VectorXd(*)(const VectorXd&, const vector<Point>&)>* p_gHats = (const vector<VectorXd(*)(const VectorXd&, const vector<Point>&)>*)params[1];
	const vector<double>* p_deltas = (const vector<double>*)params[2];
	const unsigned int safetyDataSize = *((const unsigned int*)params[3]);

	double result = fHat(theta, *p_Data);		// Get the primary objective
	bool predictSafetyTest = true;				// Prediction of what the safety test will say - initialized to "true" = pass.
	for (unsigned int i = 0; i < p_gHats->size(); i++) {	// Loop over the behavioral constraints
		double ub = ttestUpperBound((*p_gHats)[i](theta, *p_Data), (*p_deltas)[i], safetyDataSize);
		if (ub > 0) {							// We don't think the i'th behavioral constraint will pass the safety test if we return theta as the candidate solution
			if (predictSafetyTest) {
				predictSafetyTest = false;		// We don't think the safety test will pass
				result = -100000;				// Put a barrier in the objective - any solution that we think will pass all tests should have a value greater than what we return for this solution. Also, remove any shaping due to the primary objective so that we focus on the behavioral constraint.
			}
			result -= ub;						// Add a shaping to the objective function that will push the search toward solutions that will pass the prediction of the safety test.
		}
	}
	return result;
}

// Use the provided data to get a solution expected to pass the safety test
VectorXd getCandidateSolution(const vector<Point> & Data, const vector<VectorXd(*)(const VectorXd&, const vector<Point>&)>& gHats, const vector<double>& deltas, const unsigned int& safetyDataSize, mt19937_64 & generator) {
	VectorXd initialSolution = leastSquares(Data);							// Where should the search start? Let's just use the linear fit that we would get from ordinary least squares linear regression
	double initialSigma = 2.0*(initialSolution.dot(initialSolution) + 1.0);	// A heuristic to select the width of the search based on the weight magnitudes we expect to see.
	int numIterations = 100;												// Larger is better, but takes longer.
	bool minimize = false;													// We want to maximize the candidate objective.
	// Pack parameters of candidate objective into params. In candidateObjective we need to unpack in the same order.
	const void* params[4];
	params[0] = &Data;
	params[1] = &gHats;
	params[2] = &deltas;
	params[3] = &safetyDataSize;
	// Use CMA-ES to get a solution that approximately maximizes candidateObjective
	return CMAES(initialSolution, initialSigma, numIterations, candidateObjective, params, minimize, generator);
}

// Our quasi-Seldonian linear regression algorithm. The result is a pair of items, the second
// is a Boolean denoting whether a solution is being returned. If it is false, it indicates
// No Solution Found (NSF), and the first element is irrelevant. If the second element is
// true, then the first element is the solution that was found.
pair<VectorXd,bool> QSA(
	const vector<Point>& Data,												// The training data to use
	const vector<VectorXd(*)(const VectorXd&, const vector<Point>&)>& gHats,// Unbiased estimators of g(theta) for each of the behavioral constraints
	const vector<double>& deltas,											// Confidence levels for the behavioral constraints
	mt19937_64& generator)													// The random number generator to use
{
	std::vector<Point> candData(Data.begin(), Data.begin() + (int)(Data.size() * 0.4));			// Put 40% of the data in candidateData
	std::vector<Point> safetyData(Data.begin() + candData.size(), Data.end());					// Put the rest of the data in safetyData
	pair<VectorXd, bool> result;																// Create the object that we will return
	result.first = getCandidateSolution(candData, gHats, deltas, (unsigned int)safetyData.size(), generator);	// Get the candidate solution
	result.second = safetyTest(result.first, safetyData, gHats, deltas);	// Run the safety test
	return result;															// Return the result object
}

// Entry point for the program
int main(int argc, char* argv[]) {
	mt19937_64 generator(0);								// Create the random number generator to use, with seed zero
	unsigned int numPoints = 5000;							// Let's use 5000 points
	vector<Point> Data = generateData(numPoints, generator);// Generate the data

	// Create the behavioral constraints - each is a gHat function and a confidence level delta. Put these in vector objects, one element per constraint
	vector<VectorXd(*)(const VectorXd&, const vector<Point>&)> gHats(2);	// The array of gHat functions to use
	gHats[0] = gHat1;														// This gHat requires the MSE to be less than 2.0
	gHats[1] = gHat2;														// This gHat requires the MSE to be at least 1.25
	vector<double> deltas(2, 0.1);											// The array of confidence levels, delta, to use. Initialize with two values, both equal to 0.1.

	pair<VectorXd, bool> result = QSA(Data, gHats, deltas, generator);		// Run the Seldonian algorithm.
	if (result.second) cout << "A solution was found: " << result.first.transpose() << endl;
	else cout << "No solution found." << endl;
}