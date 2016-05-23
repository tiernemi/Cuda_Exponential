/*
 * =====================================================================================
 *
 *       Filename:  integral.cu
 *
 *    Description:  Cuda source code for evaluating exponential integral.
 *
 *        Version:  1.0
 *        Created:  04/13/2016 03:37:44 PM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include "integral.hpp"

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cudaExponentialIntegral
 *    Arguments:  const int n 
 *    Arguments:  const int n - The order of the function E_n
 *                const Datatype & x - The argument x of E_n(x)
 *      Returns:  The value of the integral \int_1^\inf \frac{e^{-xt}}{t^n} dt 
 *  Description:  Evaluates the integral E_n using two different converging series
 *                depending on the magnitude of x.
 * =====================================================================================
 */

template <typename DataType>
DataType cudaExponentialIntegral(const int n, const DataType x) {
	return 0 ;
}

template float cudaExponentialIntegral<float>(const int n, const float x) ;
template double cudaExponentialIntegral<double>(const int n, const double x) ;

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cudaRunExponentials
 *    Arguments:  
 *      Returns:  
 *  Description:  
 * =====================================================================================
 */

void cudaRunExponentials(const int order, const int numberOfSamples, double & sampleRegionStart, double & sampleRegionEnd,
		              std::vector< std::vector<float> > & resultsFloatGpu, 
					  std::vector< std::vector<double> > & resultsDoubleGpu,
					  double & timeTotalGpuFloat, double & timeTotalGpuDouble) {
	float elapsedTime ;
	cudaEvent_t start, finish ;
	cudaEventCreate(&start) ;
	cudaEventCreate(&finish) ;

	double x,division=(sampleRegionEnd-sampleRegionStart)/((double)(numberOfSamples));
	// Float. //
	cudaEventRecord(start, 0) ;
	for (int ui=1;ui<=order;ui++) {
		for (int uj=1;uj<=numberOfSamples;uj++) {
			x = sampleRegionStart+uj*division;
			resultsFloatGpu[ui-1][uj-1] = cudaExponentialIntegral<float>(ui,x) ;
		}
	}
	cudaEventRecord(finish, 0) ;
	cudaEventSynchronize(finish) ;
	cudaEventElapsedTime(&elapsedTime, start, finish);
	timeTotalGpuFloat = elapsedTime ;

	// Double. //
	cudaEventRecord(start, 0) ;
	for (int ui=1;ui<=order;ui++) {
		for (int uj=1;uj<=numberOfSamples;uj++) {
			x = sampleRegionStart+uj*division;
			resultsDoubleGpu[ui-1][uj-1] = cudaExponentialIntegral<double>(ui,x) ;
		}
	}
	cudaEventSynchronize(finish) ;
	cudaEventElapsedTime(&elapsedTime, start, finish);
	timeTotalGpuDouble = elapsedTime ;

}		/* -----  end of function cudaRunExponentials  ----- */


