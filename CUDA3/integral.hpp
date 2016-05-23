#ifndef INTEGRAL_HPP_O4WBDGSQ
#define INTEGRAL_HPP_O4WBDGSQ

/*
 * =====================================================================================
 *
 *       Filename:  integral.hpp
 *
 *    Description:  Header file for linking cuda code with c++ code.
 *
 *        Version:  1.0
 *        Created:  04/13/2016 03:37:44 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <vector>

void cudaRunExponentials(const int order, const int numberOfSamples, double sampleRegionStart, double sampleRegionEnd,
		              std::vector< std::vector<float> > & resultsFloatGpu, 
					  std::vector< std::vector<double> > & resultsDoubleGpu,
					  double & timeTotalGpuFloat, double & timeTotalGpuDouble ) ;

#endif /* end of include guard: INTEGRAL_HPP_O4WBDGSQ */
