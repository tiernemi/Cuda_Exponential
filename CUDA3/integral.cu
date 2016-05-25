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
#include "stdio.h"
#include <math_constants.h>

#define WARPSIZE 32
#define MAXEVALS 1E7
#define FEPSILON 1.19209e-07
#define DEPSILON 2.22045e-16


// Helpful templated getters for GPU EPS and MAX vals.

template <typename data>
__inline__ __device__ data getEulerC() {
	return  0.5772157 ;
}

template <>
__inline__ __device__ float getEulerC<>() {
	return  0.5772157 ;
}

template <>
__inline__ __device__ double getEulerC<>() {
	return 0.5772156649015329 ;
}

template <typename data>
__inline__ __device__ data getMaxVal() {
	return CUDART_INF_F ;
}

template <>
__inline__ __device__ float getMaxVal<>() {
	return CUDART_INF_F ;
}

template <>
__inline__ __device__ double getMaxVal<>() {
	return CUDART_INF ;
}

template <typename data>
__inline__ __device__ data getEPS() {
	return FEPSILON ;
}

template <>
__inline__ __device__ float getEPS<>() {
	return FEPSILON ;
}

template <>
__inline__ __device__ double getEPS<>() {
	return DEPSILON ;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  integral
 *    Arguments:  int orderm1 - The order n minus 1.
 *                Datatype arg - The argument x in E(n,x)
 *      Returns:  The value of the integral \int_1^\inf \frac{e^{-xt}}{t^n} dt 
 *  Description:  Evaluates the integral E_n using a continued fraction series for x > 1.
 * =====================================================================================
 */

template <typename DataType>
__inline__ __device__ DataType evalExpIntegralGt1(int orderm1, DataType arg) {
	DataType del ;
	DataType a = 0 ;
	DataType b = arg+orderm1+1 ;
	DataType c = getMaxVal<DataType>() ;
	DataType d = 1.0/b ;
	DataType h = d ;
	DataType eps = getEPS<DataType>() ;
	for (int i = 1 ; i <= MAXEVALS ; i++) {
		a = -i*(orderm1+i) ;
		b += 2.0 ;
		d = 1.0/(a*d+b) ;
		c = b+a/c ;
		del = c*d ;
		h *= del ;
		if (fabs(del-1.0) <= eps) {
			return h*exp(-arg) ;
		}
	}
	return 0 ;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  integral
 *    Arguments:  int orderm1 - The order n minus 1.
 *                Datatype arg - The argument x in E(n,x)
 *      Returns:  The value of the integral \int_1^\inf \frac{e^{-xt}}{t^n} dt 
 *  Description:  Evaluates the integral E_n using a converging series for x < 1.
 * =====================================================================================
 */

template <typename DataType>
__inline__ __device__ DataType evalExpIntegralLt1(int orderm1, DataType arg) {
	DataType ans = (orderm1 !=0 ? 1.0/orderm1 : -log(arg)-getEulerC<DataType>()) ;
	DataType fact = 1.0 ;
	DataType del = 0.0 ;
	DataType psi = 0.0 ;
	DataType eps = getEPS<DataType>() ;
	DataType meuler = -getEulerC<DataType>() ;
	for (DataType i = 1 ; i <= MAXEVALS ; i++) {
		fact *= -arg/i ;
		if (i != orderm1) {
			del = -fact/(i-orderm1) ;
		} else {
			psi = meuler ;
			for (DataType ii = 1; ii <= orderm1 ; ii++) {
				psi += 1.0/ii ;
			}
			del = fact*(-log(arg)+psi) ;
		}
		ans += del ;
		if (fabs(del) < fabs(ans)*eps) {
			return ans ;
		}
	}
	return 0 ;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  evalSamples
 *    Arguments:  int numOrders - The maximum number of orders to evaluate.
 *                int numberOfSamples - The number of samples to take.
 *                Datatype sampleRegionStart - The start or the region (a,b)
 *                Datatype division - Distance bewteen evaluations in (a,b)
 *                Datatype * gpuData - Location of data on GPU.
 *  Description:  Evaluates E_(n,x) over domain n element of (1,n) and x element of
 *                (a,b) where there are numSamples evaluations of x.
 * =====================================================================================
 */

template <typename DataType>
__global__ void evalSamples(int numOrders, int numberOfSamples, DataType sampleRegionStart, DataType division, DataType * gpuData) {
	int globalIDx = threadIdx.x + blockIdx.x*blockDim.x ;
	int globalIDy = threadIdx.y + blockIdx.y*blockDim.y ;
	if (globalIDx < numberOfSamples && globalIDy < numOrders) {
		DataType x = sampleRegionStart+(globalIDy+1)*division ;
		if (x > 1) {
			gpuData[globalIDx*numberOfSamples+globalIDy] = evalExpIntegralGt1(globalIDx,x) ;
		} else {
			gpuData[globalIDx*numberOfSamples+globalIDy] = evalExpIntegralLt1(globalIDx,x) ;
		}
	}
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cudaRunExponentials
 *    Arguments:  int order - The maximum order being evaluated.
 *                int numberOfSamples - The number of samples to take in domain (a,b)
 *                double & sampleRegionStart - The start of the domain (a,b)
 *                double & sampleRegionEnd - The end of the interval (a,b)
 *                float * resultsFloatGpu - The results for the GPU evaluations.
 *                double * resultsDoubleGpu - The results for the GP evaluations.
 *                double & timeTotalGpuFloat - Time taken to evaluate floats on GPU.
 *                double & timeTotalGpuDouble - Time taken to evaluate doubles on GPU.
 *                int blockSizeOr - The block size associated with orders.
 *                int blockSizeSm - The block size associated with samples.
 *                double & transferTimeFloat - Time taken to transfer data from GPU to DRAM.
 *                double & transferTimeDouble - Time taken to transfer data from GPU to DRAM.
 *               
 *  Description:  Evaluates the exponential integral between (a,b) for a number of
 *                orders and samples.
 *
 * =====================================================================================
 */

void cudaRunExponentials(int order, int numberOfSamples, double & sampleRegionStart, double & sampleRegionEnd,
						float * resultsFloatGpu, double * resultsDoubleGpu, double & timeTotalGpuFloat, double & timeTotalGpuDouble, 
						int blockSizeOr, int blockSizeSm, double & transferTimeFloat, double & transferTimeDouble) {

	int numResults = numberOfSamples*order ;
	dim3 dim3BlockOuter(blockSizeOr,blockSizeSm) ;
	dim3 dim3GridOuter((order/dim3BlockOuter.x) + (!(order%dim3BlockOuter.x)?0:1) , 
			(numberOfSamples/dim3BlockOuter.x) + (!(numberOfSamples%dim3BlockOuter.x)?0:1));
	float elapsedTime ;
	cudaEvent_t start, finish ;
	cudaEvent_t transStart, transFinish ;
	cudaEventCreate(&start) ;
	cudaEventCreate(&finish) ;
	cudaEventCreate(&transStart) ;
	cudaEventCreate(&transFinish) ;
    double division=(sampleRegionEnd-sampleRegionStart)/((double)(numberOfSamples));
	
	// Float. //
	cudaEventRecord(start, 0) ;

	// Eval. //
	float * gpuFloatData ;
	cudaMalloc((void**) &gpuFloatData, sizeof(float)*numResults) ;
	evalSamples<<<dim3GridOuter,dim3BlockOuter>>>(order, numberOfSamples, float(sampleRegionStart), float(division), gpuFloatData) ;

	// Write Back. //
	cudaEventRecord(transStart,0) ;
	cudaMemcpy(resultsFloatGpu,gpuFloatData,sizeof(float)*numResults, cudaMemcpyDeviceToHost) ;
	cudaEventRecord(transFinish,0) ;
	cudaEventSynchronize(transFinish) ;
	cudaEventElapsedTime(&elapsedTime, transStart, transFinish);
	transferTimeFloat = elapsedTime/1E3 ;

	cudaFree(gpuFloatData) ;

	cudaEventRecord(finish, 0) ;
	cudaEventSynchronize(finish) ;
	cudaEventElapsedTime(&elapsedTime, start, finish);
	timeTotalGpuFloat = elapsedTime/1E3 ;

	// Double. //
	cudaEventRecord(start, 0) ;

	// Eval. //
	double * gpuDoubleData ;
	cudaMalloc((void**) &gpuDoubleData, sizeof(double)*numResults) ;
	evalSamples<<<dim3GridOuter,dim3BlockOuter>>>(order, numberOfSamples, sampleRegionStart, division, gpuDoubleData) ;

	// Write Back. //
	cudaEventRecord(transStart,0) ;
	cudaMemcpy(resultsDoubleGpu,gpuDoubleData,sizeof(double)*numResults, cudaMemcpyDeviceToHost) ;
	cudaEventRecord(transFinish,0) ;
	cudaEventSynchronize(transFinish) ;
	cudaEventElapsedTime(&elapsedTime, transStart, transFinish);
	transferTimeDouble = elapsedTime/1E3 ;

	cudaFree(gpuDoubleData) ;

	cudaEventRecord(finish, 0) ;
	cudaEventSynchronize(finish) ;
	cudaEventElapsedTime(&elapsedTime, start, finish);
	printf("%f\n", elapsedTime);
	timeTotalGpuDouble = elapsedTime/1E3 ;
}		/* -----  end of function cudaRunExponentials  ----- */


