///// Created by Jose Mauricio Refojo - 2014-04-02		Last changed: 2014-04-02
//------------------------------------------------------------------------------
// File : main.cpp
//------------------------------------------------------------------------------

#include <iostream>
#include <limits>       // std::numeric_limits
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <time.h>
#include "getopt.h"
#include "integral.hpp"

using namespace std;

template <class DataType>
DataType exponentialIntegral(const int n, const DataType & x) ;
void outputResultsCpu(const std::vector<std::vector<float>> &resultsFloatCpu,const std::vector<std::vector<double>> &resultsDoubleCpu) ;
void outputResultsGpu(const float * resultsFloatGpu, const double * resultsDoubleGpu) ;
int	parseArguments(int argc, char **argv) ;
void printUsage(void) ;
template <typename DataType>
void checkDeviation(std::vector<std::vector<DataType>> & resultsCpu, DataType * resultsGpu) ;

template <typename data>
data testFunc() {
	return 0 ;
}

template <>
float testFunc() {
	return 0 ;
}


bool verbose,timing,cpu,gpu ;
unsigned int order,numberOfSamples,blockSizeOr,blockSizeSm ;
double sampleRegionStart,sampleRegionEnd;	// The interval that we are going to use

int main(int argc, char *argv[]) {
	unsigned int ui,uj;
	cpu=true;
	gpu=true;
	verbose=false;
	timing=false;
	// n is the maximum order of the exponential integral that we are going to test
	// numberOfSamples is the number of samples in the interval [sampleRegionStart,sampleRegionEnd] that we are going to calculate
	order=10 ; // default
	numberOfSamples=10 ; // default
	sampleRegionStart=0.0 ; // default
	sampleRegionEnd=10.0 ; // default
	blockSizeOr = 32 ;
	blockSizeSm = 32 ;

	struct timeval expoStart, expoEnd;

	parseArguments(argc, argv);

	if (verbose) {
		cout << "n=" << order << endl;
		cout << "numberOfSamples=" << numberOfSamples << endl;
		cout << "a=" << sampleRegionStart << endl;
		cout << "b=" << sampleRegionEnd << endl;
		cout << "timing=" << timing << endl;
		cout << "verbose=" << verbose << endl;
	}

	// Sanity checks
	if (sampleRegionStart>=sampleRegionEnd) {
		cout << "Incorrect interval ("<< sampleRegionStart <<","<< sampleRegionEnd<<") has been stated!" << endl;
		return 0;
	}
	if (order<=0) {
		cout << "Incorrect orders ("<< order <<") have been stated!" << endl;
		return 0;
	}
	if (numberOfSamples<=0) {
		cout << "Incorrect number of samples ("<<numberOfSamples<<") have been stated!" << endl;
		return 0;
	}

	std::vector<std::vector<float>> resultsFloatCpu;
	std::vector<std::vector<double>> resultsDoubleCpu;
	float * resultsFloatGpu;
	double * resultsDoubleGpu;

	double timeTotalCpuFloat=0.0;
	double timeTotalGpuFloat=0.0;
	double timeTotalCpuDouble=0.0;
	double timeTotalGpuDouble=0.0;
	double timeTransferFloat=0.0;
	double timeTransferDouble=0.0;

	try {
		resultsFloatCpu.resize(order,vector<float>(numberOfSamples));
	} catch (std::bad_alloc const&) {
		cout << "resultsFloatCpu memory allocation fail!" << endl;	exit(1);
	}
	try {
		resultsDoubleCpu.resize(order,vector<double>(numberOfSamples));
	} catch (std::bad_alloc const&) {
		cout << "resultsDoubleCpu memory allocation fail!" << endl;	exit(1);
	}
	resultsDoubleGpu = (double*) malloc(sizeof(double)*numberOfSamples*order) ;
	resultsFloatGpu = (float*) malloc(sizeof(float)*numberOfSamples*order) ;

	double x,division=(sampleRegionEnd-sampleRegionStart)/((double)(numberOfSamples));

	if (cpu) {
		gettimeofday(&expoStart, NULL);
		for (ui=1;ui<=order;ui++) {
			for (uj=1;uj<=numberOfSamples;uj++) {
				x=sampleRegionStart+uj*division;
				resultsFloatCpu[ui-1][uj-1]=exponentialIntegral<float>(ui,x);
			}
		}
		gettimeofday(&expoEnd, NULL);
		timeTotalCpuFloat=((expoEnd.tv_sec + expoEnd.tv_usec*0.000001) - (expoStart.tv_sec + expoStart.tv_usec*0.000001));
		gettimeofday(&expoStart, NULL);
		for (ui=1;ui<=order;ui++) {
			for (uj=1;uj<=numberOfSamples;uj++) {
				x=sampleRegionStart+uj*division;
				resultsDoubleCpu[ui-1][uj-1]=exponentialIntegral<double>(ui,x);
			}
		}
		gettimeofday(&expoEnd, NULL);
		timeTotalCpuDouble=((expoEnd.tv_sec + expoEnd.tv_usec*0.000001) - (expoStart.tv_sec + expoStart.tv_usec*0.000001));

	}
	if (gpu) {
		cudaRunExponentials(order,numberOfSamples,sampleRegionStart,sampleRegionEnd,resultsFloatGpu,
				resultsDoubleGpu,timeTotalGpuFloat,timeTotalGpuDouble, blockSizeOr, blockSizeSm,
				timeTransferFloat, timeTransferDouble) ;
	}

	if (timing) {
		if (cpu) {
			printf ("calculating the exponentials on the cpu took: %f seconds for floats\n",timeTotalCpuFloat);
			printf ("calculating the exponentials on the cpu took: %f seconds for doubles\n",timeTotalCpuDouble);
		}
		if (gpu) {
			printf ("calculating the exponentials on the gpu took: %f seconds for floats\n",timeTotalGpuFloat);
			printf ("calculating the exponentials on the gpu took: %f seconds for doubles\n",timeTotalGpuDouble);
		}
		if (gpu && cpu) {
			printf ("speed-up for cpu Vs gpu is %f for floats\n", timeTotalCpuFloat / timeTotalGpuFloat );
			printf ("speed-up for cpu Vs gpu is %f for doubles\n", timeTotalCpuDouble / timeTotalGpuDouble );
		}
		if (gpu && cpu) {
			printf (" %% transfer/total %f for floats\n", timeTransferFloat/timeTotalGpuFloat );
			printf (" %% transfer/total %f for doubles\n",  timeTransferDouble/timeTotalGpuDouble );
		}
	}

	if (verbose) {
		if (cpu) {
			outputResultsCpu (resultsFloatCpu,resultsDoubleCpu) ;
		}
		if (gpu) {
			outputResultsGpu (resultsFloatGpu,resultsDoubleGpu) ;
		}
	}

	if (cpu && gpu) {
		printf("FLOAT DEVIATIONS >>> \n");
		checkDeviation(resultsFloatCpu,resultsFloatGpu) ;
		printf("DOUBLE DEVIATIONS >>> \n");
		checkDeviation(resultsDoubleCpu,resultsDoubleGpu) ;
	}
	return 0;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  outputResultsCpu
 *    Arguments:  const std::vector<std::vector<float>> & resultsFloatCpu - Single 
 *                   precision results from cpu evaluation of integral.
 *                const std::vector<std::vector<float>> & resultsFloatCpu - Double
 *                   precision results from cpu evaluation of integral.
 *  Description:  Outputs results from cpu evaluation of integrals.
 * =====================================================================================
 */

void outputResultsCpu(const std::vector<std::vector<float>> &resultsFloatCpu, const std::vector<std::vector<double>> &resultsDoubleCpu) {
	double x = 0 ;
	double division=(sampleRegionEnd-sampleRegionStart)/((double)(numberOfSamples)); // delta x
	for (unsigned ui=1;ui<=order;ui++) {
		for (unsigned uj=1;uj<=numberOfSamples;uj++) {
			x=sampleRegionStart+uj*division ;
			std::cout << "CPU==> exponentialIntegralDouble (" << ui << "," << x <<")=" << resultsDoubleCpu[ui-1][uj-1] << " ,";
			std::cout << "exponentialIntegralFloat  (" << ui << "," << x <<")=" << resultsFloatCpu[ui-1][uj-1] << endl;
		}
	}
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  outputResultsGpu
 *    Arguments:  const std::vector<std::vector<float>> & resultsFloatGpu - Single 
 *                   precision results from gpu evaluation of integral.
 *                const std::vector<std::vector<float>> & resultsFloatGpu - Double
 *                   precision results from gpu evaluation of integral.
 *  Description:  Outputs results from gpu evaluation of integrals.
 * =====================================================================================
 */

void outputResultsGpu(const float * resultsFloatGpu, const double * resultsDoubleGpu) {
	double x = 0 ;
	double division=(sampleRegionEnd-sampleRegionStart)/((double)(numberOfSamples)); // delta x
	for (unsigned ui=0;ui<order;ui++) {
		for (unsigned uj=0;uj<numberOfSamples;uj++) {
			x=sampleRegionStart+(uj+1)*division ;
			std::cout << "GPU ==> exponentialIntegralDouble (" << ui+1 << "," << x <<")=" << resultsDoubleGpu[uj+ui*numberOfSamples] << " ,";
			std::cout << "exponentialIntegralFloat  (" << ui+1 << "," << x <<")=" << resultsFloatGpu[uj+ui*numberOfSamples] << endl;
		}
	}
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  exponentialIntegral 
 *    Arguments:  const int n - The order of the function E_n
 *                const Datatype & x - The argument x of E_n(x)
 *      Returns:  The value of the integral \int_1^\inf \frac{e^{-xt}}{t^n} dt 
 *  Description:  Evaluates the integral E_n using two different converging series
 *                depending on the magnitude of x.
 * =====================================================================================
 */

template <class DataType>
DataType exponentialIntegral (const int n, const DataType & x) {
	static const int maxIterations=2000000000 ; // Maximum number of iters in sequence.
	static const DataType eulerConstant=0.5772156649015329 ;
	DataType epsilon = std::numeric_limits<DataType>::epsilon() ;
	DataType maxTypeVal = std::numeric_limits<DataType>::max() ;
	int i,ii,nm1=n-1 ;
	DataType a,b,c,d,del,fact,h,psi,ans=0.0 ;

	if (n<0.0 || x<0.0 || (x==0.0 && ((n==0) || (n==1)))) {
		cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
		exit(1);
	}
	// Order 0 just results in exp(-x)/x
	if (n==0) {
		ans=exp(-x)/x;
	} 
	// Depending on x different series used.
	else {
		// This series converges faster for x > 1. //
		if (x>1.0) {
			b=x+n;
			c=maxTypeVal;
			d=1.0/b;
			h=d;
			for (i=1;i<=maxIterations;i++) {
				a=-i*(nm1+i);
				b+=2.0;
				d=1.0/(a*d+b);
				c=b+a/c;
				del=c*d;
				h*=del;
				if (fabs(del-1.0)<=epsilon) {
					ans=h*exp(-x);
					return ans;
				}
			}
			return ans;
		} 
		// This series converges faster for x < 1. //
		else {
			ans=(nm1!=0 ? 1.0/nm1 : -log(x)-eulerConstant);	// First term
			fact=1.0;
			for (i=1;i<=maxIterations;i++) {
				fact*=-x/i;
				if (i != nm1) {
					del = -fact/(i-nm1);
				} else {
					psi = -eulerConstant;
					for (ii=1;ii<=nm1;ii++) {
						psi += 1.0/ii;
					}
					del=fact*(-log(x)+psi);
				}
				ans+=del;
				if (fabs(del)<fabs(ans)*epsilon) return ans;
			}
			//cout << "Series failed in exponentialIntegral" << endl;
			return ans;
		}
	}
	return ans;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  parseArguments
 *    Arguments:  int argc - Number of command line arguments.
 *      Returns:  0 if success otherwise -1.
 *  Description:  Parses the command line arguments and determines what options to set.
 * =====================================================================================
 */

int parseArguments (int argc, char *argv[]) {
	int c;
	while ((c = getopt (argc, argv, "cghn:m:a:b:tvo:s:")) != -1) {
		switch(c) {
			case 'c':
				cpu=false; break;	 //Skip the CPU test
			case 'h':
				printUsage(); exit(0); break;
			case 'n':
				order = atoi(optarg); break;
			case 'm':
				numberOfSamples = atoi(optarg); break;
			case 'a':
				sampleRegionStart = atof(optarg); break;
			case 'b':
				sampleRegionEnd = atof(optarg); break;
			case 'o':
				blockSizeOr = atof(optarg); break;
			case 's':
				blockSizeSm = atof(optarg); break;
			case 't':
				timing = true; break;
			case 'v':
				verbose = true; break;
			case 'g':
				gpu = false ; break;
			default:
				fprintf(stderr, "Invalid option given\n");
				printUsage();
				return -1;
		}
	}
	return 0;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  printUsage
 *  Description:  Prints the usage info for this program.
 * =====================================================================================
 */

void printUsage () {
	printf("exponentialIntegral program\n");
	printf("by: Jose Mauricio Refojo <refojoj@tcd.ie>\n");
	printf("This program will calculate a number of exponential integrals\n");
	printf("usage:\n");
	printf("exponentialIntegral.out [options]\n");
	printf("      -a   value   : will set the a value of the (a,b) interval in which the samples are taken to value (default: 0.0)\n");
	printf("      -b   value   : will set the b value of the (a,b) interval in which the samples are taken to value (default: 10.0)\n");
	printf("      -c           : will skip the CPU test\n");
	printf("      -g           : will skip the GPU test\n");
	printf("      -h           : will show this usage\n");
	printf("      -n   size    : will set the n (the order up to which we are calculating the exponential integrals) to size (default: 10)\n");
	printf("      -m   size    : will set the number of samples taken in the (a,b) interval to size (default: 10)\n");
	printf("      -t           : will output the amount of time that it took to generate each norm (default: no)\n");
	printf("      -v           : will activate the verbose mode  (default: no)\n");
	printf("      -o           : Number of threads in a order blocks  (default: 32)\n");
	printf("      -s           : Number of threads in a sample blocks  (default: 32)\n");
	printf("     \n");
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  checkDeviation
 *    Arguments:  
 *      Returns:  
 *  Description:  
 * =====================================================================================
 */

template <typename DataType>
void checkDeviation(std::vector<std::vector<DataType>> & resultsCpu, DataType * resultsGpu) {
	DataType dev = 0.f ;
	bool noDevs = true ;
	for (int i = 0; i < resultsCpu.size() ; ++i) {
		for (int j = 0 ; j < resultsCpu[i].size() ; ++j) {
			if ( (dev = std::abs(resultsCpu[i][j] - resultsGpu[i*resultsCpu[i].size()+j])) > 1E-5) {
				printf("Deviation at result [%d %d %lf]\n", i+1, j+1, dev) ; 
				noDevs = false ;
			}
		}
	}
	if (noDevs) {
		std::cout << "No Deviations!" << std::endl;
	}
}		/* -----  end of function checkDeviation  ----- */

