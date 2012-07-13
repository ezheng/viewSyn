//#include <stdio>
//#include <stdlib.h>
#include <string>
#include "GaussianBlurCUDA.h"



//filter kernel width range (don't change these)
#define KERNEL_MAX_WIDTH 45       //do not change!!!
#define KERNEL_MIN_WIDTH  5       //do not change!!!
#define FILTER_WIDTH_FACTOR 5.0f

#define THREADS_NUMBER_H 16
#define THREADS_NUMBER_V 16

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel
//////////////////////////////////////////////////////////////////////////////// 
__device__ __constant__ float g_Kernel[KERNEL_MAX_WIDTH]; 

// declare texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex32F0;


surface<void, cudaSurfaceType3D> colorTex_Surface3D;
surface<void, cudaSurfaceType2D> temp_Surface2D;


////////////////////////////////////////////////////////////////////////////////
// GPU-specific defines
////////////////////////////////////////////////////////////////////////////////
//Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )
//Use unrolled innermost convolution loop
//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b){ return (a % b != 0) ? (a / b + 1) : (a / b); }
//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b){ return (a % b != 0) ?  (a - a % b + b) : a; }



////////////////////////////////////////////////////////////////////////////////
// Kernel Row convolution filter
////////////////////////////////////////////////////////////////////////////////
template<int FR> __global__ void convolutionRowsKernel( int imageW, int imageH, int layerId )
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
   // const float  x = (float)ix + 0.5f;
   // const float  y = (float)iy + 0.5f;
	//if(ix >= imageW || iy >= imageH) return;
	
	if(ix <imageW && iy<imageH)
	{
		uchar4 sum = make_uchar4(0, 0, 0, 0);
		uchar4 data;
	  //  for(int k = -FR; k <= FR; k++){ sum += tex2D(tex32F0, x + (float)k, y) * g_Kernel[FR - k]; }
#pragma unroll
		for(int k = -FR; k <= FR; k++)
		{
			//sum += tex2D(tex32F0, x + (float)k, y) * g_Kernel[FR - k];	
			//surf3Dread(&data, colorTex_Surface3D, 0, 0, layerId, cudaBoundaryModeClamp);
			surf3Dread(&data, colorTex_Surface3D, (ix + k) * 4, iy, layerId, cudaBoundaryModeClamp);			
			sum.x += data.x * g_Kernel[FR - k];
			sum.y += data.y * g_Kernel[FR - k];
			sum.z += data.z * g_Kernel[FR - k];			
		}
		if(ix==0 && iy<479 && layerId == 0)
		{printf("x: %u, y: %u, z: %u, w: %u\n", sum.x, sum.x, sum.z, sum.w);}
		sum.w = 255;
		surf2Dwrite(sum, temp_Surface2D, ix * 4, iy, cudaBoundaryModeTrap);
	} 

	//d_Dst[ IMAD(iy, imageW, ix) ] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// Kernel Column convolution filter
////////////////////////////////////////////////////////////////////////////////
template<int FR> __global__ void convolutionColsKernel( int imageW, int imageH, int layerId )
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

	if(ix <imageW && iy<imageH)
	{
		uchar4 sum = make_uchar4(0, 0, 0, 0);
		uchar4 data;
	  //for(int k = -FR; k <= FR; k++){ sum += tex2D(tex32F0, x, y + (float)k) * g_Kernel[FR - k]; }
		for(int k = -FR; k <= FR; k++)
		{
			//sum += tex2D(tex32F0, x + (float)k, y) * g_Kernel[FR - k];
			surf2Dread(&data, temp_Surface2D, ix * 4, iy + k, cudaBoundaryModeClamp);
			sum.x += data.x * g_Kernel[FR - k];
			sum.y += data.y * g_Kernel[FR - k];
			sum.z += data.z * g_Kernel[FR - k];
		}
		sum.w = 255;
		surf3Dwrite(sum, colorTex_Surface3D, ix * 4, iy, layerId, cudaBoundaryModeTrap);
	} 

	//d_Dst[IMAD(iy, imageW, ix)] = sum;
}


GaussianBlurCUDA::GaussianBlurCUDA(int width, int height, float sigma): m_nWidth(width), m_nHeight(height), m_paraSigma(sigma)
{
	// 
	cudaChannelFormatDesc ucharTex  = cudaCreateChannelDesc<uchar4>();
	cudaMallocArray(&_temp2DArray, &ucharTex, m_nWidth, m_nHeight, cudaArraySurfaceLoadStore);
	
	//construct kernel for smoothing gradients
	float filter_kernel[KERNEL_MAX_WIDTH]; 
	CreateFilterKernel(m_paraSigma, filter_kernel, m_nKernelWidth);
	cudaMemcpyToSymbol(g_Kernel, filter_kernel, m_nKernelWidth*sizeof(float), 0, cudaMemcpyHostToDevice); //copy kernel to device memory.
}

GaussianBlurCUDA::~GaussianBlurCUDA()
{
	cudaFreeArray(_temp2DArray);
}

void GaussianBlurCUDA::CreateFilterKernel(float sigma, float* kernel, int& width)
{
	int i, sz;
	width = (int)(FILTER_WIDTH_FACTOR * sigma);
	if( width%2 == 0 ){ width+=1; }
	sz = (width-1)>>1;

	if(width > KERNEL_MAX_WIDTH)
	{
		//filter size truncation
		sz = KERNEL_MAX_WIDTH >> 1;
		width = KERNEL_MAX_WIDTH;
	}else if(width < KERNEL_MIN_WIDTH)
	{
		sz = KERNEL_MIN_WIDTH >> 1;
		width = KERNEL_MIN_WIDTH;
	}

	float rv = -0.5f/(sigma*sigma), v, ksum = 0.f; 

	// pre-compute filter
	for( i = -sz ; i <= sz ; ++i) {
		kernel[i+sz] = v = exp( i * i * rv ) ;
		ksum += v;
	}

	//normalize the kernel
	rv = 1.0f/ksum; for(i=0; i<width; i++) kernel[i]*=rv;
}
#include <iostream>
namespace{
void CUDA_SAFE_CALL( cudaError_t err, std::string file = __FILE__, int line = __LINE__)
{
	if (err != cudaSuccess) {
		std::cout<< cudaGetErrorString( err ) << " in file: " << file << " at line: " << line << std::endl;
        //printf( "%s in %s at line %i\n", cudaGetErrorString( err ),
          //      file.c_str(), line );
        exit( EXIT_FAILURE );
    }
}}

template<int FR> void GaussianBlurCUDA::FilterImage(cudaArray *array3D, int numOfLayers)
{
	dim3 threads(THREADS_NUMBER_H, THREADS_NUMBER_V);
    dim3 blocks( iDivUp(m_nWidth, threads.x), iDivUp(m_nHeight, threads.y) ); //number of blocks required

	// horizontal pass
	CUDA_SAFE_CALL(cudaBindSurfaceToArray(colorTex_Surface3D, array3D));
	CUDA_SAFE_CALL(cudaBindSurfaceToArray(temp_Surface2D, _temp2DArray));


	
	//horizontal pass:
	//cudaBindTextureToArray(tex32F0, src);
	for(int layerId = 0; layerId<numOfLayers; layerId++)
	{
	convolutionRowsKernel<FR><<<blocks, threads>>>( m_nWidth, m_nHeight, layerId );
	//cudaUnbindTexture(tex32F0);
	//cudaMemcpyToArray(m_cuaTmp, 0, 0, m_buf32FA, m_nWidth*m_nHeight*sizeof(float), cudaMemcpyDeviceToDevice);
	
	//vertical pass:
	//cudaBindTextureToArray(tex32F0, m_cuaTmp);
	convolutionColsKernel<FR><<<blocks, threads>>>( m_nWidth, m_nHeight, layerId );
	//cudaUnbindTexture(tex32F0);
	//cudaMemcpyToArray(    dst, 0, 0, m_buf32FA, m_nWidth*m_nHeight*sizeof(float), cudaMemcpyDeviceToDevice);	
	
	}
}


void GaussianBlurCUDA::Filter(cudaArray *array3D, int numOfLayers)
{
	switch( m_nKernelWidth>>1 /*kernel radius*/ )
	{
		case 2:	 FilterImage< 2>(array3D,  numOfLayers);	break;
		case 3:	 FilterImage< 3>(array3D,  numOfLayers);	break;
		case 4:	 FilterImage< 4>(array3D,  numOfLayers);	break;
		case 5:	 FilterImage< 5>(array3D,  numOfLayers);	break;
		case 6:	 FilterImage< 6>(array3D,  numOfLayers);	break;
		case 7:	 FilterImage< 7>(array3D,  numOfLayers);	break;
		case 8:	 FilterImage< 8>(array3D,  numOfLayers);	break;
		case 9:	 FilterImage< 9>(array3D,  numOfLayers);	break;
		case 10: FilterImage<10>(array3D,  numOfLayers);	break;
		case 11: FilterImage<11>(array3D,  numOfLayers);	break;
		case 12: FilterImage<12>(array3D,  numOfLayers);	break;
		case 13: FilterImage<13>(array3D,  numOfLayers);	break;
		case 14: FilterImage<14>(array3D,  numOfLayers);	break;
		case 15: FilterImage<15>(array3D,  numOfLayers);	break;
		case 16: FilterImage<16>(array3D,  numOfLayers);	break;
		case 17: FilterImage<17>(array3D,  numOfLayers);	break;
		case 18: FilterImage<18>(array3D,  numOfLayers);	break;
		case 19: FilterImage<19>(array3D,  numOfLayers);	break;
		case 20: FilterImage<20>(array3D,  numOfLayers);	break;
		case 21: FilterImage<21>(array3D,  numOfLayers);	break;
		case 22: FilterImage<22>(array3D,  numOfLayers);	break;
		default: break;
	}
}