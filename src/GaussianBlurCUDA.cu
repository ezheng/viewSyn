//#include <stdio>
//#include <stdlib.h>
#include <string>
#include "GaussianBlurCUDA.h"
#include <iostream>


//filter kernel width range (don't change these)
#define KERNEL_MAX_WIDTH 45       //do not change!!!
#define KERNEL_MIN_WIDTH  5       //do not change!!!
#define FILTER_WIDTH_FACTOR 5.0f

#define THREADS_NUMBER_H 8
#define THREADS_NUMBER_V 8
//
#define THRESHOLD_DETECT_HOLES 5.0f

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel
//////////////////////////////////////////////////////////////////////////////// 
__device__ __constant__ float g_Kernel[KERNEL_MAX_WIDTH]; 

// declare texture reference for 2D float texture
//texture<float, 2, cudaReadModeElementType> tex32F0;


//surface<void, cudaSurfaceType3D> colorTex_Surface3D;
surface<void, cudaSurfaceType3D> cost_Surface3D;

surface<void, cudaSurfaceType2D> temp_Surface2D;
surface<void, cudaSurfaceType2D> temp_Surface2D_2;
surface<void, cudaSurfaceType2D> depthmap2D_Surface2D;
surface<void, cudaSurfaceType2D> depthmap2DBackup_Surface2D;

texture<uchar4, cudaTextureType2D, cudaReadModeElementType> colorImageTex;


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
template<int FR> __global__ void convolutionRowsKernel( int imageW, int imageH, int layerId)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	
	if(ix <imageW && iy<imageH)
	{
		float sum(0.0f);
		float data;
#pragma unroll
		for(int k = -FR; k <= FR; k++)
		{
			surf3Dread(&data, cost_Surface3D, (ix + k) * 4, iy, layerId, cudaBoundaryModeClamp);
			sum += (data * g_Kernel[FR - k]);
		}
		surf2Dwrite(sum, temp_Surface2D, ix * 4, iy, cudaBoundaryModeTrap);
	} 
}

template<int FR> __global__ void convolutionRowsKernel_2( int imageW, int imageH, int layerId)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	
	if(ix <imageW && iy<imageH)
	{
		float sum(0.0f);
		float data;
#pragma unroll
		for(int k = -FR; k <= FR; k++)
		{
			surf3Dread(&data, cost_Surface3D, (ix + k) * 4, iy, layerId, cudaBoundaryModeClamp);
			sum += (data * g_Kernel[FR - k]);
		}
		surf2Dwrite(sum, temp_Surface2D_2, ix * 4, iy, cudaBoundaryModeTrap);
	} 
}

template<int FR> __global__ void convolutionRowsKernelRemoveNoise(int imageW, int imageH )
{
	 const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
     const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	 if(ix < imageW && iy < imageH)
	 {
		float sum = 0;
		int planeIdx;
#pragma unroll
		for(int k = -FR; k <= FR; k++)
		{
			surf2Dread( &planeIdx, depthmap2D_Surface2D, (ix + k) * 4, iy, cudaBoundaryModeClamp);
			sum += ( float(planeIdx) * g_Kernel[FR - k]);
		}
		surf2Dwrite(sum, temp_Surface2D, ix * 4, iy, cudaBoundaryModeTrap);
	 }

}

////////////////////////////////////////////////////////////////////////////////
// Kernel Column convolution filter
////////////////////////////////////////////////////////////////////////////////
template<int FR> __global__ void convolutionColsKernel( int imageW, int imageH, int layerId)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

	if(ix <imageW && iy<imageH)
	{
		float sum = 0;
		float data;
#pragma unroll
		for(int k = -FR; k <= FR; k++)
		{
			surf2Dread(&data, temp_Surface2D, ix * 4, iy + k, cudaBoundaryModeClamp);
			sum += (data * g_Kernel[FR - k]);
		}
		surf3Dwrite(sum, cost_Surface3D, ix * 4, iy, layerId, cudaBoundaryModeTrap);
	}
}

template<int FR> __global__ void convolutionColsKernel_2( int imageW, int imageH, int layerId)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

	if(ix <imageW && iy<imageH)
	{
		float sum = 0;
		float data;
#pragma unroll
		for(int k = -FR; k <= FR; k++)
		{
			surf2Dread(&data, temp_Surface2D_2, ix * 4, iy + k, cudaBoundaryModeClamp);
			sum += (data * g_Kernel[FR - k]);
		}
		surf3Dwrite(sum, cost_Surface3D, ix * 4, iy, layerId, cudaBoundaryModeTrap);
	}
}



template<int FR> __global__ void convolutionColsKernelRemoveNoise(int imageW, int imageH)
{
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

	if(ix <imageW && iy<imageH)
	{
		float sum = 0;
		float data;
		for(int k = -FR; k <= FR; k++)
		{
			surf2Dread(&data, temp_Surface2D, ix * 4, iy + k, cudaBoundaryModeClamp);
			sum += (data * g_Kernel[FR - k]);
		}
		int planeIdx;
		surf2Dread( &planeIdx, depthmap2D_Surface2D, ix * 4, iy, cudaBoundaryModeClamp);
		if( abs(sum - float(planeIdx)) > THRESHOLD_DETECT_HOLES )
		{
			surf2Dwrite(-1, depthmap2D_Surface2D, ix * 4, iy, cudaBoundaryModeTrap); // write the unreliable depth index with a special number
		}
	}

}

GaussianBlurCUDA::GaussianBlurCUDA(int width, int height, float sigma): m_nWidth(width), m_nHeight(height), m_paraSigma(sigma)
{
	// 
	cudaChannelFormatDesc ucharTex  = cudaCreateChannelDesc<uchar4>();
	cudaMallocArray(&_temp2DArray, &ucharTex, m_nWidth, m_nHeight, cudaArraySurfaceLoadStore);
	cudaMallocArray(&_temp2DArray_2, &ucharTex, m_nWidth, m_nHeight, cudaArraySurfaceLoadStore);

	//cudaArray * depthmap2D_backup;
	cudaChannelFormatDesc intTex  = cudaCreateChannelDesc<int>();
	cudaMallocArray(&_depthmap2D_backup, &intTex, m_nWidth, m_nHeight, cudaArraySurfaceLoadStore);
	
	//
	for (int i = 0; i < 2; ++i) cudaStreamCreate(&stream[i]);

	//construct kernel for smoothing gradients
	float filter_kernel[KERNEL_MAX_WIDTH]; 
	CreateFilterKernel(m_paraSigma, filter_kernel, m_nKernelWidth);
	cudaMemcpyToSymbol(g_Kernel, filter_kernel, m_nKernelWidth*sizeof(float), 0, cudaMemcpyHostToDevice); //copy kernel to device memory.
}

GaussianBlurCUDA::~GaussianBlurCUDA()
{
	cudaFreeArray(_temp2DArray);
	cudaFreeArray(_temp2DArray_2);
	cudaFreeArray(_depthmap2D_backup);
	for (int i = 0; i < 2; ++i) 
		cudaStreamDestroy(stream[i]);
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
	//CUDA_SAFE_CALL(cudaBindSurfaceToArray(colorTex_Surface3D, array3D));
	CUDA_SAFE_CALL(cudaBindSurfaceToArray(cost_Surface3D, array3D));
	CUDA_SAFE_CALL(cudaBindSurfaceToArray(temp_Surface2D, _temp2DArray));
	CUDA_SAFE_CALL(cudaBindSurfaceToArray(temp_Surface2D_2, _temp2DArray_2));

	//horizontal pass:
	//cudaBindTextureToArray(tex32F0, src);
	//cudaStream_t stream[2]; for (int i = 0; i < 2; ++i) cudaStreamCreate(&stream[i]);

	for(int layerId = 0; layerId< (numOfLayers/2); layerId++)
	{
		convolutionRowsKernel<FR><<<blocks, threads, 0, stream[0]>>>( m_nWidth, m_nHeight, layerId * 2);
		convolutionRowsKernel_2<FR><<<blocks, threads, 0, stream[1]>>>( m_nWidth, m_nHeight, layerId * 2 + 1);

		convolutionColsKernel<FR><<<blocks, threads, 0, stream[0]>>>( m_nWidth, m_nHeight, layerId * 2);
		convolutionColsKernel_2<FR><<<blocks, threads, 0, stream[1]>>>( m_nWidth, m_nHeight, layerId * 2 + 1);
	}
	//cudaStreamSynchronize()
	//for (int i = 0; i < 2; ++i) cudaStreamDestroy(stream[i]);
	cudaDeviceSynchronize();
}

template<int FR> void GaussianBlurCUDA::RemoveUnreliableDepthImage(cudaArray *depthmap_CUDAArray)
{
	dim3 threads(THREADS_NUMBER_H, THREADS_NUMBER_V);
    dim3 blocks( iDivUp(m_nWidth, threads.x), iDivUp(m_nHeight, threads.y) ); //number of blocks required

	CUDA_SAFE_CALL(cudaBindSurfaceToArray(temp_Surface2D, _temp2DArray));
	CUDA_SAFE_CALL(cudaBindSurfaceToArray(depthmap2D_Surface2D, depthmap_CUDAArray));

	convolutionRowsKernelRemoveNoise<FR><<<blocks, threads>>>( m_nWidth, m_nHeight);
	convolutionColsKernelRemoveNoise<FR><<<blocks, threads>>>( m_nWidth, m_nHeight);
}

void GaussianBlurCUDA::RemoveUnreliableDepth( cudaArray *depthmap_CUDAArray)
{
	switch( m_nKernelWidth>>1 /*kernel radius*/ )
	{
		case 2:	 RemoveUnreliableDepthImage< 2>(depthmap_CUDAArray);	break;
		case 3:	 RemoveUnreliableDepthImage< 3>(depthmap_CUDAArray);	break;
		case 4:	 RemoveUnreliableDepthImage< 4>(depthmap_CUDAArray);	break;
		case 5:	 RemoveUnreliableDepthImage< 5>(depthmap_CUDAArray);	break;
		case 6:	 RemoveUnreliableDepthImage< 6>(depthmap_CUDAArray);	break;
		case 7:	 RemoveUnreliableDepthImage< 7>(depthmap_CUDAArray);	break;
		case 8:	 RemoveUnreliableDepthImage< 8>(depthmap_CUDAArray);	break;
		case 9:	 RemoveUnreliableDepthImage< 9>(depthmap_CUDAArray);	break;
		case 10: RemoveUnreliableDepthImage<10>(depthmap_CUDAArray);	break;
		case 11: RemoveUnreliableDepthImage<11>(depthmap_CUDAArray);	break;
		case 12: RemoveUnreliableDepthImage<12>(depthmap_CUDAArray);	break;
		case 13: RemoveUnreliableDepthImage<13>(depthmap_CUDAArray);	break;
		case 14: RemoveUnreliableDepthImage<14>(depthmap_CUDAArray);	break;
		case 15: RemoveUnreliableDepthImage<15>(depthmap_CUDAArray);	break;
		case 16: RemoveUnreliableDepthImage<16>(depthmap_CUDAArray);	break;
		case 17: RemoveUnreliableDepthImage<17>(depthmap_CUDAArray);	break;
		case 18: RemoveUnreliableDepthImage<18>(depthmap_CUDAArray);	break;
		case 19: RemoveUnreliableDepthImage<19>(depthmap_CUDAArray);	break;
		case 20: RemoveUnreliableDepthImage<20>(depthmap_CUDAArray);	break;
		case 21: RemoveUnreliableDepthImage<21>(depthmap_CUDAArray);	break;
		case 22: RemoveUnreliableDepthImage<22>(depthmap_CUDAArray);	break;
		default: break;
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

__device__ void findPosLeft(int x, int y, float *weight, int *planeIdx, int halfPatchSize)
{
	*weight = 0.0f; *planeIdx = 0;
	uchar4 centerColor = tex2D(colorImageTex, x + 0.5, y + 0.5);
	for(int i = 1; i <= halfPatchSize; i++)
	{
		int newX = x - i;
		if(newX < 0)
			continue;
		else
		{	
			surf2Dread( planeIdx, depthmap2DBackup_Surface2D, newX * 4, y, cudaBoundaryModeClamp);
			if((*planeIdx) != -1)
			{
				//calculate weight and then return
				uchar4 color = tex2D(colorImageTex, newX + 0.5, y + 0.5);

				*weight = (255.0f - abs(float(color.x) - float(centerColor.x)))/255.0f; 
				*weight *= (255.0f - abs(float(color.y) - float(centerColor.y)))/255.0f; 
				*weight *= (255.0f - abs(float(color.z) - float(centerColor.z)))/255.0f; 
				*weight *= float(halfPatchSize - i)/float(halfPatchSize);
				break;
			}
		}
	}

}

__device__ void findPosRight(int x, int y, float *weight, int *planeIdx, int halfPatchSize, int imageW)
{
	*weight = 0.0f; *planeIdx = 0;
	uchar4 centerColor = tex2D(colorImageTex, x + 0.5, y + 0.5);
	for(int i = 1; i <= halfPatchSize; i++)
	{
		int newX = x + i;
		if(newX >= imageW)
			continue;
		else
		{	
			surf2Dread( planeIdx, depthmap2DBackup_Surface2D, newX * 4, y, cudaBoundaryModeTrap);
			if(*planeIdx != -1)
			{
				uchar4 color = tex2D(colorImageTex, newX + 0.5, y + 0.5);
				*weight = (255.0f - abs(float(color.x) - float(centerColor.x)))/255.0f; 
				*weight *= (255.0f - abs(float(color.y) - float(centerColor.y)))/255.0f; 
				*weight *= (255.0f - abs(float(color.z) - float(centerColor.z)))/255.0f; 
				*weight *= float(halfPatchSize - i)/float(halfPatchSize);
				break;
			}
		}
	}
}

__device__ void findPosUp(int x, int y, float *weight, int *planeIdx, int halfPatchSize)
{
	*weight = 0.0f; *planeIdx = 0;
	uchar4 centerColor = tex2D(colorImageTex, x + 0.5, y + 0.5);
	for(int i = 1; i <= halfPatchSize; i++)
	{
		int newY = y - i;
		if(newY < 0)
			continue;
		else
		{	
			
			surf2Dread( planeIdx, depthmap2DBackup_Surface2D, x * 4, newY, cudaBoundaryModeTrap);
			if(*planeIdx != -1)
			{
				//calculate weight and then return
				uchar4 color = tex2D(colorImageTex, x + 0.5, newY + 0.5);
				*weight = (255.0f - abs(float(color.x) - float(centerColor.x)))/255.0f; 
				*weight *= (255.0f - abs(float(color.y) - float(centerColor.y)))/255.0f; 
				*weight *= (255.0f - abs(float(color.z) - float(centerColor.z)))/255.0f; 
				*weight *= float(halfPatchSize - i)/float(halfPatchSize);
				break;
			}
		}
	}
}

__device__ void findPosDown(int x, int y, float *weight, int *planeIdx, int halfPatchSize, int imageH)
{
	*weight = 0.0f; *planeIdx = 0;
	uchar4 centerColor = tex2D(colorImageTex, x + 0.5, y + 0.5);
	for(int i = 1; i <= halfPatchSize; i++)
	{
		int newY = y + i;
		if(newY >= imageH)
			continue;
		else
		{	
			surf2Dread( planeIdx, depthmap2DBackup_Surface2D, x * 4, newY, cudaBoundaryModeTrap);
			if(*planeIdx != -1)
			{
				uchar4 color = tex2D(colorImageTex, x + 0.5, newY + 0.5);
				*weight = (255.0f - abs(float(color.x) - float(centerColor.x)))/255.0f; 
				*weight *= (255.0f - abs(float(color.y) - float(centerColor.y)))/255.0f; 
				*weight *= (255.0f - abs(float(color.z) - float(centerColor.z)))/255.0f; 
				*weight *= float(halfPatchSize - i)/float(halfPatchSize);
				break;
			}
		}
	}
}


__global__ void fillHolesDepth_kernel(int imageW, int imageH)
{
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

	if(ix <imageW && iy<imageH)
	{
		//int newPlaneIdx_int = 120;
		int centerPlaneIdx;
		surf2Dread( &centerPlaneIdx, depthmap2DBackup_Surface2D, ix * 4, iy, cudaBoundaryModeClamp);
		if( centerPlaneIdx == -1)
		{
			// search in depthmap2DBackup_Surface2D and colorImage_Surface2D. 
			float weight[4]= {0}; int planeIndex[4] = {0};
			int halfPatchSize = 20;
			findPosLeft(ix, iy, &(weight[0]), &(planeIndex[0]), halfPatchSize);
			findPosRight(ix, iy, &(weight[1]), &(planeIndex[1]), halfPatchSize, imageW);
			findPosUp(ix, iy, &(weight[2]), &(planeIndex[2]), halfPatchSize);
			findPosDown(ix, iy, &(weight[3]), &(planeIndex[3]), halfPatchSize, imageH);
			////	// do interpolation, then round up
			float sumWeight = 0.0f;
			float newplaneIdx = 0.0f;
			for(int i = 0; i<4; i++)
			{
				sumWeight += weight[i];
				newplaneIdx += weight[i] * float(planeIndex[i]);
			}
			//sumWeight = 1;
			int newPlaneIdx_int;
			if(sumWeight == 0)
			{
				newPlaneIdx_int = -1;
			}
			else
			{
				newPlaneIdx_int = int((newplaneIdx/sumWeight + 0.5));
			}
			surf2Dwrite(newPlaneIdx_int, depthmap2D_Surface2D, ix * 4, iy, cudaBoundaryModeTrap);
		}
	}
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

void GaussianBlurCUDA::fillHolesDepth(cudaArray *depthmap_CUDAArray, cudaArray *colorImage_CUDAArray)
{
	dim3 threads(THREADS_NUMBER_H, THREADS_NUMBER_V);
    dim3 blocks( iDivUp(m_nWidth, threads.x), iDivUp(m_nHeight, threads.y) ); //number of blocks required

	CUDA_SAFE_CALL(cudaMemcpyArrayToArray(	_depthmap2D_backup, 0, 0, depthmap_CUDAArray, 0, 0, m_nWidth * m_nHeight * sizeof(int), cudaMemcpyDeviceToDevice));

	CUDA_SAFE_CALL(cudaBindSurfaceToArray(depthmap2DBackup_Surface2D, _depthmap2D_backup));
	CUDA_SAFE_CALL(cudaBindSurfaceToArray(depthmap2D_Surface2D, depthmap_CUDAArray));
	CUDA_SAFE_CALL(cudaBindTextureToArray(colorImageTex, colorImage_CUDAArray));
	colorImageTex.normalized = false;

	fillHolesDepth_kernel<<<blocks, threads>>>( m_nWidth, m_nHeight);
	
	__cudaCheckError(__FILE__, __LINE__);

	CUDA_SAFE_CALL(cudaUnbindTexture( colorImageTex));
}

