//#include <stdio.h>
//#include <stdlib.h>
#include <iostream>
#include <string>
#include "ApproBilaterFilterHoleFilling.h"

//filter kernel width range (don't change these)

#define COLORWEIGHT_LUT_LENGTH 256       //do not change!!!
#define KERNEL_MAX_WIDTH        51       //do not change!!!
#define KERNEL_MIN_WIDTH         5       //do not change!!!
#define FILTER_WIDTH_FACTOR   5.0f

#define HOLE_REGION_VALUE -1

#define THREADS_NUMBER_H 16
#define THREADS_NUMBER_V 12

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel
//////////////////////////////////////////////////////////////////////////////// 
__device__ __constant__ float g_Kernel[KERNEL_MAX_WIDTH]; 
__device__ __constant__ float g_ColorWeightLUT[COLORWEIGHT_LUT_LENGTH];

// declare texture reference for 2D float texture
//texture<float, 2, cudaReadModeElementType> tex32F0;
//texture<float, 2, cudaReadModeElementType> tex32F1;
surface<void, cudaSurfaceType2D> temp_Surface2D;
surface<void, cudaSurfaceType2D> depthmap2D_Surface2D;

// declare texture reference for 2D RGBA texture
//texture<uchar4, 2, cudaReadModeElementType> texRGBA8;
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
template<int FR> __global__ void convolutionRowsKernel(int imageW, int imageH )
{
    const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if(ix >= imageW || iy >= imageH) return;
	
	float sumval = 0.f;
	float sumwei = 0.f;
	float result = 0.f;
	float weight, delta_r, delta_g, delta_b, similarity;
	int value;
	int lut_bin_index;
	
	int value_cen ;
	surf2Dread(&value_cen, depthmap2D_Surface2D, ix * 4, iy , cudaBoundaryModeClamp);

	if( value_cen == HOLE_REGION_VALUE ){
		uchar4 clr_cen = tex2D(colorImageTex, ix + 0.5, iy + 0.5);
		for(int k = -FR; k <= FR; k++){ 
			//value = tex2D(tex32F1, x + (float)k, y);
			surf2Dread(&value, depthmap2D_Surface2D, (ix + k)* 4, iy , cudaBoundaryModeClamp);

			if( value == HOLE_REGION_VALUE ){ continue; }
			uchar4 clr_ref = tex2D(colorImageTex, ix + k + 0.5f, iy + 0.5f);

			delta_r = (float)clr_cen.x - (float)clr_ref.x;
			delta_g = (float)clr_cen.y - (float)clr_ref.y;
			delta_b = (float)clr_cen.z - (float)clr_ref.z;
			similarity = sqrtf( delta_r*delta_r + delta_g*delta_g + delta_b*delta_b );
			lut_bin_index = (int)floorf( fmin(similarity, (float)COLORWEIGHT_LUT_LENGTH) );
			weight = g_Kernel[FR - k] * g_ColorWeightLUT[lut_bin_index];
			sumval += value * weight; 
			sumwei += weight;
		}
		result = (sumwei>0.f) ? sumval/sumwei : HOLE_REGION_VALUE;
    }else{
		result = value_cen;
	}
	// write to a float texture
	surf2Dwrite(result, temp_Surface2D , ix * 4, iy, cudaBoundaryModeTrap); // write the unreliable depth index with a special number
}

////////////////////////////////////////////////////////////////////////////////
// Kernel Column convolution filter
////////////////////////////////////////////////////////////////////////////////
template<int FR> __global__ void convolutionColsKernel(int imageW, int imageH )
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if(ix >= imageW || iy >= imageH) return;

	float sumval = 0.f;
	float sumwei = 0.f;
	float result = 0.f;
	float weight, delta_r, delta_g, delta_b, similarity;
	int lut_bin_index;

	float value;
	float value_cen;
	surf2Dread(&value_cen, temp_Surface2D, ix * 4, iy , cudaBoundaryModeClamp);
	
	if( value_cen == HOLE_REGION_VALUE ){
		uchar4 clr_cen = tex2D(colorImageTex, ix + 0.5, iy + 0.5);
		for(int k = -FR; k <= FR; k++){ 
			//value = tex2D(tex32F1, x, y + (float)k);
			surf2Dread(&value, temp_Surface2D, ix * 4, iy + k , cudaBoundaryModeClamp);

			if( value == HOLE_REGION_VALUE ){ continue; }
			uchar4 clr_ref = tex2D(colorImageTex, float(ix) + 0.5f, float(iy) + 0.5f + float(k));
			delta_r = (float)clr_cen.x - (float)clr_ref.x;
			delta_g = (float)clr_cen.y - (float)clr_ref.y;
			delta_b = (float)clr_cen.z - (float)clr_ref.z;
			similarity = sqrtf( delta_r*delta_r + delta_g*delta_g + delta_b*delta_b );
			lut_bin_index = (int)floorf( fmin(similarity, (float)COLORWEIGHT_LUT_LENGTH) );
			weight = g_Kernel[FR - k] * g_ColorWeightLUT[lut_bin_index];
			sumval += value * weight; 
			sumwei += weight;
		}
		result = (sumwei>0.f) ? sumval/sumwei : HOLE_REGION_VALUE;
    }else{
		result = value_cen;
	}	
	int result_int = (int)roundf(result);
	surf2Dwrite(result_int, depthmap2D_Surface2D , ix * 4, iy, cudaBoundaryModeTrap); 
}


ApproBilaterFilterHoleFilling::ApproBilaterFilterHoleFilling(int width, int height, float sigma_proximity, float sigma_similarity): m_nWidth(width), m_nHeight(height), m_paraSigmaProx(sigma_proximity), m_paraSigmaSimi(sigma_similarity)
{
	cudaChannelFormatDesc floatTex  = cudaCreateChannelDesc<float>();
	cudaMallocArray(&m_cuaTmp, &floatTex, m_nWidth, m_nHeight, cudaArraySurfaceLoadStore);
	
	//construct kernel for bilateral filter:
	float filter_kernel[KERNEL_MAX_WIDTH]; 
	CreateProximityFilterKernel(m_paraSigmaProx, filter_kernel, m_nKernelWidth);
	cudaMemcpyToSymbol(g_Kernel, filter_kernel, m_nKernelWidth*sizeof(float), 0, cudaMemcpyHostToDevice); //copy kernel to device memory.
	
	float colorweight_lut[COLORWEIGHT_LUT_LENGTH];
	CreateColorWeightLUT(m_paraSigmaSimi, colorweight_lut, COLORWEIGHT_LUT_LENGTH);
	cudaMemcpyToSymbol(g_ColorWeightLUT, colorweight_lut, COLORWEIGHT_LUT_LENGTH*sizeof(float), 0, cudaMemcpyHostToDevice); //copy lut to device memory.
}

ApproBilaterFilterHoleFilling::~ApproBilaterFilterHoleFilling()
{
	cudaFreeArray(m_cuaTmp); 
}

void ApproBilaterFilterHoleFilling::CreateProximityFilterKernel(float sigma, float* kernel, int& width)
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

void ApproBilaterFilterHoleFilling::CreateColorWeightLUT(float sigma, float* lut, const int length)
{
	for(int i=0; i<length; ++i){
		lut[i] = exp(-(float)i/sigma);
	}
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

template<int FR> void ApproBilaterFilterHoleFilling::FillImage(cudaArray *src, cudaArray* im_color)
{
	dim3 threads(THREADS_NUMBER_H, THREADS_NUMBER_V);
    dim3 blocks( iDivUp(m_nWidth, threads.x), iDivUp(m_nHeight, threads.y) ); //number of blocks required
	
	CUDA_SAFE_CALL(cudaBindSurfaceToArray(temp_Surface2D, m_cuaTmp));
	CUDA_SAFE_CALL(cudaBindSurfaceToArray(depthmap2D_Surface2D, src));
	CUDA_SAFE_CALL(cudaBindTextureToArray(colorImageTex, im_color));
	colorImageTex.normalized = false;
	
	convolutionRowsKernel<FR><<<blocks, threads>>>( m_nWidth, m_nHeight );
	convolutionColsKernel<FR><<<blocks, threads>>>( m_nWidth, m_nHeight );
	
	// unbind texture
	cudaUnbindTexture(colorImageTex);
}


void ApproBilaterFilterHoleFilling::Fill(cudaArray *src, cudaArray* im_color)
{
	switch( m_nKernelWidth>>1 /*kernel radius*/ )
	{
		case 2:	 FillImage< 2>(src, im_color);	break;
		case 3:	 FillImage< 3>(src, im_color);	break;
		case 4:	 FillImage< 4>(src, im_color);	break;
		case 5:	 FillImage< 5>(src, im_color);	break;
		case 6:	 FillImage< 6>(src, im_color);	break;
		case 7:	 FillImage< 7>(src, im_color);	break;
		case 8:	 FillImage< 8>(src, im_color);	break;
		case 9:	 FillImage< 9>(src, im_color);	break;
		case 10: FillImage<10>(src, im_color);	break;
		case 11: FillImage<11>(src, im_color);	break;
		case 12: FillImage<12>(src, im_color);	break;
		case 13: FillImage<13>(src, im_color);	break;
		case 14: FillImage<14>(src, im_color);	break;
		case 15: FillImage<15>(src, im_color);	break;
		case 16: FillImage<16>(src, im_color);	break;
		case 17: FillImage<17>(src, im_color);	break;
		case 18: FillImage<18>(src, im_color);	break;
		case 19: FillImage<19>(src, im_color);	break;
		case 20: FillImage<20>(src, im_color);	break;
		case 21: FillImage<21>(src, im_color);	break;
		case 22: FillImage<22>(src, im_color);	break;
		case 23: FillImage<23>(src, im_color);	break;
		case 24: FillImage<24>(src, im_color);	break;
		case 25: FillImage<25>(src, im_color);	break;
		default: break;
	}
}