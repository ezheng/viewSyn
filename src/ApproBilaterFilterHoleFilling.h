#pragma once

#include <cuda_runtime.h>

class ApproBilaterFilterHoleFilling
{
public:
	ApproBilaterFilterHoleFilling(int width, int height, float sigma_proximity, float sigma_similarity); //the filter size will be filterwidthfactor*sigma*2+1
	~ApproBilaterFilterHoleFilling();
	//int Fill( float* dst, const float* src, const unsigned char* im_color, const bool is_readback);
	void Fill(cudaArray *src, cudaArray *im_color);
private:
	template<int FR> void FillImage(cudaArray *src, cudaArray* im_color); //filter width
	void CreateProximityFilterKernel(float sigma, float* kernel, int& width);
	void CreateColorWeightLUT(float sigma, float* lut, const int length);
private:
	int m_nWidth, m_nHeight, m_nKernelWidth;
	float m_paraSigmaProx, m_paraSigmaSimi;
private:
	//cuda array
	//cudaArray* m_cuaSrc;   //store the src image
	//cudaArray* m_cuaColor; //store the src color image
	cudaArray* m_cuaTmp;
	//cudaArray* m_cuaResult;
	//float*	   m_buf32FA;
	//float*     m_bufRGB8A;
};