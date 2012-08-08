#ifndef GAUSSIANBLURCUDA_H
#define GAUSSIANBLURCUDA_H

#include <cuda_runtime.h>


class GaussianBlurCUDA
{
public:
	GaussianBlurCUDA(int width, int height, float sigma); //the filter size will be filterwidthfactor*sigma*2+1
	~GaussianBlurCUDA();

	void Filter(cudaArray * array3D, int numOfLayers);
	void RemoveUnreliableDepth( cudaArray *depthmap_CUDAArray);
	void fillHolesDepth(cudaArray *depthmap_CUDAArray, cudaArray *colorImage_CUDAArray);
private:
	template<int FR> void FilterImage(cudaArray *array3D, int numOfLayers); //filter width
	template<int FR> void RemoveUnreliableDepthImage(cudaArray *depthmap_CUDAArray);
	void CreateFilterKernel(float sigma, float* kernel, int& width);
private:
	int m_nWidth, m_nHeight, m_nKernelWidth;
	float m_paraSigma;
private:

	cudaArray *_temp2DArray;
	cudaArray *_depthmap2D_backup;
};



#endif