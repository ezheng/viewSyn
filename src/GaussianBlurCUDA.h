#ifndef GAUSSIANBLURCUDA_H
#define GAUSSIANBLURCUDA_H

#include <cuda_runtime.h>


class GaussianBlurCUDA
{
public:
	GaussianBlurCUDA(int width, int height, float sigma); //the filter size will be filterwidthfactor*sigma*2+1
	~GaussianBlurCUDA();

	void Filter(cudaArray * array3D, int numOfLayers);

private:
	template<int FR> void FilterImage(cudaArray *array3D, int numOfLayers); //filter width
	void CreateFilterKernel(float sigma, float* kernel, int& width);
private:
	int m_nWidth, m_nHeight, m_nKernelWidth;
	float m_paraSigma;
private:

	cudaArray *_temp2DArray;

};



#endif