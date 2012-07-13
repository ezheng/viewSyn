#include <string>
#include <iostream>


//texture<uchar4, cudaTextureType3D, cudaReadModeElementType> layeredTex;
texture<float1, cudaTextureType3D, cudaReadModeElementType> layeredTex;
texture<uchar4, cudaTextureType3D, cudaReadModeElementType> colorTex;

surface<void, cudaSurfaceType3D> colorTex_Surface;

void CUDA_SAFE_CALL( cudaError_t err, std::string file = __FILE__, int line = __LINE__)
{
	if (err != cudaSuccess) {
		std::cout<< cudaGetErrorString( err ) << " in file: " << file << " at line: " << line << std::endl;
        //printf( "%s in %s at line %i\n", cudaGetErrorString( err ),
          //      file.c_str(), line );
        exit( EXIT_FAILURE );
    }
}


__global__ void cudaProcess(unsigned char *out_array, int imageWidth, int imageHeight, int numOfImages, unsigned int numOfCandidatePlanes, float *_maskCUDA, int maskHalfSize)
//__global__ void cudaProcess(unsigned char *out_array, int imageWidth, int imageHeight, int numOfImages, unsigned int numOfCandidatePlanes)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	extern __shared__ float shared[];
	int idInBlock = threadIdx.y * blockDim.x + threadIdx.x;
	int numOfElemMask = (maskHalfSize * 2 + 1) * (maskHalfSize * 2 + 1);
	if(idInBlock < numOfElemMask)
		shared[idInBlock] = _maskCUDA[idInBlock];
	__syncthreads();
	//
	//if(x == 0 && y ==0)
	//{
	//	int maskSize =  maskHalfSize * 2 + 1;
	//	for(int i = 0; i< maskSize; i++)
	//	{
	//		for(int j = 0; j< maskSize; j++)
	//		{
	//			printf("%f ", shared[i* maskSize + j]);
	//			//m += _mask[i*fullsize + j];
	//		}
	//		printf("\n");
	//	}	
	//}	
	float cost = 1000000.0f;
	int planeIndex; float dataCost = 0; 
	//maskHalfSize = 0;
	for(unsigned int i = 0; i<numOfCandidatePlanes; i++)
	{
		int maskSize =  maskHalfSize * 2 + 1;
		for(int row = -maskHalfSize; row <= maskHalfSize; row++)	
		{
			int pos = (row + maskHalfSize) * maskSize + (maskHalfSize);
			for(int col = -maskHalfSize; col <= maskHalfSize; col++)
			{				
				dataCost += tex3D(layeredTex, x + 0.5 + row, y + 0.5 + col, i + 0.5).x * shared[pos + col] ;
			}
		}
		//float1 dataCost = tex3D(layeredTex, x + 0.5, y + 0.5, i + 0.5);
		float dataCost = tex3D(layeredTex, x + 0.5, y + 0.5, i + 0.5).x;

		if(dataCost < cost)
		{
			cost = dataCost;
			planeIndex = i;
		}
	}

	uchar4 pixelColor = tex3D(colorTex, x + 0.5, y + 0.5, planeIndex + 0.5);
	out_array[(y * imageWidth  + x ) * 4 + 0] = pixelColor.x;
	out_array[(y * imageWidth  + x ) * 4 + 1] = pixelColor.y;
	out_array[(y * imageWidth  + x ) * 4 + 2] = pixelColor.z;
	out_array[(y * imageWidth  + x ) * 4 + 3] = pixelColor.w;

	/*if(x < 340 && y<120)
	{
		out_array[(y * imageWidth  + x ) * 4 + 0] = 255;
		out_array[(y * imageWidth  + x ) * 4 + 1] = 255;
		out_array[(y * imageWidth  + x ) * 4 + 2] = 0;
		out_array[(y * imageWidth  + x ) * 4 + 3] = 255;
	}
	else
	{
		out_array[(y * imageWidth  + x ) * 4 + 0] = 255;
		out_array[(y * imageWidth  + x ) * 4 + 1] = 0;
		out_array[(y * imageWidth  + x ) * 4 + 2] = 0;
		out_array[(y * imageWidth  + x ) * 4 + 3] = 255;
	}*/
}

__global__ void writeToSurfaceColor(int width, int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < width && y < height)
	{
		for(int i = 0; i<25; i++)
		{
			uchar4 d1;
			surf3Dread(&d1, colorTex_Surface, x*4, y, i);
			uchar4 data = make_uchar4(255.0f/ float(i)*2.0, 255, 0, 255);
			surf3Dwrite(data, colorTex_Surface, x * 4, y, i);
		}
	}

}


void launchCudaProcess(cudaArray *cost3D_CUDAArray, cudaArray *color3D_CUDAArray, unsigned char *out_array, int imgWidth, int imgHeight, int numOfImages, unsigned int numOfCandidatePlanes, float *maskCUDA, int maskHalfSize)
{
	CUDA_SAFE_CALL(cudaBindTextureToArray(layeredTex, cost3D_CUDAArray));
	layeredTex.normalized = false;

	/*CUDA_SAFE_CALL(cudaBindTextureToArray(colorTex, color3D_CUDAArray));
	colorTex.normalized = false;*/

	cudaBindSurfaceToArray(colorTex_Surface, color3D_CUDAArray);


	int blockDimX = 16; int blockDimY = 16;
	dim3 block(blockDimX, blockDimY, 1); 	
    dim3 grid( (imgWidth+block.x - 1) / block.x, (imgHeight + block.y - 1) / block.y, 1);
	//std::cout<< "grad.x: " << grid.x << " grid.y: " << grid.y << std::endl;
	int sharedMem = (maskHalfSize*2+1) *(maskHalfSize*2+1) *sizeof(float);
//	cudaProcess<<<grid, block, sharedMem >>>(out_array, imgWidth, imgHeight, numOfImages, numOfCandidatePlanes, maskCUDA, maskHalfSize);
	//cudaProcess<<<grid, block >>>(out_array, imgWidth, imgHeight, numOfImages, numOfCandidatePlanes);

	writeToSurfaceColor<<<grid, block>>>( imgWidth, imgHeight);
	

	CUDA_SAFE_CALL(cudaUnbindTexture(layeredTex));

	if ( cudaSuccess != cudaGetLastError() )
	   printf( "Error!\n" );

	//CUDA_SAFE_CALL(cudaUnbindTexture(layeredTex));
	//CUDA_SAFE_CALL(cudaUnbindTexture(colorTex));

}