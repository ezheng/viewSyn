#include <string>
#include <iostream>


//texture<uchar4, cudaTextureType3D, cudaReadModeElementType> layeredTex;
texture<float1, cudaTextureType3D, cudaReadModeElementType> layeredTex;
texture<uchar4, cudaTextureType3D, cudaReadModeElementType> colorTex;

void CUDA_SAFE_CALL( cudaError_t err, std::string file = __FILE__, int line = __LINE__)
{
	if (err != cudaSuccess) {
		std::cout<< cudaGetErrorString( err ) << " in file: " << file << " at line: " << line << std::endl;
        //printf( "%s in %s at line %i\n", cudaGetErrorString( err ),
          //      file.c_str(), line );
        exit( EXIT_FAILURE );
    }
}


__global__ void cudaProcess(unsigned char *out_array, int imageWidth, int imageHeight, int numOfImages, unsigned int numOfCandidatePlanes)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	//uchar4 data;
	//data = tex3D(layeredTex, x, y, 16);
	//float1 data = tex3D(layeredTex, x, y, 16);
	float cost = 1000000.0f;
	int planeIndex;
	for(unsigned int i = 0; i<numOfCandidatePlanes; i++)
	{
		float1 dataCost = tex3D(layeredTex, x + 0.5, y + 0.5, i + 0.5);
		
		if(dataCost.x < cost)
		{
			cost = dataCost.x;
			planeIndex = i;
		}		
	}
	out_array[y * imageWidth + x] = planeIndex;
	uchar4 pixelColor = tex3D(colorTex, x, y, planeIndex);
	out_array[(y * imageWidth  + x ) * 4 + 0] = pixelColor.x;
	out_array[(y * imageWidth  + x ) * 4 + 1] = pixelColor.y;
	out_array[(y * imageWidth  + x ) * 4 + 2] = pixelColor.z;
	out_array[(y * imageWidth  + x ) * 4 + 3] = pixelColor.w;
}


void launchCudaProcess(cudaArray *cost3D_CUDAArray, cudaArray *color3D_CUDAArray, unsigned char *out_array, int imgWidth, int imgHeight, int numOfImages, unsigned int numOfCandidatePlanes)
{

	CUDA_SAFE_CALL(cudaBindTextureToArray(layeredTex, cost3D_CUDAArray));
	layeredTex.normalized = false;

	CUDA_SAFE_CALL(cudaBindTextureToArray(colorTex, color3D_CUDAArray));
	colorTex.normalized = false;


	int blockDimX = 16; int blockDimY = 16;
	dim3 block(blockDimX, blockDimY, 1); 	
    dim3 grid( (imgWidth+block.x - 1) / block.x, (imgHeight + block.y - 1) / block.y, 1);

	cudaProcess<<<grid, block>>>(out_array, imgWidth, imgHeight, numOfImages, numOfCandidatePlanes);
	
	if ( cudaSuccess != cudaGetLastError() )
	   printf( "Error!\n" );

	//CUDA_SAFE_CALL(cudaUnbindTexture(layeredTex));
	//CUDA_SAFE_CALL(cudaUnbindTexture(colorTex));

}