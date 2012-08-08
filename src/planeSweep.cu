#include <string>
#include <iostream>


//texture<uchar4, cudaTextureType3D, cudaReadModeElementType> layeredTex;
texture<float, cudaTextureType3D, cudaReadModeElementType> layeredTex;
texture<uchar4, cudaTextureType3D, cudaReadModeElementType> colorTex;


surface<void, cudaSurfaceType3D> colorTex_Surface;
surface<void, cudaSurfaceType2D> depthmap_Surface;
surface<void, cudaSurfaceType2D> depthmapView_Surface;

#define CUDA_SAFE_CALL(err) _CUDA_SAFE_CALL( err,__FILE__, __LINE__)

void _CUDA_SAFE_CALL( cudaError_t err, std::string file = __FILE__, int line = __LINE__)
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
	
	if(x < imageWidth && y < imageHeight)
	{
		float cost = 1000000.0f;		
		int planeIndex; float dataCost = 0; 
		for(unsigned int i = 0; i<numOfCandidatePlanes; i++)
		{		
			dataCost = tex3D(layeredTex, x + 0.5, y + 0.5, i + 0.5);
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
	}
}

__global__ void findDepthMap(int imageWidth, int imageHeight, unsigned int numOfCandidatePlanes,
	float near, float far, float step)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(x < imageWidth && y < imageHeight)
	{
		float cost = 1000000.0f;
		float cost2nd = 1000000.0f;
		int planeIndex; float dataCost = 0; 
		int planeIndex2nd;
		for(unsigned int i = 0; i<numOfCandidatePlanes; i++)
		{		
			dataCost = tex3D(layeredTex, x + 0.5, y + 0.5, i + 0.5);
			
			if(dataCost < cost)
			{
				cost2nd = cost;	// the smallest cost replace the second smallest cost
				planeIndex2nd = planeIndex;
				//-----------------------------------------
				cost = dataCost;
				planeIndex = i;
			}
			else if( dataCost < cost2nd)
			{
				cost2nd = dataCost;
				planeIndex2nd = i;
			}
		}
		float depth;
		//if( (cost2nd - cost)/(cost2nd + 0.00001) < -0.99 && abs(planeIndex - planeIndex2nd)>1)	// the depth is not reliable
		//{			
			//printf("cost: %f\n", (cost2nd - cost)/(cost2nd + 0.00001));
			//planeIndex = numOfCandidatePlanes - 1;		// set the index to the last plane
			//depth = far;
		//	planeIndex = 1;
		//	depth = near + 0.02;
		//}
		//else
		//{
			float d = -1.0f + step * float( planeIndex + 1);
			depth = -2 * far * near/ (d * (far - near) - (far + near));
		//}
		//printf("%u \n", planeIndex);
		surf2Dwrite( planeIndex, depthmap_Surface, x * 4, y, cudaBoundaryModeTrap);
		
		float normalizedDepth = 255.0f * (depth - near)/ (far - near);
		uchar4 depthValue = make_uchar4(normalizedDepth, normalizedDepth,normalizedDepth, 255);

		//surf2Dwrite( depthValue, depthmapView_Surface, x * 4, y, cudaBoundaryModeTrap);
	}
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


__global__ void writeDepthValue(int width, int height, float near, float far, float step)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(x < width && y < height)
	{
		int planeIdx;
		surf2Dread( &planeIdx, depthmap_Surface, x * 4, y,  cudaBoundaryModeTrap);
		float d = -1.0f + step * float( planeIdx + 1);
		float depth = -2 * far * near/ (d * (far - near) - (far + near));
		float normalizedDepth = 255.0f * (depth - near)/ (far - near);
		uchar4 depthValue = make_uchar4(normalizedDepth, normalizedDepth,normalizedDepth, 255);

		surf2Dwrite( depthValue, depthmapView_Surface, x * 4, y, cudaBoundaryModeTrap);
	}
}


void launchCudaGetDepthMap(cudaArray *cost3D_CUDAArray, cudaArray *depthmap_CUDAArray  , cudaArray *depthmapView_CUDAArray,
	int imgWidth, int imgHeight, unsigned int numOfCandidatePlanes, float near, float far, float step)
{
	// bind texture and surface 
	CUDA_SAFE_CALL(cudaBindTextureToArray(layeredTex, cost3D_CUDAArray));
 	layeredTex.normalized = false;
	CUDA_SAFE_CALL(cudaBindSurfaceToArray(depthmapView_Surface, depthmapView_CUDAArray));
	CUDA_SAFE_CALL(cudaBindSurfaceToArray(depthmap_Surface, depthmap_CUDAArray));

	

	// launch kernel
	int blockDimX = 16; int blockDimY = 16;
	dim3 block(blockDimX, blockDimY, 1); 	
    dim3 grid( (imgWidth+block.x - 1) / block.x, (imgHeight + block.y - 1) / block.y, 1);
	findDepthMap<<<grid, block >>>(imgWidth, imgHeight, numOfCandidatePlanes, near, far, step);

	CUDA_SAFE_CALL(cudaUnbindTexture(layeredTex));

}

void launchCudaWriteDepthIndexToImage(cudaArray *depthmap_CUDAArray, cudaArray *depthmapView_CUDAArray, int width, int height, float near, float far, float step)
{
	CUDA_SAFE_CALL(cudaBindSurfaceToArray(depthmapView_Surface, depthmapView_CUDAArray));
	CUDA_SAFE_CALL(cudaBindSurfaceToArray(depthmap_Surface, depthmap_CUDAArray));
	int blockDimX = 16; int blockDimY = 16;
	dim3 block(blockDimX, blockDimY, 1); 	
    dim3 grid( (width+block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);

	writeDepthValue<<<grid, block>>> (width, height, near, far , step);

}

void launchCudaProcess(cudaArray *cost3D_CUDAArray, cudaArray *color3D_CUDAArray, unsigned char *out_array, int imgWidth, int imgHeight, int numOfImages, unsigned int numOfCandidatePlanes)
{
	CUDA_SAFE_CALL(cudaBindTextureToArray(layeredTex, cost3D_CUDAArray));
	layeredTex.normalized = false;

	CUDA_SAFE_CALL(cudaBindTextureToArray(colorTex, color3D_CUDAArray));
	colorTex.normalized = false;

	//cudaBindSurfaceToArray(colorTex_Surface, color3D_CUDAArray);


	int blockDimX = 16; int blockDimY = 16;
	dim3 block(blockDimX, blockDimY, 1); 	
    dim3 grid( (imgWidth+block.x - 1) / block.x, (imgHeight + block.y - 1) / block.y, 1);

	cudaProcess<<<grid, block >>>(out_array, imgWidth, imgHeight, numOfImages, numOfCandidatePlanes);

	//writeToSurfaceColor<<<grid, block>>>( imgWidth, imgHeight);
	

	//CUDA_SAFE_CALL(cudaUnbindTexture(layeredTex));

	if ( cudaSuccess != cudaGetLastError() )
	   printf( "Error!\n" );

	CUDA_SAFE_CALL(cudaUnbindTexture(layeredTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(colorTex));

}


