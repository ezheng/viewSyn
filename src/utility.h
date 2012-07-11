#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

namespace{
#define CUDA_SAFE_CALL(err) _CUDA_SAFE_CALL(err, __FILE__, __LINE__)
void _CUDA_SAFE_CALL( cudaError_t err, std::string file, int line)
{
   if (err != cudaSuccess) {
       // printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
       //         file, line );
	   std::cout<< cudaGetErrorString( err ) << " in file: " << file << " at line: " << line << std::endl;
        exit( EXIT_FAILURE );
    }
}
}

class cudaTimer
{
public:
	cudaTimer()
	{
		CUDA_SAFE_CALL( cudaEventCreate( &_start));
		CUDA_SAFE_CALL( cudaEventCreate( &_stop));
	}

	~cudaTimer()
	{
		CUDA_SAFE_CALL( cudaEventDestroy(_start));
		CUDA_SAFE_CALL( cudaEventDestroy(_stop));
	}
	
	void start()
	{
		CUDA_SAFE_CALL( cudaEventRecord( _start, 0 ));
	}

	void stop()
	{
		CUDA_SAFE_CALL( cudaEventRecord( _stop, 0));
		CUDA_SAFE_CALL( cudaEventSynchronize( _stop));

		float elapsedTime;
		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime, _start, _stop));

		std::cout<< "kernel takes: " << elapsedTime << std::endl;
	}

private:
	cudaEvent_t _start;
	cudaEvent_t	_stop;




};




#endif