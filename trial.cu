#include <iostream>
#include <time.h>
#include <stdio.h>

#define gpuErrchck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


using namespace std;

__global__ void gpu_loop(int n, int *n_ret_device)
{
	int bsize = blockDim.x;
	int tid = threadIdx.x;
	int i;
	for(i = tid*(n/bsize) ; i < (tid+1)*(n/bsize) ; ++i)
		n_ret_device[i] = i;
}

void cpu_loop(int n, int *n_ret_host)
{
	int i;
	for(i=0;i<n;++i)
		n_ret_host[i] = i;
}

int main()
{

	int N = 10000000;

	int *n_ret_host = new int[N];
	int *n_ret_device = NULL;

	gpuErrchck( cudaMalloc((void**)&n_ret_device,N*sizeof(int)) );

	clock_t start, end;
	double cpu_time_used;

	start = clock();
	gpu_loop<<< 1,5 >>>(N,n_ret_device);
	gpuErrchck( cudaMemcpy((void*)n_ret_host, (void*)n_ret_device, N*sizeof(int), cudaMemcpyDeviceToHost) );
	gpuErrchck( cudaDeviceSynchronize() );
	end = clock();

	cout<<n_ret_host[1]<<endl;

	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	cout<<cpu_time_used<<endl;

	start = clock();
	cpu_loop(N,n_ret_host);
	end = clock();

	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	cout<<cpu_time_used<<endl;

	delete[] n_ret_host;
	cudaFree((void*)n_ret_device);

	return 0;
}
