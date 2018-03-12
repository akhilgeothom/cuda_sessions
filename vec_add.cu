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

__global__ void gpu_add(int n, int *a_device, int *b_device, int *c_device)
{
	int bsize = blockDim.x;
	int tid = threadIdx.x;
	int i = blockIdx.x * bsize + tid;

	if( i < n )
		c_device[i] = a_device[i] + b_device[i];
}

void cpu_add(int n, int *a_host, int *b_host, int *c_host)
{
	int i;
	for(i=0;i<n;++i)
		c_host[i] = a_host[i] + b_host[i];
}

int main()
{

	int N = 100000;

	int *a_host = new int[N];
	int *b_host = new int[N];
	int *c_host = new int[N];

	int *a_device = NULL;
	int *b_device = NULL;
	int *c_device = NULL;

	gpuErrchck( cudaMalloc((void**)&a_device,N*sizeof(int)) );
	gpuErrchck( cudaMalloc((void**)&b_device,N*sizeof(int)) );
	gpuErrchck( cudaMalloc((void**)&c_device,N*sizeof(int)) );

	int i ;

	for(i=0;i<N;++i){
		a_host[i] = 1;
		b_host[i] = 2;
	}

	clock_t start, end;
	double cpu_time_used;

	gpuErrchck( cudaMemcpy((void*)a_device, (void*)a_host, N*sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchck( cudaMemcpy((void*)b_device, (void*)b_host, N*sizeof(int), cudaMemcpyHostToDevice) );

	gpuErrchck( cudaDeviceSynchronize() );

	start = clock();

	gpu_add<<< N/1000,1000 >>>(N,a_device,b_device,c_device);

	gpuErrchck( cudaDeviceSynchronize() );

	end = clock();

	gpuErrchck( cudaMemcpy((void*)c_host, (void*)c_device, N*sizeof(int), cudaMemcpyDeviceToHost) );

	gpuErrchck( cudaDeviceSynchronize() );

	cout<<"GPU result: "<<c_host[1]<<endl;

	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	cout<<"GPU Time used: "<<cpu_time_used<<endl;

	start = clock();
	cpu_add(N,a_host,b_host,c_host);
	end = clock();

	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	cout<<"CPU Time used: "<<cpu_time_used<<endl;

	cout<<"CPU result: "<<c_host[1]<<endl;

	delete[] a_host, b_host, c_host;
	cudaFree((void*)a_device);
	cudaFree((void*)b_device);
	cudaFree((void*)c_device);

	return 0;
}
