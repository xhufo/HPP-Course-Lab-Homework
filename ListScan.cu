// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}
// hard due time : Wednesday, March 11, 2015
#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void layer3(float *input,float *S,int len)
{
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;
    if (blockIdx.x)
    {
       if (start + t < len)
          input[start + t] += S[blockIdx.x - 1];
       if (start + BLOCK_SIZE + t < len)
          input[start + BLOCK_SIZE + t] += S[blockIdx.x - 1];
    }
}

__global__ void scan(float *input, float *output, float *S, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
    __shared__ float XY[2*BLOCK_SIZE];

    unsigned int t=threadIdx.x;
    unsigned int start=2*blockIdx.x*BLOCK_SIZE;
    int stride;
    int index;

    // load data to share memory
    if(start+t<len)
    {
        XY[t]=input[start+t]; 
    }
    else
    {
        XY[t]=0.0f;
    }
    if(start+BLOCK_SIZE+t<len)
    {
        XY[BLOCK_SIZE+t]=input[start+BLOCK_SIZE+t]; 
    }
    else
    {
        XY[BLOCK_SIZE+t]=0.0f;
    }
    __syncthreads();

    // reduction phase 
    for(stride=1;stride<=BLOCK_SIZE;stride*=2)
    {
        index=(threadIdx.x+1)*stride*2-1;
        if(index<2*BLOCK_SIZE)
            XY[index]+=XY[index-stride];
        __syncthreads();
    }
    // post reduction reverse phase
    for(stride=BLOCK_SIZE/2;stride>0;stride/=2)
    {
        __syncthreads();
        index=(threadIdx.x+1)*stride*2-1;
        if(index+stride<2*BLOCK_SIZE)
        {
            XY[index+stride]+=XY[index];
        }
    }
    __syncthreads();
    if(S==NULL)
    {
        ;
    }
    else if(t==0)
        S[blockIdx.x]=XY[2*BLOCK_SIZE-1];

    // output
    if(start+t<len)
        output[start+t]=XY[t];
    if(start+BLOCK_SIZE+t<len)
        output[start+BLOCK_SIZE+t]=XY[t+BLOCK_SIZE];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    float * deviceAuxin;
    float * deviceAuxout;
    int numElements; // number of elements in the list
    int numAux;
    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    
    numAux=2*BLOCK_SIZE;
    //(numElements-1)/(2*BLOCK_SIZE)+1;

    wbCheck(cudaMalloc((void**)&deviceAuxin, numAux*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceAuxout, numAux*sizeof(float)));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbCheck(cudaMemset(deviceAuxin, 0, numAux*sizeof(float)));
    wbCheck(cudaMemset(deviceAuxout, 0, numAux*sizeof(float)));    
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGrid((numElements-1)/(2*BLOCK_SIZE)+1,1,1); 
    wbLog(TRACE, "The number of blocks is ", (numElements-1)/(2*BLOCK_SIZE)+1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    scan<<<dimGrid,dimBlock>>>(deviceInput,deviceOutput,deviceAuxin,numElements);
    cudaDeviceSynchronize();
    scan<<<dim3(1,1,1),dimBlock>>>(deviceAuxin,deviceAuxout,NULL,numAux);
    cudaDeviceSynchronize();
    layer3<<<dimGrid,dimBlock>>>(deviceOutput,deviceAuxout,numElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(deviceAuxin);
    cudaFree(deviceAuxout);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

