// Vector Addition with Streams (Extra Credit)
// Hard deadline : Thu 26 Mar 2015 6:00 AM CST
#include	<wb.h>
#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<len) out[i]=in1[i]+in2[i];
}

int main(int argc, char ** argv) {
    // multi-stream host code
    cudaStream_t stream0,stream1,stream2,stream3;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    wbArg_t args;
    int inputLength;
    
    float *h_A,*h_B,*h_C;
    float *d_A0,*d_B0,*d_C0;    // stream 0
    float *d_A1,*d_B1,*d_C1;    //        1
    float *d_A2,*d_B2,*d_C2;    // stream 2
    float *d_A3,*d_B3,*d_C3;    //        3

    int n;
    int size;
    int SegSize;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    h_A = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    h_B = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    h_C = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    

    n=inputLength;
    SegSize=inputLength/4;
    size=n*sizeof(float);


    wbCheck(cudaMalloc((void **) &d_A0, size));
    wbCheck(cudaMalloc((void **) &d_B0, size));
    wbCheck(cudaMalloc((void **) &d_C0, size));

    wbCheck(cudaMalloc((void **) &d_A1, size));
    wbCheck(cudaMalloc((void **) &d_B1, size));
    wbCheck(cudaMalloc((void **) &d_C1, size));

    wbCheck(cudaMalloc((void **) &d_A2, size));
    wbCheck(cudaMalloc((void **) &d_B2, size));
    wbCheck(cudaMalloc((void **) &d_C2, size));

    wbCheck(cudaMalloc((void **) &d_A3, size));
    wbCheck(cudaMalloc((void **) &d_B3, size));
    wbCheck(cudaMalloc((void **) &d_C3, size));


    // dim
    dim3 DimGrid((n-1)/256+1,1,1);
    dim3 DimBlock(256,1,1);

    for(int i=0;i<n;i+=SegSize*4)
    {
        cudaMemcpyAsync(d_A0,h_A+i,SegSize*sizeof(float),cudaMemcpyHostToDevice,stream0);
        cudaMemcpyAsync(d_B0,h_B+i,SegSize*sizeof(float),cudaMemcpyHostToDevice,stream0);
        cudaMemcpyAsync(d_A1+i,h_A+i+SegSize,SegSize*sizeof(float),cudaMemcpyHostToDevice,stream1);
        cudaMemcpyAsync(d_B1+i,h_B+i+SegSize,SegSize*sizeof(float),cudaMemcpyHostToDevice,stream1);
        
        cudaMemcpyAsync(d_A2,h_A+i+2*SegSize,SegSize*sizeof(float),cudaMemcpyHostToDevice,stream2);
        cudaMemcpyAsync(d_B2,h_B+i+2*SegSize,SegSize*sizeof(float),cudaMemcpyHostToDevice,stream2);
        cudaMemcpyAsync(d_A3+i,h_A+i+3*SegSize,SegSize*sizeof(float),cudaMemcpyHostToDevice,stream3);
        cudaMemcpyAsync(d_B3+i,h_B+i+3*SegSize,SegSize*sizeof(float),cudaMemcpyHostToDevice,stream3);

        vecAdd<<<DimGrid,256,0,stream0>>>(d_A0,d_B0,d_C0,n);
        vecAdd<<<DimGrid,256,0,stream1>>>(d_A1,d_B1,d_C1,n);
        vecAdd<<<DimGrid,256,0,stream2>>>(d_A2,d_B2,d_C2,n);
        vecAdd<<<DimGrid,256,0,stream3>>>(d_A3,d_B3,d_C3,n);

        cudaMemcpyAsync(h_C+i,d_C0,SegSize*sizeof(float),cudaMemcpyDeviceToHost,stream0);
        cudaMemcpyAsync(h_C+i+SegSize,d_C1,SegSize*sizeof(float),cudaMemcpyDeviceToHost,stream1);
        cudaMemcpyAsync(h_C+i+2*SegSize,d_C2,SegSize*sizeof(float),cudaMemcpyDeviceToHost,stream2);
        cudaMemcpyAsync(h_C+i+3*SegSize,d_C3,SegSize*sizeof(float),cudaMemcpyDeviceToHost,stream3);
    }

    cudaFree(d_A0);
    cudaFree(d_B0);
    cudaFree(d_C0);

    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_C1);

    cudaFree(d_A2);
    cudaFree(d_B2);
    cudaFree(d_C2);

    cudaFree(d_A3);
    cudaFree(d_B3);
    cudaFree(d_C3);

    wbSolution(args, h_C, inputLength);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

