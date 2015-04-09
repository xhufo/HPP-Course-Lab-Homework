// Histogram Equalization
// Hard deadline : Thu 19 Mar 2015 6:00 AM CST 
#include    <wb.h>

#define HISTOGRAM_LENGTH 256
#define Block_width 16
#define Block_height 16

#define clamp(x, start, end) (min((float)max((float)(x), (float)start), (float)end))    
#define correct_color(val) clamp(255*(cdf[val]-cdfmin)/(1-cdfmin),0,255)
//@@ insert code here
#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void rbg2gray(unsigned char *ucharImage,unsigned char *grayImage,int width,int height)
{
    unsigned int tx;
    unsigned int ty;
    unsigned int idx;

    int r;
    int g;
    int b;

    tx=blockIdx.x*blockDim.x+threadIdx.x;
    ty=blockIdx.y*blockDim.y+threadIdx.y;
    idx=ty*width+tx;

    if((tx<width)&&(ty<height))
    {
        r=ucharImage[3*idx];
        g=ucharImage[3*idx+1];
        b=ucharImage[3*idx+2];
        grayImage[idx]=(unsigned char)(0.21*r+0.71*g+0.07*b);
    }
    if(tx<10&&ty==0)
        printf("grayImage[%d]=%d\n",tx,grayImage[tx]);
}

__global__ void histogram_kernel(unsigned char *buffer, long size, unsigned int *histo)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    // stride is total number of threads
    int stride=blockDim.x*gridDim.x;

    if(i<80&&i>=70)
        printf("buffer[%d]=%d\n",i,buffer[i]);
    // All threads handle blockDim.x*griDim.x
     // consecutive elements
    while(i<size)
    {
        atomicAdd(&histo[buffer[i]],1);
        i+=stride;
    }
}

// float correct_color(float val,int cdfmin)
// {
//  return clamp(255*(cdf[val]-cdfmin)/(1-cdfmin),0,255);
// }

int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;

    //@@ Insert more code here
    unsigned char *hostUcharImage;
    unsigned char *deviceUcharImage;
    unsigned char *deviceGrayImage;

    unsigned int *hostHistogram;
    unsigned int *deviceHistogram;

    float *cdf;
    float cdfmin;

    int i;
    int numElem;
    int numPix;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here
    numElem=imageHeight*imageWidth;
    numPix=imageHeight*imageWidth*imageChannels;
    wbLog(TRACE, "The number of pix is ", numPix);
    wbLog(TRACE, "The number of element is ", numElem);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    // cast the image from float to unsigned char
    hostUcharImage = (unsigned char *) malloc(numPix * sizeof(unsigned char));
    for(i=0;i<numPix;i++)
    {
        hostUcharImage[i]=(unsigned char)(255*hostInputImageData[i]);
    }
    for(i=0;i<10;i++)
        printf("hostUcharImage[%d]=%d\n",i,hostUcharImage[i]);

    // convert the image from RGB to GrayScal
    wbTime_start(GPU, "Convert the image from RGB to GrayScal.");
    wbCheck(cudaMalloc((void **) &deviceUcharImage, numPix*sizeof(unsigned char)));
    wbCheck(cudaMalloc((void **) &deviceGrayImage, numElem*sizeof(unsigned char)));
    cudaMemcpy(deviceUcharImage,
               hostUcharImage,
               numPix*sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    dim3 dimBlock(Block_width,Block_height,1);
    dim3 dimGrid((imageWidth-1)/Block_width+1,(imageHeight-1)/Block_height+1,1);
    wbLog(TRACE, "The number of block is ", (imageWidth-1)/Block_width+1);
    wbLog(TRACE, "The number of block is ", (imageHeight-1)/Block_width+1);

    rbg2gray<<<dimGrid,dimBlock>>>(deviceUcharImage,deviceGrayImage,imageWidth,imageHeight);
    wbTime_stop(GPU, "Convert the image from RGB to GrayScal.");

    // compute the histogram of grayImage
    wbTime_start(GPU, "Compute the histogram of grayImage.");    
    wbCheck(cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH*sizeof(unsigned int)));
    dim3 dimBlock1(Block_width,1,1);
//    dim3 dimGrid1((numElem-1)/Block_width+1,1,1);
    dim3 dimGrid1(32,1,1);
    wbLog(TRACE, "The number of block is ", (numElem-1)/Block_width+1);

    histogram_kernel<<<dimGrid1,dimBlock1>>>(deviceGrayImage,numElem,deviceHistogram);
    wbTime_stop(GPU, "Compute the histogram of grayImage."); 

    // compute the Cumulative Distribution Function of histogram
    hostHistogram = (unsigned int *) malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));
    cdf = (float *) malloc(HISTOGRAM_LENGTH * sizeof(float));
    cudaMemcpy(hostHistogram,
               deviceHistogram,
               HISTOGRAM_LENGTH*sizeof(unsigned int),
               cudaMemcpyDeviceToHost);

    // debug
    for(i=70;i<80;i++)
        printf("histo[%d]=%d\n",i,hostHistogram[i]);

    cdf[0]=1.0*hostHistogram[0]/(numElem);
    for(i=1;i<HISTOGRAM_LENGTH;i++)
    {
        cdf[i] = cdf[i-1] + 1.0*hostHistogram[i]/(numElem);
    }
    // int * 1.0 --> float
    for(i=0;i<80;i++)
        printf("cdf[%d]=%f\n",i,cdf[i]);
    // compute the minimum value of the CDF
    cdfmin=cdf[0];
    for(i=0;i<HISTOGRAM_LENGTH;i++)
    {
        cdfmin=min(cdfmin,cdf[i]);
    }

    // define the histogram equalization function 
    // #define before 

    // apply the histogram equalization function 
    for(i=0;i<numPix;i++)
    {
        hostUcharImage[i]=correct_color(hostUcharImage[i]);
    }

    // cast back to float 
    for(i=0;i<numPix;i++)
    {
        hostOutputImageData[i]=(float)(hostUcharImage[i]/255.0);
    }

    wbSolution(args, outputImage);

    //@@ insert code here
    cudaFree(deviceUcharImage);
    cudaFree(deviceGrayImage);
    cudaFree(deviceHistogram);

    free(hostUcharImage);
    free(hostHistogram);
    free(cdf);
    return 0;
}

