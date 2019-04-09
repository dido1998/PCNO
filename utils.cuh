__global__ void ComputeSquaredDiff(float* A, float* C) {
    int pixelindex = threadIdx.x;
    int imageindex = blockIdx.x;
    int pixelpos = imageindex*blockDim.x + pixelindex;
    float sum = 0;
    for (int i = 0; i < gridDim.x; i++) {
        C[gridDim.x*imageindex*blockDim.x + pixelpos]  = -expf((A[pixelpos] - A[i*blockDim.x + pixelindex])*(A[pixelpos] - A[i*blockDim.x + pixelindex])
                                                        /(2.0));

        sum+= C[gridDim.x*imageindex*blockDim.x + pixelpos];
    }


    for(int  i = 0; i < gridDim.x; i++) {
        C[gridDim.x*imageindex*blockDim.x + pixelpos]/= sum;
    }
}

__global__ void average(float* A, float* B) {
    int pixelindex = threadIdx.x;
    int imageindex = blockIdx.x;
    
}


