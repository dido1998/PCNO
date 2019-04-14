extern "C" {
#include "utils.h"
}


__global__ void ReduceSum(float* A, float* B, int dim) {
    
    
    if(dim == 1)
    {
        int row = blockIdx.x;
        B[row] = 0;
        if (threadIdx.x == 0) {
            for(int  i =0;i<blockDim.x;i++) {
                
                B[row]+= A[row*blockDim.x + i]; 
            }
           
        }
    }else if(dim==0) {
        int col = threadIdx.x;
        if(blockIdx.x == 0) {
            for(int i=0;i<GridDim.x;i++) {
                B[col]+= A[i * blockIdx.x + col];
            }
        }
    }else {
        __global__ float stash_sum[gridDim.x];
        int row = blockIdx.x;
        stash_sum[row] = 0;
        if (threadIdx.x == 0) {
            for(int  i =0;i<blockDim.x;i++) {
                
                stash_sum[row]+= A[row*blockDim.x + i]; 
            }
            *B = 0;
            for(int i = 0;i<gridDim.x;i++) {
                (*B)+= stash_sum[i];
            }
           
        }
        
    }

}

__global__ void square(float* A, float *B) {
    
    B[blockIdx.x * blockDim.x + threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x]*A[blockIdx.x * blockDim.x + threadIdx.x]; 
}

__global__ void transpose(float* A, float* B)
{
   B[threadIdx.x*gridDim.x + blockIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x];
}



__global__ void dot(float* A, float* B, float* C, int col_size) {
    
    for(int i = 0; i < col_a;i++) {
        C[row*blockDim.x + threadIdx.x]+=A[row*col_size + i]*B[i*blockDim.x + threadIdx.x];
    }
}

__global__ void Add(float* A, float* B, float* C) {
   

    C[blockIdx.x * blockDim.x + threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x] + B[blockIdx.x * blockDim.x + threadIdx.x];

}

__global__ void AddAS(float* A, float* C, float scalar) {
   

    C[blockIdx.x * blockDim.x + threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x] + scalar;

}

__global__ void Sub(float* A, float* B, float* C) {
   

    C[blockIdx.x * blockDim.x + threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x] - B[blockIdx.x * blockDim.x + threadIdx.x];

}

__global__ void zeros(float* A) {

    A[blockIdx.x * blockDim.x + threadIdx.x] = 0.0;
}

__global__ void ones(float* A) {
    A[blockIdx.x * blockDim.x + threadIdx.x] = 1.0;
}

__global__ void Negative(float* A) {
    A[blockIdx.x * blockDim.x + threadIdx.x] = - A[blockIdx.x * blockDim.x + threadIdx.x];
}

__global__ void MultiplyAS(float* A, float *B, float scalar) {
   B[blockIdx.x * blockDim.x + threadIdx.x] =  A[blockIdx.x * blockDim.x + threadIdx.x] * scalar;
}

__global__ void Exp(float* A, float* B) {
    B[blockIdx.x * blockDim.x + threadIdx.x] = expf(A[blockIdx.x * blockDim.x + threadIdx.x]);
}

__global__ void MultiplyAA(float* A, float* B, float* C) {
   

    C[blockIdx.x * blockDim.x + threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x] * B[blockIdx.x * blockDim.x + threadIdx.x];

}

__global__ void DivideAS(float* A, float* B, float scalar) { 
    B[blockIdx.x * blockDim.x + threadIdx.x] =  A[blockIdx.x * blockDim.x + threadIdx.x] / scalar;
}

__global__ void MaxAS(float* A, float* B, float scalar) {
    if(A[blockIdx.x * blockDim.x + threadIdx.x]>scalar) {
        B[blockIdx.x * blockDim.x + threadIdx.x] =  A[blockIdx.x * blockDim.x + threadIdx.x];
    }else {
        B[blockIdx.x * blockDim.x + threadIdx.x] =  scalar;
    }
}

__global__ void DivideSA(float* A, float* B, float scalar) { 
    B[blockIdx.x * blockDim.x + threadIdx.x] =   scalar / A[blockIdx.x * blockDim.x + threadIdx.x] ;
}

__global__ void SetDiagonal(float* A, float scalar) {
    if(blockIdx.x == threadIdx.x) {
        A[blockIdx.x*blockDim.x+threadIdx.x] = scalar;
    }
}

__global__ void BroadcastArrayToMatrix(float* A, float* B) {
    B[blockIdx.x*blockDim.x + threadIdx.x] = A[blockIdx.x];
}

__global__ void Range(float* A, int n ) {
    A[blockIdx.x*blockDim.x + threadIdx.x] = blockIdx.x*blockDim.x + threadIdx.x;
}

__global__ void Randn(float* A) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(seed, i, 0, &state);
    A[i] = curand_uniform(&state);  
}

__global__ void Copy(float* A, float* B){
    B[blockIdx.x*blockDim.x + threadIdx.x] = A[blockIdx.x*blockDim.x + threadIdx.x];
}

__global__ void Stack(float* A, float *B){
    B[blockIdx.x*blockDim.x + threadIdx.x] = A[threadIdx];

}

__global__ void IsGreaterThan(float* A, int* B, float scalar){
    if(A[blockIdx.x*blockDim.x + threadIdx.x] > scalar) {
        B[blockIdx.x*blockDim.x + threadIdx.x] = 1;
    }else{
        B[blockIdx.x*blockDim.x + threadIdx.x] = 0
    }
}

__global__ void IsEqual(int* A, int* B, int* C){
    if(A[blockIdx.x*blockDim.x + threadIdx.x] == B[blockIdx.x*blockDim.x + threadIdx.x])
    {
        C[blockIdx.x*blockDim.x + threadIdx.x] = 1;
    }else{
        C[blockIdx.x*blockDim.x + threadIdx.x] = 0;
    }
}

__global__ void IsNotEqual(int* A, int* B, int* C){
    if(A[blockIdx.x*blockDim.x + threadIdx.x] != B[blockIdx.x*blockDim.x + threadIdx.x])
    {
        C[blockIdx.x*blockDim.x + threadIdx.x] = 1;
    }else{
        C[blockIdx.x*blockDim.x + threadIdx.x] = 0;
    }
}

__global__ SetWhereLessThan(float A[], float scalar1, float scalar2){
    if(A[blockIdx.x*blockDim.x + threadIdx.x]<scalar1) {
        B[blockIdx.x*blockDim.x + threadIdx.x] = scalar2;
    }
}


extern "C"
void ReduceSumDriver(float A[], float B[], int rowa, int cola, int dimb, int dim) {
    
    float *d_a, *d_b; 
    cudaMalloc((void **) &d_a, sizeof(float)*rowa*cola);
    cudaMalloc((void **) &d_b, sizeof(float)*dimb);
    cudaMemcpy(d_a, A, sizeof(float)*rowa*cola, cudaMemcpyHostToDevice);

    dim3 BlockDim(cola);
    dim3 GridDim(rowa);
    ReduceSum<<<GridDim, BlockDim>>>(d_a, d_b, dim);
    cudaMemcpy(B, d_b, sizeof(float)*dimb, cudaMemcpyDeviceToHost); 
    
    cudaDeviceSynchronize();
}


extern "C"
void SquareDriver(float A[], float B[], int rowa, int cola) {
    float *d_a, *d_b;
    cudaMalloc((void**)&d_a, sizeof(float)*rowa*cola);
    cudaMalloc((void**)&d_b, sizeof(float)*rowa*cola);
    cudaMemcpy(d_a, A, sizeof(float)*rowa*cola, cudaMemcpyHostToDevice);
    dim3 BlockDim(cola);
    dim3 GridDim(rowa);
    square<<<GridDim, BlockDim>>>(d_a, d_b);
    cudaMemcpy(B, d_b, sizeof(float)*rowa*cola, cudaMemcpyDeviceToHost); 
    
    cudaDeviceSynchronize();
}

extern "C"
void TransposeDriver(float A[], float B[], int rowa, int cola) {
    
    float *d_a, *d_b; 
    cudaMalloc((void **) &d_a, sizeof(float)*rowa*cola);
    cudaMalloc((void **) &d_b, sizeof(float)*rowa*cola);
    cudaMemcpy(d_a, A, sizeof(float)*rowa*cola, cudaMemcpyHostToDevice);

    dim3 BlockDim(cola);
    dim3 GridDim(rowa);
    transpose<<<GridDim, BlockDim>>>(d_a, d_b);
    cudaMemcpy(B, d_b, sizeof(float)*rowa*cola, cudaMemcpyDeviceToHost); 
    
    cudaDeviceSynchronize();
}

extern "C"
void DotDriver(float A[], float B[], float C[], int rowa, int cola, int rowb, int colb) {
    float *d_a, *d_b, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(float)*rowa*cola);
    cudaMalloc((void **) &d_b, sizeof(float)*rowb*colb);
    cudaMalloc((void **) &d_c, sizeof(float)*rowa*colb);
    cudaMemcpy(d_a, A, sizeof(float)*rowa*cola, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(float)*rowb*colb, cudaMemcpyHostToDevice);
    dim3 BlockDim(colb);
    dim3 GridDim(rowa);
    dot<<<GridDim, BlockDim>>>(d_a, d_b, cola);
    cudaMemcpy(C, d_c, sizeof(float)*rowa*colb, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
}

extern "C"
void AddDriver(float A[], float B[], float C[], int row, int col) {
    float *d_a, *d_b, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(float)*row*col);
    cudaMalloc((void **) &d_b, sizeof(float)*row*col);
    cudaMalloc((void **) &d_c, sizeof(float)*row*col);
    cudaMemcpy(d_a, A, sizeof(float)*row*col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(float)*row*col, cudaMemcpyHostToDevice);
    dim3 BlockDim(col);
    dim3 GridDim(row);
    Add<<<GridDim, BlockDim>>>(d_a, d_b, d_c);
    cudaMemcpy(C, d_c, sizeof(float)*row*col, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
}

extern "C"
void SubDriver(float A[], float B[], float C[], int row, int col) {
    float *d_a, *d_b, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(float)*row*col);
    cudaMalloc((void **) &d_b, sizeof(float)*row*col);
    cudaMalloc((void **) &d_c, sizeof(float)*row*col);
    cudaMemcpy(d_a, A, sizeof(float)*row*col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(float)*row*col, cudaMemcpyHostToDevice);
    dim3 BlockDim(col);
    dim3 GridDim(row);
    Add<<<GridDim, BlockDim>>>(d_a, d_b, d_c);
    cudaMemcpy(C, d_c, sizeof(float)*row*col, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
}

extern "C"
void ZerosDriver(float A[], int row, int col) {
    float *d_a;
    cudaMalloc((void**)&d_a, sizeof(float)*rowa*cola);
    
    dim3 BlockDim(col);
    dim3 GridDim(row);
    zeros<<<GridDim, BlockDim>>>(d_a);
    cudaMemcpy(A, d_a, sizeof(float)*row*col, cudaMemcpyDeviceToHost); 
    
    cudaDeviceSynchronize();
}

extern "C"
void OnesDriver(float A[], int row, int col) {
    float *d_a;
    cudaMalloc((void**)&d_a, sizeof(float)*row*col);
    
    dim3 BlockDim(col);
    dim3 GridDim(row);
    ones<<<GridDim, BlockDim>>>(d_a);
    cudaMemcpy(A, d_a, sizeof(float)*row*col, cudaMemcpyDeviceToHost); 
    
    cudaDeviceSynchronize();
}

extern "C"
void GetReducedRow(float A[], float B[], int row, int col, int rowtoget, int coltoremove) {
    int b_index = 0;
    for(int i = 0; i<col;i++) {
        if(i==coltoremove) {
            continue;
        }
        B[b_index++]=A[rowtoget*col + i];
    }
}



extern "C"

void NegativeDriver(float A[], int size) {
    float *d_a;
    cudaMalloc((void**)&d_a, sizeof(float)*size);
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    dim3 BlockDim(size);
    dim3 GridDim(1);
    Negative<<<GridDim, BlockDim>>>(d_a);
    cudaMemcpy(A, d_a, sizeof(float)*size, cudaMemcpyDeviceToHost); 
    
    cudaDeviceSynchronize();
}

extern "C"
void ExpDriver(float A[], float B[], int size) {
    float *d_a, *d_b;
    cudaMalloc((void**)&d_a, sizeof(float)*size);
    cudaMalloc((void**)&d_b, sizeof(float)*size);
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    dim3 BlockDim(size);
    dim3 GridDim(1);
    Exp<<<GridDim, BlockDim>>>(d_a, d_b);
    cudaMemcpy(B, d_b, sizeof(float)*size, cudaMemcpyDeviceToHost); 
    
    cudaDeviceSynchronize();
}

extern "C"
void ReduceSumDriver(float A[], float* B, int size) {
    float *d_a;
    cudaMalloc((void**)&d_a, sizeof(float)*size);
   
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    dim3 BlockDim(size);
    dim3 GridDim(1);
    ReduceSum<<<GridDim, BlockDim>>>(d_a, B, -1);
    cudaMemcpy(B, d_b, sizeof(float)*size, cudaMemcpyDeviceToHost); 
    
    cudaDeviceSynchronize();   
}


extern "C"
void MultiplyDriver(float A[], float B[], float C[], int size) {
    float *d_a, *d_b, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(float)*size);
    cudaMalloc((void **) &d_b, sizeof(float)*size);
    cudaMalloc((void **) &d_c, sizeof(float)*size);
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(float)*size, cudaMemcpyHostToDevice);
    dim3 BlockDim(size);
    dim3 GridDim(1);
    MultiplyAA<<<GridDim, BlockDim>>>(d_a, d_b, d_c);
    cudaMemcpy(C, d_c, sizeof(float)*size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
}

extern "C"
void MultiplyDriver(float A[], float B, float C[], int size) {
    float *d_a, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(float)*size);
    cudaMalloc((void **) &d_c, sizeof(float)*size);
    
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(size);
    dim3 GridDim(1);
    MultiplyAS<<<GridDim, BlockDim>>>(d_a, d_c, B );
    cudaMemcpy(C, d_c, sizeof(float)*size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
}

extern "C"
void DivideDriver(float A[], float B, float C[], int size) {
    float *d_a, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(float)*size);
    cudaMalloc((void **) &d_b, sizeof(float)*size);
    
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(size);
    dim3 GridDim(1);
    DivideAS<<<GridDim, BlockDim>>>(d_a, d_c, B );
    cudaMemcpy(C, d_c, sizeof(float)*size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
}

extern "C" 
void ReplaceRowExceptCol(float A[], float B[] int row, int col, int rowtoreplace, int colexcept) {
    int b_index = 0;
    for(int i=0;i<col;i++) {
        if(i==colexcept)
            continue;
        A[rowtoreplace*col+i] = B[b_index++];
    }
}

extern "C"
void MaxASDriver(float A[], float B, float C[], int row, int col) {
    int size = row*col;
    float *d_a, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(float)*size);
    cudaMalloc((void **) &d_b, sizeof(float)*size);
    
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(col);
    dim3 GridDim(row);
    DivideAS<<<GridDim, BlockDim>>>(d_a, d_c, B );
    cudaMemcpy(C, d_c, sizeof(float)*size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();

}

extern "C"
void DivideDriver( float B, float A[], float C[], int row, int col) {
    int size = row*col;
    float *d_a, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(float)*size);
    cudaMalloc((void **) &d_b, sizeof(float)*size);
    
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(col);
    dim3 GridDim(row);
    DivideSA<<<GridDim, BlockDim>>>(d_a, d_c, B );
    cudaMemcpy(C, d_c, sizeof(float)*size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
}


extern "C"
void SetDiagonalDriver(float A[], float B, int row, int col) {
    int size = row*col;
    float *d_a; 
    cudaMalloc((void **) &d_a, sizeof(float)*size);
    
    
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(col);
    dim3 GridDim(row);
    SetDiagonal<<<GridDim, BlockDim>>>(d_a, B );
    cudaMemcpy(C, d_c, sizeof(float)*size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
}

extern "C"
void BroadcastArrayToMatrixDriver(float A[], float B[], int row, int col) {
    int size = row*col;
    float *d_a, *d_b; 
    cudaMalloc((void **) &d_a, sizeof(float)*row);
    cudaMalloc((void**) &d_b, sizeof(float)*size)
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    dim3 BlockDim(col);
    dim3 GridDim(row);
    BroadcastArrayToMatrix<<<col, row>>>(d_a, d_b);
    cudaMemcpy(B, d_b, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

extern "C"
void RangeDriver(float A[], int size, int n) {
    
    float* d_a;
    cudaMalloc((void**)&d_a, sizeof(float)*size);
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    dim3 BlockDim(size);
    dim3 GridDim(1);
    Range<<<GridDim, BlockDim>>>(A, n);
    cudaMemcpy(A, d_a, sizeof(float)*size, cudaMemcpyDeviceToHost);
}

extern "C"
void RandnDriver(float A[], int row, int col) {
    int size = row*col;
    float *d_a;
    cudaMalloc((void**)&d_a, sizeof(float)*size);
    dim3 BlockDim(col);
    dim3 GridDim(row);
    Randn<<<GridDim, BlockDim>>>(A);
    cudaMemcpy(A, d_a, sizeof(float)*size, cudaMemcpyDeviceToHost); 
}

void MultiplyDriver(float A[], float B, float C, int row, int col) {
    int size = row*col;
    
    float *d_a, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(float)*size);
    cudaMalloc((void **) &d_c, sizeof(float)*size);
    
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(col);
    dim3 GridDim(row);
    MultiplyAS<<<GridDim, BlockDim>>>(d_a, d_c, B );
    cudaMemcpy(C, d_c, sizeof(float)*size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
}

extern "C"
void CopyDriver(float A[], float B[], int size){
    float *d_a,*d_b;
    cudaMalloc((void**)&d_a, sizeof(float)*size);
    cudaMalloc((void**)&d_b, sizeof(float)*size);
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    dim3 BlockDim(size);
    dim3 GridDim(1);
    Copy<<<GridDim, BlockDim>>>(d_a, d_b);
    cudaMemcpy(B, d_b, sizeof(float)*size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
}

extern "C"
void GetRow(float A[], float B[], int row, int col, int rownum) {
    for(int  i = 0 ;i<col;i++) {
        B[i] = A[rownum*col + i];
    }
}

extern "C"
void SetRow(float A[], float B[], int row, int col, int rownum) {
    for(int  i = 0 ;i<col;i++) {
         A[rownum*col + i] = B[i];
    }
}


extern "C"
void GetCol(float A[], float B[], int row, int col, int colnum) {
    for(int  i = 0 ;i<col;i++) {
        B[i] = A[i*col + colnum];
    }
}

extern "C"
void StackDriver(float A[], float B[], int row, int col){
    int size = row*col;    
    float *d_a,*d_b;
    cudaMalloc((void**)&d_a, sizeof(float)*size);
    cudaMalloc((void**)&d_b, sizeof(float)*size);
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    dim3 BlockDim(col);
    dim3 GridDim(row);
    Stack<<<GridDim, BlockDim>>>(d_a, d_b);
    cudaMemcpy(B, d_b, sizeof(float)*size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
}

extern "C"
void IsGreaterThanDriver(float A[], float B, int* C, int size){

    
    float *d_a;
    int *d_c 
    cudaMalloc((void **) &d_a, sizeof(float)*size);
    cudaMalloc((void **) &d_c, sizeof(int)*size);
    
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(size);
    dim3 GridDim(1);
    IsGreaterThan<<<GridDim, BlockDim>>>(d_a, d_c, B );
    cudaMemcpy(C, d_c, sizeof(int)*size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
}

extern "C"
void IsEqualDriver(int A[], int B[], int C[], int size){
    int *d_a, *d_b, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(int)*size);
    cudaMalloc((void **) &d_b, sizeof(int)*size);
    cudaMalloc((void **) &d_c, sizeof(int)*size);
    
    cudaMemcpy(d_a, A, sizeof(int)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(int)*size, cudaMemcpyHostToDevice);

    dim3 BlockDim(size);
    dim3 GridDim(1);
    IsEqual<<<GridDim, BlockDim>>>(d_a, d_b, d_c );
    cudaMemcpy(C, d_c, sizeof(int)*size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
}

extern "C"
void IsNotEqualDriver(int A[], int B[], int C[], int size){
    int *d_a, *d_b, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(int)*size);
    cudaMalloc((void **) &d_b, sizeof(int)*size);
    cudaMalloc((void **) &d_c, sizeof(int)*size);
    
    cudaMemcpy(d_a, A, sizeof(int)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(int)*size, cudaMemcpyHostToDevice);

    dim3 BlockDim(size);
    dim3 GridDim(1);
    IsNotEqual<<<GridDim, BlockDim>>>(d_a, d_b, d_c );
    cudaMemcpy(C, d_c, sizeof(int)*size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
}

extern "C"
void AddDriver(float A[], float B, float C[], int size) {
    float *d_a, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(float)*size);
    cudaMalloc((void **) &d_c, sizeof(float)*size);
    
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(size);
    dim3 GridDim(1);
    AddAS<<<GridDim, BlockDim>>>(d_a, d_c, B );
    cudaMemcpy(C, d_c, sizeof(float)*size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
}

extern "C"
void MultiplyDriver(float A[], int B[], float C[], int size){
    float B_fl[size];
    for(int  i = 0; i<size;i++ ){
        B_fl[i] = (float)B[i];
    }
    MultiplyDriver(A, B_fl, C, size);
}

extern "C"
void SetWhereLessThanDriver(float A[], float scalar1, float scalar2, int size){
    float *d_a; 
    cudaMalloc((void **) &d_a, sizeof(float)*size);
    
    
    cudaMemcpy(d_a, A, sizeof(float)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(size);
    dim3 GridDim(1);
    SetWhereLessThan<<<GridDim, BlockDim>>>(d_a, scalar1, scalar2 );
    cudaMemcpy(A, d_a, sizeof(float)*size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
}

extern "C"
void ReduceMeanDriver(float A[], float B[], int row, int col, int dimb, int dim){
    float *d_a, *d_b, *d_c;
    float C[col]; 
    cudaMalloc((void **) &d_a, sizeof(float)*row*col);
    cudaMalloc((void **) &d_b, sizeof(float)*dimb);
    cudaMalloc((void **) &d_c, sizeof(float)*dimb);
    cudaMemcpy(d_a, A, sizeof(float)*rowa*cola, cudaMemcpyHostToDevice);

    dim3 BlockDim(col);
    dim3 GridDim(row);
    ReduceSum<<<GridDim, BlockDim>>>(d_a, d_b, dim);
     
    DivideAS<<<1, col>>>(d_b, d_c, row);
    cudaMemcpy(B, d_c, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
