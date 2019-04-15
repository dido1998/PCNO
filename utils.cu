
#include "utils.h"

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

__global__ void ReduceSum(double* A, double* B, int dim) {
    
    
    if(dim == 1)
    {
        int row = blockIdx.x;
        B[row] = 0;
        if (threadIdx.x == 0) {
            for(int  i =0;i<blockDim.x;i++) {
                
                B[row]+= A[row*blockDim.x + i]; 
            }
           
        }
    }

}

__global__ void square(double* A, double *B) {
    
    B[blockIdx.x * blockDim.x + threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x]*A[blockIdx.x * blockDim.x + threadIdx.x]; 
}

__global__ void transpose(double* A, double* B)
{
   B[threadIdx.x*gridDim.x + blockIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x];
}



__global__ void dot(double* A, double* B, double* C, int col_size) {
    
    for(int i = 0; i < col_size;i++) {
        C[blockIdx.x*blockDim.x + threadIdx.x]+=A[blockIdx.x*col_size + i]*B[i*blockDim.x + threadIdx.x];
    }
}

__global__ void Add(double* A, double* B, double* C) {
   

    C[blockIdx.x * blockDim.x + threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x] + B[blockIdx.x * blockDim.x + threadIdx.x];

}

__global__ void AddAS(double* A, double* C, double scalar) {
   

    C[blockIdx.x * blockDim.x + threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x] + scalar;

}

__global__ void Sub(double* A, double* B, double* C) {
   

    C[blockIdx.x * blockDim.x + threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x] - B[blockIdx.x * blockDim.x + threadIdx.x];

}

__global__ void zeros(double* A) {

    A[blockIdx.x * blockDim.x + threadIdx.x] = 0.0;
}

__global__ void ones(double* A) {
    A[blockIdx.x * blockDim.x + threadIdx.x] = 1.0;
}

__global__ void Negative(double* A) {
    A[blockIdx.x * blockDim.x + threadIdx.x] = - A[blockIdx.x * blockDim.x + threadIdx.x];
}

__global__ void MultiplyAS(double* A, double *B, double scalar) {
   B[blockIdx.x * blockDim.x + threadIdx.x] =  A[blockIdx.x * blockDim.x + threadIdx.x] * scalar;
}

__global__ void Exp(double* A, double* B) {
    B[blockIdx.x * blockDim.x + threadIdx.x] = exp(A[blockIdx.x * blockDim.x + threadIdx.x]);
}

__global__ void Log(double* A, double* B) {
    B[blockIdx.x * blockDim.x + threadIdx.x] = log(A[blockIdx.x * blockDim.x + threadIdx.x]);
}

__global__ void MultiplyAA(double* A, double* B, double* C) {
   

    C[blockIdx.x * blockDim.x + threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x] * B[blockIdx.x * blockDim.x + threadIdx.x];

}

__global__ void DivideAS(double* A, double* B, double scalar) { 

    B[blockIdx.x * blockDim.x + threadIdx.x] =  A[blockIdx.x * blockDim.x + threadIdx.x] / scalar;
}

__global__ void Divide(double* A, double* B, double* C) { 
    C[blockIdx.x * blockDim.x + threadIdx.x] =  A[blockIdx.x * blockDim.x + threadIdx.x] / B[blockIdx.x * blockDim.x + threadIdx.x];
}

__global__ void MaxAS(double* A, double* B, double scalar) {
    if(A[blockIdx.x * blockDim.x + threadIdx.x]>scalar) {
        B[blockIdx.x * blockDim.x + threadIdx.x] =  A[blockIdx.x * blockDim.x + threadIdx.x];
    }else {
        B[blockIdx.x * blockDim.x + threadIdx.x] =  scalar;
    }
}

__global__ void DivideSA(double* A, double* B, double scalar) { 
    B[blockIdx.x * blockDim.x + threadIdx.x] =   scalar / A[blockIdx.x * blockDim.x + threadIdx.x] ;
}

__global__ void SetDiagonal(double* A, double scalar) {
    if(blockIdx.x == threadIdx.x) {
        A[blockIdx.x*blockDim.x+threadIdx.x] = scalar;
    }
}

__global__ void BroadcastArrayToMatrix(double* A, double* B) {
    
    B[blockIdx.x*blockDim.x + threadIdx.x] = A[threadIdx.x];
}

__global__ void Range(double* A, int n ) {
    A[blockIdx.x*blockDim.x + threadIdx.x] = blockIdx.x*blockDim.x + threadIdx.x;
}

__global__ void Randn(double* A) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(4, i, 0, &state);
    A[i] = curand_uniform(&state);  
}

__global__ void Copy(double* A, double* B){
    B[blockIdx.x*blockDim.x + threadIdx.x] = A[blockIdx.x*blockDim.x + threadIdx.x];
}

__global__ void Stack(double* A, double* B){
    B[blockIdx.x*blockDim.x + threadIdx.x] = A[threadIdx.x];

}

__global__ void IsGreaterThan(double* A, int* B, double scalar){
    if(A[blockIdx.x*blockDim.x + threadIdx.x] > scalar) {
        B[blockIdx.x*blockDim.x + threadIdx.x] = 1;
    }else{
        B[blockIdx.x*blockDim.x + threadIdx.x] = 0;
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

__global__ void SetWhereLessThan(double* A, double scalar1, double scalar2){
    if(A[blockIdx.x*blockDim.x + threadIdx.x]<scalar1) {
        A[blockIdx.x*blockDim.x + threadIdx.x] = scalar2;
    }
}



void ReduceSumDriver(double A[], double B[], int rowa, int cola, int dimb, int dim) {
    
    double *d_a, *d_b; 
    cudaMalloc((void **) &d_a, sizeof(double)*rowa*cola);
    cudaMalloc((void **) &d_b, sizeof(double)*dimb);
    cudaMemcpy(d_a, A, sizeof(double)*rowa*cola, cudaMemcpyHostToDevice);

    dim3 BlockDim(cola);
    dim3 GridDim(rowa);
    ReduceSum<<<GridDim, BlockDim>>>(d_a, d_b, dim);
    cudaMemcpy(B, d_b, sizeof(double)*dimb, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaDeviceSynchronize();
}



void SquareDriver(double A[], double B[], int rowa, int cola) {
    double *d_a, *d_b;
    cudaMalloc((void**)&d_a, sizeof(double)*rowa*cola);
    cudaMalloc((void**)&d_b, sizeof(double)*rowa*cola);
    cudaMemcpy(d_a, A, sizeof(double)*rowa*cola, cudaMemcpyHostToDevice);
    dim3 BlockDim(cola);
    dim3 GridDim(rowa);
    square<<<GridDim, BlockDim>>>(d_a, d_b);
    cudaMemcpy(B, d_b, sizeof(double)*rowa*cola, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaDeviceSynchronize();
}


void TransposeDriver(double A[], double B[], int rowa, int cola) {
    
    double *d_a, *d_b; 
    cudaMalloc((void **) &d_a, sizeof(double)*rowa*cola);
    cudaMalloc((void **) &d_b, sizeof(double)*rowa*cola);
    cudaMemcpy(d_a, A, sizeof(double)*rowa*cola, cudaMemcpyHostToDevice);

    dim3 BlockDim(cola);
    dim3 GridDim(rowa);
    transpose<<<GridDim, BlockDim>>>(d_a, d_b);
    cudaMemcpy(B, d_b, sizeof(double)*rowa*cola, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaDeviceSynchronize();
}


void DotDriver(double A[], double B[], double C[], int rowa, int cola, int rowb, int colb) {
    double *d_a, *d_b, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(double)*rowa*cola);
    cudaMalloc((void **) &d_b, sizeof(double)*rowb*colb);
    cudaMalloc((void **) &d_c, sizeof(double)*rowa*colb);
    cudaMemcpy(d_a, A, sizeof(double)*rowa*cola, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(double)*rowb*colb, cudaMemcpyHostToDevice);
    dim3 BlockDim(colb);
    dim3 GridDim(rowa);
    dot<<<GridDim, BlockDim>>>(d_a, d_b, d_c, cola);
    cudaMemcpy(C, d_c, sizeof(double)*rowa*colb, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceSynchronize();
}


void AddDriver(double A[], double B[], double C[], int row, int col) {

    double *d_a, *d_b, *d_c; 

    cudaMalloc((void **) &d_a, sizeof(double)*row*col);
    cudaMalloc((void **) &d_b, sizeof(double)*row*col);
    cudaMalloc((void **) &d_c, sizeof(double)*row*col);

    cudaMemcpy(d_a, A, sizeof(double)*row*col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(double)*row*col, cudaMemcpyHostToDevice);

    dim3 BlockDim(col);
    dim3 GridDim(row);
    Add<<<GridDim, BlockDim>>>(d_a, d_b, d_c);

    cudaMemcpy(C, d_c, sizeof(double)*row*col, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceSynchronize();
}


void SubDriver(double A[], double B[], double C[], int row, int col) {
    
    double *d_a, *d_b, *d_c; 
    

    cudaMalloc((void **) &d_a, sizeof(double)*row*col);
    cudaMalloc((void **) &d_b, sizeof(double)*row*col);
    cudaMalloc((void **) &d_c, sizeof(double)*row*col);
   
    cudaMemcpy(d_a, A, sizeof(double)*row*col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(double)*row*col, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(col);
    dim3 GridDim(row);
    Sub<<<GridDim, BlockDim>>>(d_a, d_b, d_c);

    cudaMemcpy(C, d_c, sizeof(double)*row*col, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceSynchronize();
}


void ZerosDriver(double A[], int row, int col) {
    double *d_a;
    cudaMalloc((void**)&d_a, sizeof(double)*row*col);
    
    dim3 BlockDim(col);
    dim3 GridDim(row);
    zeros<<<GridDim, BlockDim>>>(d_a);
    cudaMemcpy(A, d_a, sizeof(double)*row*col, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaDeviceSynchronize();
}


void OnesDriver(double A[], int row, int col) {
    double *d_a;
    cudaMalloc((void**)&d_a, sizeof(double)*row*col);
    
    dim3 BlockDim(col);
    dim3 GridDim(row);
    ones<<<GridDim, BlockDim>>>(d_a);
    cudaMemcpy(A, d_a, sizeof(double)*row*col, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaDeviceSynchronize();
}


void GetReducedRow(double A[], double B[], int row, int col, int rowtoget, int coltoremove) {
    int b_index = 0;
    for(int i = 0; i<col;i++) {
        if(i==coltoremove) {
            continue;
        }
        B[b_index++]=A[rowtoget*col + i];
    }
}





void NegativeDriver(double A[], int size) {
    double *d_a;
    cudaMalloc((void**)&d_a, sizeof(double)*size);
    cudaMemcpy(d_a, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    dim3 BlockDim(1);
    dim3 GridDim(size);
    Negative<<<GridDim, BlockDim>>>(d_a);
    cudaMemcpy(A, d_a, sizeof(double)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaDeviceSynchronize();
}


void ExpDriver(double A[], double B[], int size) {
    double *d_a, *d_b;
    cudaMalloc((void**)&d_a, sizeof(double)*size);
    cudaMalloc((void**)&d_b, sizeof(double)*size);
    cudaMemcpy(d_a, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    dim3 BlockDim(1);
    dim3 GridDim(size);
    Exp<<<GridDim, BlockDim>>>(d_a, d_b);
    cudaMemcpy(B, d_b, sizeof(double)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaDeviceSynchronize();

}


void LogDriver(double A[], double B[], int size) {
    double *d_a, *d_b;
    cudaMalloc((void**)&d_a, sizeof(double)*size);
    cudaMalloc((void**)&d_b, sizeof(double)*size);
    cudaMemcpy(d_a, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    dim3 BlockDim(1);
    dim3 GridDim(size);
    Log<<<GridDim, BlockDim>>>(d_a, d_b);
    cudaMemcpy(B, d_b, sizeof(double)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaDeviceSynchronize();
}


void ReduceSumDriver(double A[], double* B, int size) {
    for(int i = 0;i<size;i++){
        (*B)+=A[i];
    }  
}



void MultiplyDriver(double A[], double B[], double C[], int size) {
    double *d_a, *d_b, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(double)*size);
    cudaMalloc((void **) &d_b, sizeof(double)*size);
    cudaMalloc((void **) &d_c, sizeof(double)*size);
    cudaMemcpy(d_a, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(double)*size, cudaMemcpyHostToDevice);
    dim3 BlockDim(1);
    dim3 GridDim(size);
    MultiplyAA<<<GridDim, BlockDim>>>(d_a, d_b, d_c);
    cudaMemcpy(C, d_c, sizeof(double)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceSynchronize();
}


void MultiplyDriver(double A[], double B, double C[], int size) {
    double *d_a, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(double)*size);
    cudaMalloc((void **) &d_c, sizeof(double)*size);
    
    cudaMemcpy(d_a, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(1);
    dim3 GridDim(size);
    MultiplyAS<<<GridDim, BlockDim>>>(d_a, d_c, B );
    cudaMemcpy(C, d_c, sizeof(double)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_c);
    cudaDeviceSynchronize();
}


void DivideDriver(double A[], double B, double C[], int size) {
    double *d_a, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(double)*size);
    cudaMalloc((void **) &d_c, sizeof(double)*size);
    
    cudaMemcpy(d_a, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(1);
    dim3 GridDim(size);
    DivideAS<<<GridDim, BlockDim>>>(d_a, d_c, B );
    cudaMemcpy(C, d_c, sizeof(double)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_c);
    cudaDeviceSynchronize();
}

 
void ReplaceRowExceptCol(double A[], double B[], int row, int col, int rowtoreplace, int colexcept) {
    int b_index = 0;
    for(int i=0;i<col;i++) {
        if(i==colexcept)
            continue;
        A[rowtoreplace*col+i] = B[b_index++];
    }
}


void MaxASDriver(double A[], double B, double C[], int row, int col) {
    int size = row*col;
    double *d_a, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(double)*size);
    cudaMalloc((void **) &d_c, sizeof(double)*size);
    
    cudaMemcpy(d_a, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(col);
    dim3 GridDim(row);
    MaxAS<<<GridDim, BlockDim>>>(d_a, d_c, B );
    cudaMemcpy(C, d_c, sizeof(double)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_c);
    cudaDeviceSynchronize();

}


void DivideDriver( double B, double A[], double C[], int row, int col) {
    int size = row*col;
    double *d_a, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(double)*size);
    cudaMalloc((void **) &d_c, sizeof(double)*size);
    
    cudaMemcpy(d_a, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(col);
    dim3 GridDim(row);
    DivideSA<<<GridDim, BlockDim>>>(d_a, d_c, B );
    cudaMemcpy(C, d_c, sizeof(double)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_c);
    cudaDeviceSynchronize();
}



void SetDiagonalDriver(double A[], double B, int row, int col) {
    int size = row*col;
    double *d_a; 
    cudaMalloc((void **) &d_a, sizeof(double)*size);
    
    
    cudaMemcpy(d_a, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(col);
    dim3 GridDim(row);
    SetDiagonal<<<GridDim, BlockDim>>>(d_a, B );
    cudaMemcpy(A, d_a, sizeof(double)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaDeviceSynchronize();
}


void BroadcastArrayToMatrixDriver(double A[], double B[], int row, int col) {
    int size = row*col;
    double *d_a, *d_b; 

    cudaMalloc((void **) &d_a, sizeof(double)*col);
    cudaMalloc((void**) &d_b, sizeof(double)*size);
    cudaMemcpy(d_a, A, sizeof(double)*col, cudaMemcpyHostToDevice);
    dim3 BlockDim(col);
    dim3 GridDim(row);
    BroadcastArrayToMatrix<<<GridDim, BlockDim>>>(d_a, d_b);
    cudaMemcpy(B, d_b, sizeof(double)*size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaDeviceSynchronize();
}


void RangeDriver(double A[], int size, int n) {
    
    double* d_a;
    cudaMalloc((void**)&d_a, sizeof(double)*size);
    dim3 BlockDim(size);
    dim3 GridDim(1);
    Range<<<GridDim, BlockDim>>>(d_a, n);
    cudaMemcpy(A, d_a, sizeof(double)*size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
}


void RandnDriver(double A[], int row, int col) {
    int size = row*col;
    double *d_a;
    cudaMalloc((void**)&d_a, sizeof(double)*size);
    dim3 BlockDim(col);
    dim3 GridDim(row);
    Randn<<<GridDim, BlockDim>>>(d_a);
    cudaFree(d_a);
    cudaMemcpy(A, d_a, sizeof(double)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
}

void MultiplyDriver(double A[], double B, double C[], int row, int col) {
    int size = row*col;
    
    double *d_a, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(double)*size);
    cudaMalloc((void **) &d_c, sizeof(double)*size);
    
    cudaMemcpy(d_a, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(col);
    dim3 GridDim(row);
    MultiplyAS<<<GridDim, BlockDim>>>(d_a, d_c, B );
    cudaMemcpy(C, d_c, sizeof(double)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_c);
    cudaDeviceSynchronize();
}


void CopyDriver(double A[], double B[], int size){
    double *d_a,*d_b;
    cudaMalloc((void**)&d_a, sizeof(double)*size);
    cudaMalloc((void**)&d_b, sizeof(double)*size);
    cudaMemcpy(d_a, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    dim3 BlockDim(1);
    dim3 GridDim(size);
    Copy<<<GridDim, BlockDim>>>(d_a, d_b);
    cudaMemcpy(B, d_b, sizeof(double)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaDeviceSynchronize();
}


void GetRow(double A[], double B[], int row, int col, int rownum) {
    for(int  i = 0 ;i<col;i++) {
        B[i] = A[rownum*col + i];
    }
}


void SetRow(double A[], double B[], int row, int col, int rownum) {
    for(int  i = 0 ;i<col;i++) {
         A[rownum*col + i] = B[i];
    }
}



void GetCol(double A[], double B[], int row, int col, int colnum) {
    for(int  i = 0 ;i<row;i++) {
        B[i] = A[i*col + colnum];
    }
}


void StackDriver(double A[], double B[], int row, int col){
    int size = row*col;    
    double *d_a,*d_b;
    cudaMalloc((void**)&d_a, sizeof(double)*size);
    cudaMalloc((void**)&d_b, sizeof(double)*size);
    cudaMemcpy(d_a, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    dim3 BlockDim(col);
    dim3 GridDim(row);
    Stack<<<GridDim, BlockDim>>>(d_a, d_b);
    cudaMemcpy(B, d_b, sizeof(double)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaDeviceSynchronize();
}


void IsGreaterThanDriver(double A[], double B, int* C, int size){

    
    double *d_a;
    int *d_c;
    cudaMalloc((void **) &d_a, sizeof(double)*size);
    cudaMalloc((void **) &d_c, sizeof(int)*size);
    
    cudaMemcpy(d_a, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(1);
    dim3 GridDim(size);
    IsGreaterThan<<<GridDim, BlockDim>>>(d_a, d_c, B );
    cudaMemcpy(C, d_c, sizeof(int)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_c);
    cudaDeviceSynchronize();
}


void IsEqualDriver(int A[], int B[], int C[], int size){
    int *d_a, *d_b, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(int)*size);
    cudaMalloc((void **) &d_b, sizeof(int)*size);
    cudaMalloc((void **) &d_c, sizeof(int)*size);
    
    cudaMemcpy(d_a, A, sizeof(int)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(int)*size, cudaMemcpyHostToDevice);

    dim3 BlockDim(1);
    dim3 GridDim(size);
    IsEqual<<<GridDim, BlockDim>>>(d_a, d_b, d_c );
    cudaMemcpy(C, d_c, sizeof(int)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceSynchronize();
}


void IsNotEqualDriver(int A[], int B[], int C[], int size){
    int *d_a, *d_b, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(int)*size);
    cudaMalloc((void **) &d_b, sizeof(int)*size);
    cudaMalloc((void **) &d_c, sizeof(int)*size);
    
    cudaMemcpy(d_a, A, sizeof(int)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(int)*size, cudaMemcpyHostToDevice);

    dim3 BlockDim(1);
    dim3 GridDim(size);
    IsNotEqual<<<GridDim, BlockDim>>>(d_a, d_b, d_c );
    cudaMemcpy(C, d_c, sizeof(int)*size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c); 
    cudaDeviceSynchronize();
}


void AddDriver(double A[], double B, double C[], int size) {

    double *d_a, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(double)*size);
    cudaMalloc((void **) &d_c, sizeof(double)*size);
    
    cudaMemcpy(d_a, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(1);
    dim3 GridDim(size);
    AddAS<<<GridDim, BlockDim>>>(d_a, d_c, B );
    cudaMemcpy(C, d_c, sizeof(double)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_c);
    cudaDeviceSynchronize();
}


void MultiplyDriver(double A[], int B[], double C[], int size){
    double B_fl[size];
    for(int  i = 0; i<size;i++ ){
        B_fl[i] = (double)B[i];
    }
    MultiplyDriver(A, B_fl, C, size);
}


void SetWhereLessThanDriver(double A[], double scalar1, double scalar2, int size){
    double *d_a; 
    cudaMalloc((void **) &d_a, sizeof(double)*size);
    
    
    cudaMemcpy(d_a, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    
    dim3 BlockDim(1);
    dim3 GridDim(size);
    SetWhereLessThan<<<GridDim, BlockDim>>>(d_a, scalar1, scalar2 );
    cudaMemcpy(A, d_a, sizeof(double)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaDeviceSynchronize();
}


void ReduceMeanDriver(double A[], double B[], int row, int col, int dimb, int dim){
    double *d_a, *d_b, *d_c;
   
    cudaMalloc((void **) &d_a, sizeof(double)*row*col);
    cudaMalloc((void **) &d_b, sizeof(double)*dimb);
    cudaMalloc((void **) &d_c, sizeof(double)*dimb);
    cudaMemcpy(d_a, A, sizeof(double)*row*col, cudaMemcpyHostToDevice);

    dim3 BlockDim(col);
    dim3 GridDim(row);
    ReduceSum<<<GridDim, BlockDim>>>(d_a, d_b, dim);
     
    DivideAS<<<1, col>>>(d_b, d_c, row);
    cudaMemcpy(B, d_c, sizeof(double)*row, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceSynchronize();
}


void DivideDriver(double A[], double B[], double C[], int size){
    double *d_a, *d_b, *d_c; 
    cudaMalloc((void **) &d_a, sizeof(double)*size);
    cudaMalloc((void **) &d_b, sizeof(double)*size);
    cudaMalloc((void **) &d_c, sizeof(double)*size);
    cudaMemcpy(d_a, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(double)*size, cudaMemcpyHostToDevice);
    dim3 BlockDim(1);
    dim3 GridDim(size);
    Divide<<<GridDim, BlockDim>>>(d_a, d_b, d_c);
    cudaMemcpy(C, d_c, sizeof(double)*size, cudaMemcpyDeviceToHost); 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceSynchronize();
}