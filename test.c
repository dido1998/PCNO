#include <stdio.h>
#include "utils.h"

void TestZeros() {
    int row = 10;
    int col = 10;
    float A[row*col];
    ZerosDriver(A, row, col);
    for(int i = 0;i<row*col;i++){
        if(A[i]!=0.0){
            printf("TEST ZEROS FAILED\n");
            return;
        }
    }
    printf("TEST ZEROS SUCCESS\n");
}

void TestOnes() {
    int row = 10;
    int col = 10;
    float A[row*col];
    ZerosDriver(A, row, col);
    for(int i = 0;i<row*col;i++){
        if(A[i]!=1.0){
            printf("TEST ONES FAILED\n");
            return;
        }
    }
    printf("TEST ONES SUCCESS\n");
}

void TestSquare() {
    int row = 10;
    int col = 10;
    float A[row*col];
    for(int i = 0 ; i<row*col; i++) {
        A[i] = rand() % 10.0;
    }
    float B[row*col];
    SquareDriver(A, B, row, col);
    for(int i = 0;i<row*col;i++){
        if(B[i]!=A[i]*A[i]){
            printf("TEST SQUARE FAILED\n");
            return;
        }
    }
    printf("TEST SQUARE SUCCESS\n");   
}

void TestReduceSum() {
    int row=10;
    int col=10;
    float A[row*col];
    float B[row];
    for(int i = 0;i<row*col;i++) {
        A[i] = i+1;
    }
    float B_ac[row];
    for(int i = 0;i < row;i++) {
        B_ac[i] = 0;
        for(int j = 0; j < col; j++){
            B_ac[i]+= A[i*col + j]; 
        }
    }
    ReduceSumDriver(A, B, row, col, row, 1);
    for (int i = 0; i < row; ++i)
    {
        if(B[i]!=B_ac[i]){
            printf("TEST REDUCE SUM FAILED\n");
            return; 
        }
    }
    printf("TEST REDUCE SUM SUCCESS\n");
}

void TestReduceMean() {
    int row=10;
    int col=10;
    float A[row*col];
    float B[row];
    for(int i = 0;i<row*col;i++) {
        A[i] = i+1;
    }
    float B_ac[row];
    for(int i = 0;i < row;i++) {
        B_ac[i] = 0;
        for(int j = 0; j < col; j++){
            B_ac[i]+= A[i*col + j]/row; 
        }
    }
    ReduceMeanDriver(A, B, row, col, row, 1);
    for (int i = 0; i < row; ++i)
    {
        if(B[i]!=B_ac[i]){
            printf("TEST REDUCE SUM FAILED\n");
            return; 
        }
    }
    printf("TEST REDUCE SUM SUCCESS\n");
}

void TestTranspose() {
    int row=10;
    int col=9;
    float A[row*col];
    float B[row*col];
    for(int i = 0;i<row*col;i++) {
        A[i] = i+1;
    }
    float B_ac[row*col];
    for(int i = 0;i < row;i++) {

        for(int j = 0; j < col; j++){
            B_ac[j*row+i]= A[i*col + j]; 
        }
    }
    TransposeDriver(A, B, row, col);
    for (int i = 0; i < row; ++i)
    {
        if(B[i]!=B_ac[i]){
            printf("TEST TRANSPOSE FAILED\n");
            return; 
        }
    }
    printf("TEST TRANSPOSE SUCCESS\n");
}

void TestDot() {
    int rowa = 4;
    int cola = 5;
    int rowb = 5;
    int colb = 4;
    float A[rowa*cola];
    float B[rowb*colb];
    float C_ac[rowa*colb];
    float C[rowa*colb];
    for(int i = 0;i<rowa*cola;i++) {
        A[i] = i;
    }
    for(int i = 0;i<rowb*colb;i++) {
        A[i] = i;
    }

    for(int i = 0; i<rowa;i++ ){
        for(int j = 0; j<colb ; j++) {
            C_ac[i*colb+j] = 0;
            for(int k=0;k<cola;k++){
                C_ac[i*colb + j]+=A[i*cola + k]*B[k*colb + j];
            }
        }
    }
    DotDriver(A, B, C, rowa, cola, rowb, colb);
    for(int i = 0;i<rowa*colb;i++){
        if(C[i]!=C_ac[i]){
            printf("TEST DOT FAILED\n");
            return;
        }
    }
    printf("TEST DOT SUCCESS\n");
}
void TestMulMatSc(){
    int row=10;
    int col=9;
    float A[row*col];
    float B[row*col];
    for(int i = 0;i<row*col;i++) {
        A[i] = i+1;
    }
    float B_ac[row*col];
    for(int i = 0;i < row*col;i++) {
        B_ac[i] = A[i]*2;
        
    }
    MultiplyDriver(A, 2, B, row, col);
    for (int i = 0; i < row*col; ++i)
    {
        if(B[i]!=B_ac[i]){
            printf("TEST TestMulMatSc FAILED\n");
            return; 
        }
    }
    printf("TEST TestMulMatSc SUCCESS\n");
}

void TestMulArSc(){
    int row=10;
    
    float A[row];
    float B[row];
    for(int i = 0;i<row;i++) {
        A[i] = i+1;
    }
    float B_ac[row];
    for(int i = 0;i < row;i++) {
        B_ac[i] = A[i]*2;
        
    }
    MultiplyDriver(A, 2, B, row, col);
    for (int i = 0; i < row; ++i)
    {
        if(B[i]!=B_ac[i]){
            printf("TEST TestMulArSc FAILED\n");
            return; 
        }
    }
    printf("TEST TestMulArSc SUCCESS\n");
}

void TestDivArSc(){
    int row=10;
    
    float A[row];
    float B[row];
    for(int i = 0;i<row;i++) {
        A[i] = i+1;
    }
    float B_ac[row];
    for(int i = 0;i < row;i++) {
        B_ac[i] = A[i]/2;
        
    }
    DivideDriver(A, 2, B, row);
    for (int i = 0; i < row; ++i)
    {
        if(B[i]!=B_ac[i]){
            printf("TEST TestDiveArSc FAILED\n");
            return; 
        }
    }
    printf("TEST TestDivArSc SUCCESS\n");
}

void TestDivScMat(){
    int row=10;
    int col = 10;
    float A[row*col];
    float B[row*col];
    for(int i = 0;i<row*col;i++) {
        A[i] = i+1;
    }
    float B_ac[row*col];
    for(int i = 0;i < row*col;i++) {
        B_ac[i] = 2/A[i];
        
    }
    DivideDriver(2, A, B, row, col);
    for (int i = 0; i < row*col; ++i)
    {
        if(B[i]!=B_ac[i]){
            printf("TEST TestDiveScMat FAILED\n");
            return; 
        }
    }
    printf("TEST TestDivScMAT SUCCESS\n");
}


void TestAdd(){
    int rowa = 4;
    int cola = 5;
    int rowb = 4;
    int colb = 5;
    float A[rowa*cola];
    float B[rowb*colb];
    float C_ac[rowa*colb];
    float C[rowa*colb];
    for(int i = 0;i<rowa*cola;i++) {
        A[i] = i;
    }
    for(int i = 0;i<rowb*colb;i++) {
        A[i] = i;
    }

    for(int i = 0; i<rowa*cola;i++ ){
        C_ac[i] = A[i]+B[i];
        
    }
    AddDriver(A, B, C, rowa, cola, rowb, colb);
    for(int i = 0;i<rowa*colb;i++){
        if(C[i]!=C_ac[i]){
            printf("TEST ADD FAILED\n");
            return;
        }
    }
    printf("TEST ADD SUCCESS\n");
}

void TestReduceSumVec2Sc(){
    int row=10;
    
    float A[row];
    float B;
    for(int i = 0;i<row;i++) {
        A[i] = i+1;
    }
    float B_ac = 0;
    for(int i = 0;i < row;i++) {
      B_ac[i]+=A[i]; 
    }
    ReduceSumDriver(A, &B, size);
    if(B!=B_ac)
    {
        printf("TEST REDUCE SUM Vec2Sc FAILED\n");
        return;
    }
    printf("TEST REDUCE SUM SUCCESS\n");
}


void TestMaxAS(){
    int row = 4;
    int col =4;
    float A[row*col];
    float B[row*col];
    float B_ac[row*col];
    for(int  i = 0;i<row*col;i++) {
        A[i] = i+1.0;
        if(A[i]>3.0)
        {
            B_ac[i] = A[i];
        }else{
            B_ac[i] = 3.0;
        }
    }
    MaxASDriver(A, 3.0, B, row, col);
    for(int  i = 0;i<row*col;i++) {
        if(B[i]!=B_ac[i]){
            printf("TEST Max FAILED\n");
            return;
        }
    }
    printf("TEST Max SUCCESS\n");
}

void TestBroadcast() {
    int row = 5;
    int col = 5;
    float A[row];
    float B[row*col];
    float B_ac[row*col];
    for(int i = 0; i< row;i++) {
        A[i] = i+1;
        for(int j = 0; j < col ; j++)
        {
            B_ac[i*col+j] = A[i];
        }
    }

    BroadcastArrayToMatrixDriver(A, B, row, col);
    for(int  i = 0;i<row*col;i++) {
        if(B[i]!=B_ac[i]){
            printf("TEST BROADCAST FAILED\n");
            return;
        }
    }
    printf("TEST BROADCAST SUCCESS\n");
}

void TestSetDiagonal(){
    int row = 10;
    int col = 10;
    int A[row*col];
    int B[row*col];
    for(int i = 0; i<row*col;i++){
        A[i] = i+1;
        
    }
    for(int i = 0;i<row;i++){
        for(int j = 0;j<col;j++){
            if(i==j){
                B[i*col + j] = 0;
            }else
            {
                B[i*col + j] = A[i*col + j];
            }
        }
    }
    SetDiagonalDriver(A, 0, row, col);
    for(int  i =0 ; i<row*col;i++){
        if(A[i]!=B[i]){
            printf("set diaginal failed\n");
            return;
        }

    }
    printf("set diaginal success\n");

}

int main() {
	TestZeros();
	TestOnes();
    TestSquare();
    TestReduceSum();
    TestMulMatSc();
    TestMulArSc();
    TestReduceSumVec2Sc();
    TestDivArSc();
    TestMaxAS();
    TestDivScMat();
    TestReduceMean();
}