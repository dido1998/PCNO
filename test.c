/* 
    TO-DO:
        Add menu to print verbose or not
        Make print Informatives
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

int TestZeros();
int TestOnes();
int TestSquare();
int TestReduceSum();
int TestReduceMean();
int TestTranspose();
int TestDot();
int TestMulMatSc();
int TestMulArSc();
int TestDivArSc();
int TestDivScMat();
int TestAdd();
int TestReduceSumVec2Sc();
int TestMaxAS();
int TestBroadcast();
int TestSetDiagonal();
int TestSub();
int TestNegative();
int TestExp();
int TestLog();
int TestRange();

int TestZeros() {
    int row = 10;
    int col = 10;
    float A[row*col];
    ZerosDriver(A, row, col);
    for(int i = 0;i<row*col;i++){
        if(A[i]!=0.0){
            printf("TEST ZEROS FAILED\n");
            return 0;
        }
    }
    printf("TEST ZEROS SUCCESS\n");
    return 1;
}

int TestOnes() {
    int row = 10;
    int col = 10;
    float A[row*col];
    OnesDriver(A, row, col);
    for(int i = 0;i<row*col;i++){
        if(A[i]!=1.0){
            printf("TEST ONES FAILED\n");
            return 0;
        }
    }
    printf("TEST ONES SUCCESS\n");
    return 1;
}

int TestSquare() {
    int row = 10;
    int col = 10;
    float A[row*col];
    for(int i = 0 ; i<row*col; i++) {
        A[i] = rand() % 10;
    }
    float B[row*col];
    SquareDriver(A, B, row, col);
    for(int i = 0;i<row*col;i++){
        if(B[i]!=A[i]*A[i]){
            printf("TEST SQUARE FAILED\n");
            return 0;
        }
    }
    printf("TEST SQUARE SUCCESS\n");   
    return 1;
}

int TestReduceSum() {
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
            return 0; 
        }
    }
    printf("TEST REDUCE SUM SUCCESS\n");
    return 1;
}

int TestReduceMean() {
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
            return 0; 
        }
    }
    printf("TEST REDUCE SUM SUCCESS\n");
    return 1;
}

int TestTranspose() {
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
            return 0; 
        }
    }
    printf("TEST TRANSPOSE SUCCESS\n");
    return 1;
}

int TestDot() {
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
            return 0;
        }
    }
    printf("TEST DOT SUCCESS\n");
    return 1;
}

int TestMulMatSc(){
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
            return 0; 
        }
    }
    printf("TEST TestMulMatSc SUCCESS\n");
    return 1;
}

int TestMulArSc(){
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
    MultiplyDriver(A, 2, B, row);
    for (int i = 0; i < row; ++i)
    {
        if(B[i]!=B_ac[i]){
            printf("TEST TestMulArSc FAILED\n");
            return 0; 
        }
    }
    printf("TEST TestMulArSc SUCCESS\n");
    return 1;
}

int TestDivArSc(){
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
            return 0; 
        }
    }
    printf("TEST TestDivArSc SUCCESS\n");
    return 1;
}

int TestDivScMat(){
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
            return 0; 
        }
    }
    printf("TEST TestDivScMAT SUCCESS\n");
    return 1;
}


int TestAdd(){
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
    AddDriver(A, B, C, rowa, cola);
    for(int i = 0;i<rowa*colb;i++){
        if(C[i]!=C_ac[i]){
            printf("TEST ADD FAILED\n");
            return 0;
        }
    }
    printf("TEST ADD SUCCESS\n");
    return 1;
}

int TestReduceSumVec2Sc(){
    int row=10;
    
    float A[row];
    float B;
    for(int i = 0;i<row;i++) {
        A[i] = i+1;
    }
    float B_ac = 0;
    for(int i = 0;i < row;i++) {
      B_ac+=A[i]; 
    }
    ReduceSumDriver(A, &B, row);
    if(B!=B_ac)
    {
        printf("TEST REDUCE SUM Vec2Sc FAILED\n");
        return 0;
    }
    printf("TEST REDUCE SUM SUCCESS\n");
    return 1;
}


int TestMaxAS(){
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
            return 0;
        }
    }
    printf("TEST Max SUCCESS\n");
    return 1;
}

int TestBroadcast() {
    int row = 5;
    int col = 5;
    float A[col];
    float B[row*col];
    float B_ac[row*col];
    for(int i = 0; i< col;i++) {
        
        for(int j = 0; j < row ; j++)
        {
            A[i] = i+1;
            B_ac[j*col+i] = A[i];
        }
    }

    BroadcastArrayToMatrixDriver(A, B, row, col);
    
    for(int  i = 0;i<row*col;i++) {
        if(B[i]!=B_ac[i]){
            printf("TEST BROADCAST FAILED\n");
            return 0;
        }
    }
    printf("TEST BROADCAST SUCCESS\n");
    return 1;
}

int TestSetDiagonal(){
    int row = 10;
    int col = 10;
    float A[row*col];
    float B[row*col];
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
            printf("SET DIAGONAL FAILED\n");
            return 0;
        }

    }
    printf("SET DIAGONAL SUCCESS\n");
    return 1;
}

int TestSub(){
    int rowa = 6;
    int cola = 9;
    int rowb = 6;
    int colb = 9;
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
        C_ac[i] = A[i] - B[i];
        
    }
    SubDriver(A, B, C, rowa, cola);
    for(int i = 0;i<rowa*colb;i++){
        if(C[i]!=C_ac[i]){
            printf("TEST SUB FAILED\n");
            return 0;
        }
    }
    printf("TEST SUB SUCCESS\n");
    return 1;
}

int TestNegative() {
    int row = 4;
    float A[row * row];
    float A_neg[row * row];
    for(int i = 0; i < row * row; i++) {
        A[i] = rand() % 10000;
        A_neg[i] = -1 * A[i];

    }
    
    NegativeDriver(A, row * row);
    for(int i = 0; i < row * row; i++){

        if(A[i] != A_neg[i]){
            printf("TEST NEGATIVE FAILED\n");
            return 0;
        }
    }
    printf("TEST NEGATIVE SUCCESS\n");
    return 1;
}

int TestExp() {
    int row = 10;
    float A[row * row];
    float B[row * row];
    float A_exp[row * row];
    for(int i = 0; i < row * row; i++) {
        A[i] = rand() % 10000;
        A_exp[i] = exp(A[i]);
    }
    
    ExpDriver(A, B, row * row);
    for(int i = 0; i < row * row; i++){
        if(B[i] != A_exp[i]){
            printf("TEST EXP FAILED\n");
            return 0;
        }
    }
    printf("TEST EXP SUCCESS\n");
    return 1;
}

int TestLog() {
    int row = 5;
    float A[row * row];
    float B[row * row];
    float A_log[row * row];
    for(int i = 0; i < row * row; i++) {
        A[i] = rand() % 10000 + 10;
        A_log[i] = log(A[i]);
    }
    
    LogDriver(A, B, row * row);
    for(int i = 0; i < row * row; i++){
        if(B[i] != A_log[i]) {
            printf("TEST LOG FAILED\n");
            return 0;
        }
    }
    printf("TEST LOG SUCCESS\n");
    return 1;
}

int TestRange() {
    int size = 10;
    float A[size];
    RangeDriver(A, size, 0);
    for(int i = 0; i < size; i++) {
       
         if (A[i] != i) {
             printf("TEST RANGE FAILED\n");
             return 0;
             }
    }
    printf("TEST RANGE SUCCESS\n");
    return 1;
}

int main() {
    int failedCount = 0;

    failedCount += (TestZeros() == 0);
    failedCount += (TestOnes() == 0);
    failedCount += (TestSquare() == 0);
    failedCount += (TestReduceSum() == 0);
    failedCount += (TestReduceMean() == 0);
    failedCount += (TestTranspose() == 0);
    failedCount += (TestDot() == 0);
    failedCount += (TestMulMatSc() == 0);
    failedCount += (TestMulArSc() == 0);
    failedCount += (TestDivArSc() == 0);
    failedCount += (TestDivScMat() == 0);
    failedCount += (TestAdd() == 0);
    failedCount += (TestReduceSumVec2Sc() == 0);
    failedCount += (TestMaxAS() == 0);
    failedCount += (TestBroadcast() == 0);
    failedCount += (TestSetDiagonal() == 0);
    failedCount += (TestSub() == 0);
    failedCount += (TestNegative() == 0);
    failedCount += (TestExp() == 0);
    failedCount += (TestLog() == 0);
    failedCount += (TestRange() == 0);

    if (!failedCount) {
        printf("\n\n[!NOTICE!] ALL TESTS PASSED SUCCESSFULLY\n");
    } else {
        printf("\n\n[!NOTICE!] %d TESTS FAILED\n", failedCount);
    }

}