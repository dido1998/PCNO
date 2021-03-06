#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

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
int TestRandn();
int TestCopy();
int TestStack();
int TestIsGreaterThan();
int TestIsEqual();
int TestIsNotEqual();
int TestSetWhereLessThan();

int TestZeros() {
    int row = 10;
    int col = 10;
    double A[row*col];
    ZerosDriver(A, row, col);
    for(int i = 0;i<row*col;i++){
        if(A[i]!=0.0){
            printf(ANSI_COLOR_YELLOW "TEST ZEROS FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST ZEROS SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestOnes() {
    int row = 10;
    int col = 10;
    double A[row*col];
    OnesDriver(A, row, col);
    for(int i = 0;i<row*col;i++){
        if(A[i]!=1.0){
            printf(ANSI_COLOR_YELLOW "TEST ONES FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST ONES SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestSquare() {
    int row = 10;
    int col = 10;
    double A[row*col];
    for(int i = 0 ; i<row*col; i++) {
        A[i] = rand() % 10;
    }
    double B[row*col];
    SquareDriver(A, B, row, col);
    for(int i = 0;i<row*col;i++){
        if(B[i]!=A[i]*A[i]){
            printf(ANSI_COLOR_YELLOW "TEST SQUARE FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST SQUARE SUCCESS\n" ANSI_COLOR_RESET);   
    return 1;
}

int TestReduceSum() {
    int row=10;
    int col=10;
    double A[row*col];
    double B[row];
    for(int i = 0;i<row*col;i++) {
        A[i] = i+1;
    }
    double B_ac[row];
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
            printf(ANSI_COLOR_YELLOW "TEST REDUCE SUM FAILED\n" ANSI_COLOR_RESET);
            return 0; 
        }
    }
    printf(ANSI_COLOR_GREEN "TEST REDUCE SUM SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestReduceMean() {
    int row=10;
    int col=10;
    double A[row*col];
    double B[row];
    for(int i = 0;i<row*col;i++) {
        A[i] = i+1;
    }
    double B_ac[row];
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
            printf(ANSI_COLOR_YELLOW "TEST REDUCE SUM FAILED\n" ANSI_COLOR_RESET);
            return 0; 
        }
    }
    printf(ANSI_COLOR_GREEN "TEST REDUCE SUM SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestTranspose() {
    int row=500;
    int col=50;
    double A[row*col];
    double B[row*col];
    for(int i = 0;i<row*col;i++) {
        A[i] = i+1;
    }
    double B_ac[row*col];
    for(int i = 0;i < row;i++) {

        for(int j = 0; j < col; j++){
            B_ac[j*row+i]= A[i*col + j]; 
        }
    }
    TransposeDriver(A, B, row, col);
    for(int i = 0;i<row*col;i++){
        if(B[i]!=B_ac[i]){
            printf(ANSI_COLOR_YELLOW "TEST TRANSPOSE FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST TRANSPOSE SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestDot() {
    int rowa = 4;
    int cola = 5;
    int rowb = 5;
    int colb = 4;
    double A[rowa*cola];
    double B[rowb*colb];
    double C_ac[rowa*colb];
    double C[rowa*colb];
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
            printf(ANSI_COLOR_YELLOW "TEST DOT FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST DOT SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestMulMatSc(){
    int row=10;
    int col=9;
    double A[row*col];
    double B[row*col];
    for(int i = 0;i<row*col;i++) {
        A[i] = i+1;
    }
    double B_ac[row*col];
    for(int i = 0;i < row*col;i++) {
        B_ac[i] = A[i]*2;
        
    }
    MultiplyDriver(A, 2, B, row, col);
    for (int i = 0; i < row*col; ++i)
    {
        if(B[i]!=B_ac[i]){
            printf(ANSI_COLOR_YELLOW "TEST TestMulMatSc FAILED\n" ANSI_COLOR_RESET);
            return 0; 
        }
    }
    printf(ANSI_COLOR_GREEN "TEST TestMulMatSc SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestMulArSc(){
    int row=10;
    
    double A[row];
    double B[row];
    for(int i = 0;i<row;i++) {
        A[i] = i+1;
    }
    double B_ac[row];
    for(int i = 0;i < row;i++) {
        B_ac[i] = A[i]*2;
        
    }
    MultiplyDriver(A, 2, B, row);
    for (int i = 0; i < row; ++i)
    {
        if(B[i]!=B_ac[i]){
            printf(ANSI_COLOR_YELLOW "TEST TestMulArSc FAILED\n" ANSI_COLOR_RESET);
            return 0; 
        }
    }
    printf(ANSI_COLOR_GREEN "TEST TestMulArSc SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestDivArSc(){
    int row=10;
    
    double A[row];
    double B[row];
    for(int i = 0;i<row;i++) {
        A[i] = i+1;
    }
    double B_ac[row];
    for(int i = 0;i < row;i++) {
        B_ac[i] = A[i]/2;
        
    }
    DivideDriver(A, 2, B, row);
    for (int i = 0; i < row; ++i)
    {
        if(B[i]!=B_ac[i]){
            printf(ANSI_COLOR_YELLOW "TEST TestDiveArSc FAILED\n" ANSI_COLOR_RESET);
            return 0; 
        }
    }
    printf(ANSI_COLOR_GREEN "TEST TestDivArSc SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestDivScMat(){
    int row=10;
    int col = 10;
    double A[row*col];
    double B[row*col];
    for(int i = 0;i<row*col;i++) {
        A[i] = i+1;
    }
    double B_ac[row*col];
    for(int i = 0;i < row*col;i++) {
        B_ac[i] = 2/A[i];
        
    }
    DivideDriver(2, A, B, row, col);
    for (int i = 0; i < row*col; ++i)
    {
        if(B[i]!=B_ac[i]){
            printf(ANSI_COLOR_YELLOW "TEST TestDiveScMat FAILED\n" ANSI_COLOR_RESET);
            return 0; 
        }
    }
    printf(ANSI_COLOR_GREEN "TEST TestDivScMAT SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}


int TestAdd(){
    int rowa = 4;
    int cola = 5;
    int rowb = 4;
    int colb = 5;
    double A[rowa*cola];
    double B[rowb*colb];
    double C_ac[rowa*colb];
    double C[rowa*colb];
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
            printf(ANSI_COLOR_YELLOW "TEST ADD FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST ADD SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestReduceSumVec2Sc(){
    int row=10;
    
    double A[row];
    double B;
    for(int i = 0;i<row;i++) {
        A[i] = i+1;
    }
    double B_ac = 0;
    for(int i = 0;i < row;i++) {
      B_ac+=A[i]; 
    }
    ReduceSumDriver(A, &B, row);
    if(B!=B_ac)
    {
        printf(ANSI_COLOR_YELLOW "TEST REDUCE SUM Vec2Sc FAILED\n" ANSI_COLOR_RESET);
        return 0;
    }
    printf(ANSI_COLOR_GREEN "TEST REDUCE SUM SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}


int TestMaxAS(){
    int row = 4;
    int col =4;
    double A[row*col];
    double B[row*col];
    double B_ac[row*col];
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
            printf(ANSI_COLOR_YELLOW "TEST Max FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST Max SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestBroadcast() {
    int row = 5;
    int col = 5;
    double A[col];
    double B[row*col];
    double B_ac[row*col];
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
            printf(ANSI_COLOR_YELLOW "TEST BROADCAST FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST BROADCAST SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestSetDiagonal(){
    int row = 10;
    int col = 10;
    double A[row*col];
    double B[row*col];
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
            printf(ANSI_COLOR_YELLOW "TEST SET DIAGONAL FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }

    }
    printf(ANSI_COLOR_GREEN "TEST SET DIAGONAL SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestSub(){
    int rowa = 6;
    int cola = 9;
    int rowb = 6;
    int colb = 9;
    double A[rowa*cola];
    double B[rowb*colb];
    double C_ac[rowa*colb];
    double C[rowa*colb];
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
            printf(ANSI_COLOR_YELLOW "TEST SUB FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST SUB SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestNegative() {
    int row = 4;
    double A[row * row];
    double A_neg[row * row];
    for(int i = 0; i < row * row; i++) {
        A[i] = rand() % 10000;
        A_neg[i] = -1 * A[i];

    }
    
    NegativeDriver(A, row * row);
    for(int i = 0; i < row * row; i++){

        if(A[i] != A_neg[i]){
            printf(ANSI_COLOR_YELLOW "TEST NEGATIVE FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST NEGATIVE SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestExp() {
    int row = 10;
    double A[row * row];
    double B[row * row];
    double A_exp[row * row];
    for(int i = 0; i < row * row; i++) {
        A[i] = rand() % 10000;
        A_exp[i] = exp(A[i]);
    }
    
    ExpDriver(A, B, row * row);
    for(int i = 0; i < row * row; i++){
        if(B[i] != A_exp[i]){
            printf(ANSI_COLOR_YELLOW "TEST EXP FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST EXP SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestLog() {
    int row = 5;
    double A[row * row];
    double B[row * row];
    double A_log[row * row];
    for(int i = 0; i < row * row; i++) {
        A[i] = rand() % 10000 + 10;
        A_log[i] = log(A[i]);
    }
    
    LogDriver(A, B, row * row);
    for(int i = 0; i < row * row; i++){
        if(B[i] != A_log[i]) {
            printf(ANSI_COLOR_YELLOW "TEST LOG FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST LOG SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestRange() {
    int size = 10;
    double A[size];
    RangeDriver(A, size, 0);
    for(int i = 0; i < size; i++) {

        if (A[i] != (double)i) {
            printf(ANSI_COLOR_YELLOW "TEST RANGE FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST RANGE SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestRandn() {
    int row = 6;
    int col = 9;
    double A[row * col];
    RandnDriver(A, row, col);
    for(int i = 0; i < row * col; i++) {
        if (A[i] < 0 && A[i] > 1) {
            printf(ANSI_COLOR_YELLOW "TEST RANDN FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST RANDN SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestCopy() {
    int size = 69;
    double A[size];
    double B[size];
    for(int i = 0; i < size; i++) {
        A[i] = i + 420;
    }
    CopyDriver(A, B, size);
    for(int i = 0; i < size; i++) {
        if (A[i] != B[i]) {
            printf(ANSI_COLOR_YELLOW "TEST COPY FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST COPY SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestStack() {
    int row = 6;
    int col = 9;
    double A[row * col];
    double B[row * col];
    for(int i = 0; i < row * col; i++) {
        A[i] = i + 420;
    }
    StackDriver(A, B, row, col);
    for(int i = 0, k = 0; i < row * col; i++, k++) {
        if (k == col) {
            k = 0;
        }
        if (A[k] != B[i]) {
            printf(ANSI_COLOR_YELLOW "TEST STACK FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST STACK SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestIsGreaterThan() {
    int size = 10;
    double A[size];
    int R[size];
    int R_kr[size];
    double chonk = 500.0;
    for(int i = 0; i < size; i++) {
        A[i] = rand() % 999;
        if (A[i] > chonk) {
            R_kr[i] = 1;
        } else {
            R_kr[i] = 0;
        }
    }
    IsGreaterThanDriver(A, chonk, R, size);
    for(int i = 0; i < size; i++) {
        if (R[i] != R_kr[i]) {
            printf(ANSI_COLOR_YELLOW "TEST ISGREATERTHAN FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST ISGREATERTHAN SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestIsEqual() {
    int size = 10;
    int A[size];
    int B[size];
    int R[size];
    int R_kr[size];
    for(int i = 0; i < size; i++) {
        A[i] = rand() % 999;
        if (A[i] == B[i]) {
            R_kr[i] = 1;
        } else {
            R_kr[i] = 0;
        }
    }
    IsEqualDriver(A, B, R, size);
    for(int i = 0; i < size; i++) {
        if (R[i] != R_kr[i]) {
            printf(ANSI_COLOR_YELLOW "TEST ISEQUAL FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST ISEQUAL SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestIsNotEqual() {
    int size = 10;
    int A[size];
    int B[size];
    int R[size];
    int R_kr[size];
    for(int i = 0; i < size; i++) {
        A[i] = rand() % 999;
        if (A[i] != B[i]) {
            R_kr[i] = 1;
        } else {
            R_kr[i] = 0;
        }
    }
    IsNotEqualDriver(A, B, R, size);
    for(int i = 0; i < size; i++) {
        if (R[i] != R_kr[i]) {
            printf(ANSI_COLOR_YELLOW "TEST ISNOTEQUAL FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST ISNOTEQUAL SUCCESS\n" ANSI_COLOR_RESET);
    return 1;
}

int TestSetWhereLessThan() {
    int size = 10;
    double A[size];
    double A_kr[size]; 
    double chonkTest = 500.0;
    double chonkSet = 1337.0;
    for(int i = 0; i < size; i++) {
        A[i] = rand() % 999;
        if (A[i] < chonkTest) {
            A_kr[i] = chonkSet;
        } else {
            A_kr[i] = A[i];
        }
    }
    SetWhereLessThanDriver(A, chonkTest, chonkSet, size);
    for(int i = 0; i < size; i++) {
        if (A[i] != A_kr[i]) {
            printf(ANSI_COLOR_YELLOW "TEST SETWHERELESSTHAN FAILED\n" ANSI_COLOR_RESET);
            return 0;
        }
    }
    printf(ANSI_COLOR_GREEN "TEST SETWHERELESSTHAN SUCCESS\n" ANSI_COLOR_RESET);
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
    failedCount += (TestRandn() == 0);
    failedCount += (TestCopy() == 0);
    failedCount += (TestStack() == 0);
    failedCount += (TestIsGreaterThan() == 0);
    failedCount += (TestIsEqual() == 0);
    failedCount += (TestIsNotEqual() == 0);
    failedCount += (TestSetWhereLessThan() == 0);


    printf(ANSI_COLOR_YELLOW "\n\n[RESULT] " ANSI_COLOR_RESET);
    if (!failedCount) {
        printf(ANSI_COLOR_GREEN "ALL TESTS PASSED SUCCESSFULLY\n" ANSI_COLOR_RESET);
    } else {
        printf(ANSI_COLOR_RED " %d TESTS FAILED\n" ANSI_COLOR_RESET, failedCount);
    }

}