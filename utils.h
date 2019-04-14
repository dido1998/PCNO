#include <stdio.h>
void ReduceSumDriver(float A[], float B[], int rowa, int cola, int dimb, int dim);
void SquareDriver(float A[], float B[], int rowa, int cola);
void TransposeDriver(float A[], float B[], int rowa, int cola);
void DotDriver(float A[], float B[], float C[], int rowa, int cola, int rowb, int colb);
void AddDriver(float A[], float B[], float C[], int row, int col);
void SubDriver(float A[], float B[], float C[], int row, int col);
void ZerosDriver(float A[], int row, int col);
void OnesDriver(float A[], int row, int col);
void GetReducedRow(float A[], float B[], int row, int col, int rowtoget, int coltoremove);
void NegativeDriver(float A[], int size);
void ExpDriver(float A[], float B[], int size);
void LogDriver(float A[], float B[], int size);
void ReduceSumDriver(float A[], float* B, int size);
void MultiplyDriver(float A[], float B[], float C[], int size) ;
void MultiplyDriver(float A[], float B, float C[], int size);
void DivideDriver(float A[], float B, float C[], int size);
void ReplaceRowExceptCol(float A[], float B[], int row, int col, int rowtoreplace, int colexcept);
void MaxASDriver(float A[], float B, float C[], int row, int col);
void DivideDriver( float B, float A[], float C[], int row, int col);
void SetDiagonalDriver(float A[], float B, int row, int col);
void BroadcastArrayToMatrixDriver(float A[], float B[], int row, int col);
void RangeDriver(float A[], int size, int n);
void RandnDriver(float A[], int row, int col);
void MultiplyDriver(float A[], float B, float C[], int row, int col);
void GetRow(float A[], float B[], int row, int col, int rownum);
void GetCol(float A[], float B[], int row, int col, int colnum);
void StackDriver(float A[], float B[], int row, int col);
void SetRow(float A[], float B[], int row, int col, int rownum);
void IsGreaterThanDriver(float A[], float B, int* C, int size);
void IsEqualDriver(int A[], int B[], int C[], int size);
void IsNotEqualDriver(int A[], int B[], int C[], int size);
void AddDriver(float A[], float B, float C[], int size);
void MultiplyDriver(float A[], int B[], float C[], int size);
void CopyDriver(float A[], float B[], int size);
void SetWhereLessThanDriver(float A[], float scalar1, float scalar2, int size);
void ReduceMeanDriver(float A[], float B[], int row, int col, int dimb, int dim);
void DivideDriver(float A[], float B[], float C[], int size);