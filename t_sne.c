#include <stdio.h>
#include <iostream>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include "utils.h"
const int numdim = 50;
double data[500][50];
double y_start[500][2];
void print(double A[], int size){
	int cnt = 0;
	for(int i = 0; i < size; i++){
		printf("%.12f\n", A[i]);
		if(A[i]>1)
			cnt++;
	}
	printf("\n%d\n", cnt);
}

void Hbeta(double D[], double* H, double P[], int size, double beta, int i){
	double P_ar[size];
	MultiplyDriver(D, -beta, P_ar, size);
	
	double P_ar_exp[size];

	ExpDriver(P_ar, P_ar_exp, size);

	double sumP = 0;
	ReduceSumDriver(P_ar_exp, &sumP, size);
	
	double DP[size];
	MultiplyDriver(D, P_ar_exp, DP, size);

	double DP_sum = 0;
	
	ReduceSumDriver(DP, &DP_sum, size);
	
	
	*H = log(sumP) +(beta/sumP)*DP_sum;

	DivideDriver(P_ar_exp, sumP, P, size);
	
}
void x2p(double X[], double P[], int num_examples, int num_dimension, int initial_dims, double tol, double perplexity) {
	
	double* X_sq = new double[num_examples*initial_dims];
	SquareDriver(X, X_sq, num_examples, initial_dims);

	double* sum_X = new double[num_examples];
	ReduceSumDriver(X_sq, sum_X, num_examples, initial_dims, num_examples, 1);
	
	double* X_tr = new double[num_examples*initial_dims];

	TransposeDriver(X, X_tr, num_examples, initial_dims);
	
	double* X_X_tr_Dot = new double[num_examples*num_examples];

	DotDriver(X, X_tr, X_X_tr_Dot, num_examples, initial_dims, initial_dims, num_examples);
	
	double* scmul = new double[num_examples*num_examples];
	MultiplyDriver(X_X_tr_Dot, -2, scmul, num_examples, num_examples);

	double* add_d = new double[num_examples*num_examples];
	double* sum_X_br = new double[num_examples*num_examples];

	BroadcastArrayToMatrixDriver(sum_X, sum_X_br, num_examples, num_examples);
	AddDriver(scmul, sum_X_br, add_d, num_examples, num_examples);
	
	double* add_d_tr = new double[num_examples*num_examples];


	TransposeDriver(add_d, add_d_tr, num_examples, num_examples);

	double* D = new double[num_examples*num_examples];

	AddDriver(add_d_tr, sum_X_br, D, num_examples, num_examples);
	
	double beta[num_examples*1];
	OnesDriver(beta, num_examples, 1);
	double logU = log(perplexity);

	for(int i = 0; i < num_examples; i++) {
		double betamin = -INFINITY;
		double betamax = INFINITY;
		double Di[num_examples-1];
		GetReducedRow(D, Di, num_examples, num_examples, i, i);

		double H, thisP[num_examples-1];

		Hbeta(Di, &H, thisP, num_examples-1, 1.0, i);

		double Hdiff = H - logU;
		int tries = 0;
		
		while(abs(Hdiff)>tol && tries<50){

			if (Hdiff>0){
				betamin = beta[i];
				if (betamax==INFINITY||betamax==-INFINITY){
					beta[i] = beta[i]*2.0;
				}else{
					beta[i] = (beta[i] + betamax) / 2.0;
				}
			}else{
				betamax = beta[i];
				if (betamin==INFINITY||betamin==-INFINITY){
					beta[i] = beta[i]/2.0;
				}else{
					beta[i] = (beta[i] + betamin) / 2.0;
				}
			}
			Hbeta(Di, &H, thisP, num_examples-1, beta[i],i);
			
			Hdiff = H - logU;

			

			tries+=1;
		}

		ReplaceRowExceptCol(P, thisP, num_examples, num_examples, i, i);
	}

}

void tsne(double X[], double Y[], int num_examples, int num_dimension, int initial_dims, double perplexity) {
	
	int max_iter = 1000;
	double initial_momentum = 0.5;
	double final_momentum =0.8;
	int eta = 500;
	double min_gain = 0.01;
	
	//RandnDriver(Y, num_examples, num_dimension);
	double Dy[num_examples*num_dimension];
	ZerosDriver(Dy, num_examples, num_dimension);
	double iY[num_examples*num_dimension];
	ZerosDriver(iY, num_examples, num_dimension);
	double gains[num_examples*num_dimension];


	OnesDriver(gains, num_examples, num_dimension);

	double* P = new double[num_examples*num_examples];
	

	ZerosDriver(P, num_examples, num_examples);

	x2p(X, P, num_examples,num_dimension, initial_dims,  0.00005, perplexity);
	
	double* P_tr = new double[num_examples*num_examples];
	TransposeDriver(P, P_tr, num_examples, num_examples);
	double* add_P_P_tr = new double[num_examples*num_examples];
	AddDriver(P, P_tr, add_P_P_tr, num_examples, num_examples);
	double P_sum = 0;
	
	ReduceSumDriver(add_P_P_tr, &P_sum, num_examples*num_examples);
	
	double* P_div = new double[num_examples*num_examples];
	//DivideDriver(add_P_P_tr, P_sum, P_div, num_examples*num_examples);
	for(int i = 0; i <  num_examples*num_examples;i++){
		P_div[i] = add_P_P_tr[i]/P_sum;
	}
	

	double* P_mul = new double[num_examples*num_examples];
	MultiplyDriver(P_div, 4, P_mul, num_examples, num_examples);
	
	double* P_max = new double[num_examples*num_examples];

	MaxASDriver(P_mul, 0.000000000001, P_max, num_examples, num_examples);
	
	CopyDriver(P_max, P, num_examples*num_examples);
	
	for(int iter = 0; iter < max_iter; iter++) {
		
		double Y_sq[num_examples*num_dimension];

		SquareDriver(Y, Y_sq, num_examples, num_dimension);
		double sum_Y[num_examples];
		ReduceSumDriver(Y_sq, sum_Y, num_examples, num_dimension, num_examples, 1);

		double Y_tr[num_dimension*num_examples];
		TransposeDriver(Y, Y_tr, num_examples, num_dimension);	
		double Y_Y_tr_dot[num_examples*num_examples];
		
		DotDriver(Y, Y_tr, Y_Y_tr_dot, num_examples, num_dimension, num_dimension, num_examples);
		
		double num[num_examples*num_examples];
		MultiplyDriver(Y_Y_tr_dot, -2, num, num_examples, num_examples);
		
		double sum_Y_br[num_examples*num_examples];
		BroadcastArrayToMatrixDriver(sum_Y, sum_Y_br, num_examples, num_examples);
		double add_y[num_examples*num_examples];
		AddDriver(num, sum_Y_br, add_y, num_examples, num_examples);
		

		double* add_y_tr = new double[num_examples*num_examples];
		TransposeDriver(add_y, add_y_tr, num_examples, num_examples);
		double* add_add_y_tr = new double[num_examples*num_examples];
		AddDriver(add_y_tr, sum_Y_br, add_add_y_tr, num_examples, num_examples);

		double* ones = new double[num_examples*num_examples];
		double* add_ones= new double[num_examples*num_examples];
		OnesDriver(ones, num_examples, num_examples);
		AddDriver(ones, add_add_y_tr, add_ones, num_examples, num_examples);
		double* one_div = new double[num_examples*num_examples];
		DivideDriver(1, add_ones, one_div, num_examples, num_examples);
		SetDiagonalDriver(one_div, 0, num_examples, num_examples);
		CopyDriver(one_div, num, num_examples*num_examples);
		
		double num_sum = 0;
		ReduceSumDriver(num, &num_sum, num_examples*num_examples);
		
		double* num_div = new double[num_examples*num_examples];
		DivideDriver(num, num_sum, num_div, num_examples*num_examples);
		double* Q = new double[num_examples*num_examples];
		MaxASDriver(num_div, 0.000000000001, Q, num_examples, num_examples);
		double* PQ = new double[num_examples*num_examples];

		SubDriver(P, Q, PQ, num_examples, num_examples);
		
		for(int i = 0; i < num_examples; i ++) {
			double PQ_col[num_examples];
			GetCol(PQ, PQ_col, num_examples, num_examples, i);

			double num_col[num_examples];
			GetCol(num, num_col, num_examples, num_examples, i);
			
			double mul_pq_num[num_examples];
			MultiplyDriver(PQ_col, num_col, mul_pq_num, num_examples);
			
			double mul_pq_num_stack[num_examples*num_dimension];
			StackDriver(mul_pq_num, mul_pq_num_stack, num_dimension, num_examples);
			
			double mul_pq_num_stack_tr[num_examples*num_dimension];
			TransposeDriver(mul_pq_num_stack, mul_pq_num_stack_tr, num_dimension, num_examples);
			double Y_row[num_dimension];
			GetRow(Y, Y_row, num_examples, num_dimension, i);
			
			double Y_row_st[num_examples*num_dimension];
			StackDriver(Y_row, Y_row_st, num_examples, num_dimension);
			
			double sub_y_y_st[num_examples*num_dimension];
			SubDriver(Y_row_st, Y, sub_y_y_st, num_examples, num_dimension);
			
			double mul_grad[num_examples*num_dimension];
			
			MultiplyDriver(sub_y_y_st, mul_pq_num_stack_tr, mul_grad, num_examples*num_dimension);
			double mul_grad_tr[num_examples*num_dimension];
			TransposeDriver(mul_grad, mul_grad_tr, num_examples, num_dimension);
			double row_grad[num_dimension];
			ReduceSumDriver(mul_grad_tr, row_grad, num_dimension, num_examples, num_dimension, 1);
			
			SetRow(Dy, row_grad, num_examples, num_dimension, i);
		}
		
		double momentum;
		if(iter<20){
			momentum = initial_momentum;
		}else{
			momentum = final_momentum;
		}

		//CALCULATE GAINS
		double* gains_add = new double[num_examples*num_dimension];
		double* gains_mul = new double[num_examples*num_dimension];
		AddDriver(gains, 0.2, gains_add, num_examples*num_dimension);
		MultiplyDriver(gains, 0.8, gains_mul, num_examples*num_dimension);
		int* iY_gr_zero = new int[num_examples*num_dimension];
		IsGreaterThanDriver(iY, 0, iY_gr_zero, num_examples*num_dimension);
		int* dY_gr_zero = new int[num_examples*num_dimension];

		IsGreaterThanDriver(Dy, 0, dY_gr_zero, num_examples*num_dimension);
		int* iY_dY_eq = new int[num_examples*num_dimension];
		IsEqualDriver(dY_gr_zero, iY_gr_zero, iY_dY_eq, num_examples*num_dimension);
		int* iY_dY_neq = new int[num_examples*num_dimension];
		IsNotEqualDriver(dY_gr_zero, iY_gr_zero, iY_dY_neq, num_dimension*num_examples);
		double* gain1 = new double[num_examples*num_dimension];
		MultiplyDriver(gains_add, iY_dY_neq, gain1, num_dimension*num_examples);
		double* gain2 = new double[num_examples*num_dimension];
		MultiplyDriver(gains_mul, iY_dY_eq, gain2, num_examples*num_dimension);
		double* gain_final_add = new double[num_examples*num_dimension];
		AddDriver(gain1, gain2, gain_final_add, num_examples, num_dimension);
		CopyDriver(gain_final_add, gains, num_examples*num_dimension);
		SetWhereLessThanDriver(gains, min_gain, min_gain, num_examples*num_dimension);
		


		double* gains_dy = new double[num_dimension*num_examples];
		MultiplyDriver(gains, Dy, gains_dy, num_examples*num_dimension);
		


		double* gains_dy_eta = new double[num_examples*num_dimension];
		MultiplyDriver(gains_dy, (double)eta, gains_dy_eta, num_examples*num_dimension );
		double* momentum_iy = new double[num_examples*num_dimension];
		MultiplyDriver(iY, momentum, momentum_iy, num_examples*num_dimension);
		double* iY_add = new double[num_examples*num_dimension];
		SubDriver(momentum_iy, gains_dy_eta, iY_add, num_dimension, num_examples);
		CopyDriver(iY_add, iY, num_examples*num_dimension);
		


		double* Y_add = new double[num_examples*num_dimension];
		
		AddDriver(Y, iY, Y_add, num_examples, num_dimension);
		

		CopyDriver(Y_add, Y, num_examples*num_dimension);
		double* y_mean = new double[2];
		for(int i = 0;i<num_dimension;i++){
			y_mean[i] = 0;
			for(int j = 0; j < num_examples;j++){
				y_mean[i]+=Y[j*2+i];
			}
			y_mean[i]/=num_examples;
		}

		/*double* Y_tr_1 = new double[num_dimension*num_examples];
		TransposeDriver(Y, Y_tr_1, num_examples,num_dimension);
		

		double* Y_tr_mean = new double[num_dimension];

		ReduceMeanDriver(Y_tr_1, Y_tr_mean, num_dimension, num_examples, num_dimension,1);
		*/

		double* Y_stack = new double[num_examples*num_dimension];
		StackDriver(y_mean, Y_stack, num_examples, num_dimension);
				

		double* Y_sub = new double[num_examples*num_dimension];
		SubDriver(Y, Y_stack, Y_sub, num_examples, num_dimension);
		CopyDriver(Y_sub, Y, num_examples*num_dimension);
		

		/*double* update = new double[num_examples*num_dimension];
		
		MultiplyDriver(Dy, 0.005, update, num_examples*num_dimension);
		double* y_1 = new double[num_examples*num_dimension];
		SubDriver(Y, update, y_1, num_examples, num_dimension);
		CopyDriver(y_1, Y, num_examples*num_dimension);*/

		
		if((iter+1)%10==0) {
			
			double* div_p_q = new double[num_examples*num_examples];
			DivideDriver(P, Q, div_p_q, num_examples*num_examples);

			double* log = new double[num_examples*num_examples];	
			
			LogDriver(div_p_q, log, num_examples*num_examples);

			double* mul_p = new double[num_examples*num_examples];
			MultiplyDriver(P, log, mul_p, num_examples*num_examples);
			double C = 0;
			ReduceSumDriver(mul_p, &C, num_examples*num_examples);
			printf("iteration %d: error is %f\n",iter+1, C );
		}
		if(iter==100){
			double* P_d = new double[num_examples*num_examples];
			DivideDriver(P, 4, P_d, num_examples*num_examples);
			CopyDriver(P_d, P, num_examples*num_examples);
		}
	}
}


void read_csv(int row, int col, char *filename){
	FILE *file;
	file = fopen(filename, "r");

	int i = 0;
    char line[4098];
	while (fgets(line, 4098, file) && (i < row))
    {
    	// double row[ssParams->nreal + 1];
        char* tmp = strdup(line);

	    int j = 0;
	    const char* tok;
	    for (tok = strtok(line, ","); tok && *tok; j++, tok = strtok(NULL, ","))
	    {
	        data[i][j] = atof(tok);
	        
	    }
	    //printf("\n");
	    
        //free(tmp);
        i++;
    }
}

void read_csv_y(int row, int col, char *filename){
	FILE *file;
	file = fopen(filename, "r");

	int i = 0;
    char line[4098];
	while (fgets(line, 4098, file) && (i < row))
    {
    	// double row[ssParams->nreal + 1];
        char* tmp = strdup(line);

	    int j = 0;
	    const char* tok;
	    for (tok = strtok(line, ","); tok && *tok; j++, tok = strtok(NULL, ","))
	    {
	        y_start[i][j] = atof(tok);
	        
	    }
	    //printf("\n");
	    
        //free(tmp);
        i++;
    }
}

int main(int argc, char const *argv[])
{
	

	int row = 500;
	int col = 50;
	char fname[256];	strcpy(fname, "data.csv");

	
	

	read_csv(row, col, fname);
	double X[500 * 50];
	for(int  i =0 ; i<500;i++)
	{
		for(int  j = 0; j < 50 ; j++) {
			//std::cout<< fixed << setprecision(15) <<data[i][j]<<'\n';
			//printf("%f\n", data[i][j]);
			X[i * 50 + j] = data[i][j];
		}
	}
	read_csv_y(500, 2, "start_data_1.csv");
	//free(data);
	double Y[500*2];
	for(int  i =0 ; i<500;i++)
	{
		for(int  j = 0; j < 2 ; j++) {
			//std::cout<< fixed << setprecision(15) <<data[i][j]<<'\n';
			//printf("%f\n", data[i][j]);
			Y[i * 2 + j] = y_start[i][j];
		}
	}
	//print(X,500*50);
	tsne(X, Y,500, 2, 50, 30.0);
	FILE* fptr = fopen("final_c.csv","w");
	for(int i = 0; i < 500; i++)
     fprintf(fptr, "%f,%f\n", Y[2*i], Y[2*i+1]);
	return 0;
}
