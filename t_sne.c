#include <stdio.h>
#include<math.h>
#include "utils.h"
#include "mnist.h"

void converttocuda(int X[][50], int X_cuda[] int num_examples) {
	int X_cuda_index = 0;
	for(int i = 0; i<num_examples; i++) {
		for(int j = 0; j < 50; j++) {
			X_cuda[X_cuda_index++] = X[i,j];
		}
	}
}
void Hbeta(float D[], float* H, float P[], int size, float beta){
	float P_ar[size];
	MultiplyDriver(D, -beta, P_ar, size);
	float P_ar_exp[size];
	ExpDriver(P_ar, P_ar_exp, size);
	float sumP;
	ReduceSumDiver(P_ar_exp, &sumP, size);
	float DP[size];
	MultiplyDriver(D, P, DP, size);
	float DP_sum;
	ReduceSumDiver(DP, &DP_sum, size);
	*H = log(sumP) +(beta/sumP)*DP_sum;
	DivideDriver(P_ar, sumP, P, size);
}
void x2p(float X[], float P[], int num_examples, int num_dimension, int initial_dims, float tol, float perplexity) {
	float X_sq[num_examples*initial_dims];
	SquareDriver(X, X_sq, num_examples, initial_dims);
	float sum_X[num_examples];
	ReduceSumDiver(X_sq, sum_X, num_examples, initial_dims, num_examples, 1);
	float X_tr[num_examples*initial_dims];
	TransposeDriver(X, X_tr, num_examples, num_dimension);
	float X_X_tr_Dot[num_examples*num_examples];
	DotDriver(X, X_tr, X_X_tr_Dot, num_examples, initial_dims, initial_dims, num_examples);
	float scmul[num_examples*num_examples];
	MultiplyDriver(X_X_tr_Dot, -2, scmul, num_examples, num_examples);
	float add_d[num_examples*num_examples];
	AddDriver(scmul, sum_X, add_d, num_examples, num_examples);
	float add_d_tr[num_examples*num_examples];
	TransposeDriver(add_d, add_d_tr, num_examples, num_examples);
	float D[num_examples*num_examples];
	AddDriver(add_d_tr, sum_X, D, num_examples, num_examples);
	float beta[num_examples*1];
	OnesDriver(beta, num_examples, 1);
	float logU = log(perplexity);
	for(int i = 0; i < num_examples; i++) {
		float betamin = -INFINITY;
		float betamax = INFINITY;
		float Di[num_examples-1];
		GetReducedRow(D, Di, num_examples, num_examples, i, i);
		float H, thisP[num_examples-1];
		Hbeta(Di, &H, thisP, num_examples-1, 1.0);
		float Hdiff = H - logU;
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
			Hbeta(Di, &H, thisP, beta[i]);
			Hdiff = H - logU;
			tries+=1;
		}
		ReplaceRowExceptCol(P, thisP, num_examples, num_examples, i, i);
	}

}

void tsne(float X[][50], float Y[][], int num_examples, int num_dimension, int initial_dims, float perplexity) {
	float X_cuda[num_examples*initial_dims];
	converttocuda(X, X_cuda);
	
	int max_iter = 1000;
	float initial_momentum = 0.5;
	float final_momentum = 0.8;
	int eta = 500;
	float min_gain = 0.01;
	float Y_cuda[num_examples*num_dimension];
	converttocuda(Y, Y_cuda);
	RandnDiver(A, num_examples, num_dimension);
	float Dy[num_examples*num_dimension];
	ZerosDriver(Dy, num_examples, num_dimension);
	float iY[num_examples*num_dimension];
	ZerosDriver(iY, num_examples, num_dimension);
	float gains[num_examples*num_dimension];
	OnesDriver(gains, num_examples, num_dimension);
	float P[num_examples*num_examples];
	ZerosDriver(P, num_examples, num_examples);
	x2p(X, P, 0.00005, perplexity);
	float P_tr[num_examples*num_examples];
	TransposeDriver(P, P_tr, num_examples, num_examples);
	float add_P_P_tr[num_examples*num_examples];
	AddDriver(P, P_tr, add_P_P_tr, num_examples, num_examples);
	float P_sum;
	ReduceSumDiver(add_P_P_tr, &P_sum, num_examples*num_examples);
	float P_div[num_examples*num_examples];
	DivideDriver(add_P_P_tr, P_sum, P_div, num_examples, num_examples);
	float P_mul[num_examples*num_examples];
	MultiplyDriver(P_div, 4, P_mul, num_examples, num_examples);
	float P_max[num_examples*num_examples];
	MaxASDriver(P_div, 0.000000000001, P_max, num_examples, num_examples);
	CopyDriver(P_max, P, num_examples*num_examples);
	for(int iter = 0; iter < max_iter; iter++) {
		float Y_sq[num_examples*num_dimension];
		SquareDriver(Y, Y_sq, num_examples, num_dimension);
		float sum_Y[num_examples];
		ReduceSumDiver(Y_sq, sum_Y, num_examples, num_dimension, num_examples, 1);
		float Y_tr[num_dimension*num_examples];
		TransposeDriver(Y, Y_tr, num_examples, num_dimension);	
		float Y_Y_tr_dot[num_examples*num_examples];
		DotDriver(Y, Y_tr, Y_Y_tr_dot, num_examples, num_dimension, num_dimension, num_examples);
		float num[num_examples*num_examples];
		MultiplyDriver(Y_Y_tr_dot, -2, num, num_examples, num_examples);
		float sum_Y_br[num_examples*num_examples];
		BroadcastArrayToMatrixDriver(sum_Y, sum_Y_br, num_examples, num_examples);
		float add_y[num_examples*num_examples];
		AddDriver(num, sum_Y_br, add_y, num_examples, num_examples);
		float add_y_tr[num_examples*num_examples];
		TransposeDriver(add_y, add_y_tr, num_examples, num_examples);
		float add_add_y_tr[num_examples*num_examples];
		AddDriver(add_y_tr, sum_Y_br, add_add_y_tr, num_examples, num_examples);
		float ones[num_examples*num_examples];
		float add_ones[num_examples*num_examples];
		OnesDriver(ones, num_examples, num_examples);
		AddDriver(ones, add_add_y_tr, add_ones, num_examples, num_examples);
		float one_div[num_examples*num_examples];
		DivideDriver(1, add_ones, one_div, num_examples, num_examples);
		SetDiagonalDriver(one_div, 0, num_examples, num_examples);
		CopyDriver(one_div, num, num_examples*num_examples);
		float num_sum;
		ReduceSumDiver(num, &num_sum, num_examples);
		float num_div[num_examples*num_examples];
		DivideDriver(num, num_sum, num_div, num_examples*num_examples);
		float Q[num_examples*num_examples];
		MaxASDriver(num_div, 0.000000000001, Q, num_examples, num_examples);
		float PQ[num_examples*num_examples];
		SubDriver(P, Q, PQ, num_examples, num_examples);
		for(int i = 0; i < n; i ++) {
			float PQ_col[num_examples];
			GetCol(PQ, i);
			float num_col[num_examples];
			GetCol(num, i);
			float mul_pq_num[num_examples];
			MultiplyDriver(PQ_col, num_row, mul_pq_num, num_examples);
			float mul_pq_num_stack[num_examples*num_dimension];
			StackDriver(mul_pq_num, mul_pq_num_stack, num_dimension, num_examples);
			float mul_pq_num_stack_tr[num_examples*num_dimension];
			TransposeDriver(mul_pq_num_stack, mul_pq_num_stack_tr, num_dimension, num_examples);
			float Y_row[num_dimension];
			GetRow(Y, Y_row, i);
			float Y_row_st[num_examples*num_dimension];
			StackDriver(Y_row, Y_row_st, num_examples, num_dimension);
			float sub_y_y_st[num_examples*num_dimension];
			SubDriver(Y_row_st, Y, sub_y_y_st, num_examples, num_dimension);
			float mul_grad[num_examples*num_dimension];
			MultiplyDriver(sub_y_y_st, mul_pq_num_stack_tr, mul_grad, num_examples*num_dimension);
			float mul_grad_tr[num_examples*num_dimension];
			TransposeDriver(mul_grad, mul_grad_tr, num_examples, num_dimension);
			float row_grad[num_dimension];
			ReduceSumDiver(mul_grad_tr, row_grad, num_dimension, num_examples, num_dimension, 1);
			SetRow(Dy, row_grad, num_examples, num_dimension, i);
		}	
		float momentum;
		if(iter<20){
			momentum = initial_momentum;
		}else{
			momentum = final_momentum;
		}

		//CALCULATE GAINS
		float gains_add[num_examples*num_dimension];
		float gains_mul[num_examples*num_dimension];
		AddDriver(gains, 0.2, gains_add, num_examples*num_dimension);
		MultiplyDriver(gains, 0.8, gains_mul, num_examples*num_dimension);
		int iY_gr_zero[num_examples*num_dimension];
		IsGreaterThanDriver(iY, 0, iY_gr_zero, num_examples*num_dimension);
		int dY_gr_zero[num_examples*num_dimension];
		IsGreaterThanDriver(dY, 0, dY_gr_zero, num_examples*num_dimension);
		int iY_dY_eq[num_examples*num_dimension];
		IsEqualDriver(dY_gr_zero, iY_gr_zero, iY_dY_eq, num_examples*num_dimension);
		int iY_dY_neq[num_examples*num_dimension];
		IsNotEqualDriver(dY_gr_zero, iY_gr_zero, iY_dY_neq, num_dimension*num_examples);
		float gain1[num_examples*num_dimension];
		MultiplyDriver(gains_add, iY_dY_neq, gain1, num_dimension*num_examples);
		float gain2[num_examples*num_dimension];
		MultiplyDriver(gains_mul, iY_dY_eq, gain2, num_examples*num_dimension);
		float gain_final_add[num_examples*num_dimension];
		AddDriver(gain1, gain2, gain_final_add, num_examples*num_dimension);
		CopyDriver(gain_final_add, gains, num_examples*num_dimension);
		SetWhereLessThanDriver(gains, min_gain, min_gain, num_examples*num_dimension);
		float gains_dy[num_dimension*num_examples];
		MultiplyDriver(gains, dY, gains_dy, num_examples*num_dimension);
		float gains_dy_eta[num_examples*num_dimension];
		MultiplyDriver(gains_dy, (float)eta, gains_dy_eta, num_examples*num_dimension );
		float momentum_iy[num_examples*num_dimension];
		MultiplyDriver(iY, momentum, momentum_iy, num_examples*num_dimension);
		float iY_add[num_examples*num_dimension];
		AddDriver(momentum_iy, gains_dy_eta, iY_add, num_dimension*num_examples);
		CopyDriver(iY_add, iY, num_examples*num_dimension);
		float Y_add[num_examples*num_dimension];
		AddDriver(Y, iY, Y_add, num_examples*num_dimension);
		CopyDriver(Y_add, Y, num_examples*num_dimension);
		float Y_tr[num_dimension*num_examples];
		TransposeDriver(Y, Y_tr, num_examples,num_dimension);
		float Y_tr_mean[num_dimension];
		ReduceMeanDriver(Y_tr, Y_tr_mean, num_dimension, num_examples, num_dimension,1);
		float Y_stack[num_examples*num_dimension];
		StackDriver(Y_tr_mean, Y_stack, num_examples, num_dimension);
		float Y_sub[num_examples*num_dimension];
		SubDriver(Y, Y_stack, Y_sub, num_examples, num_dimension);
		CopyDriver(Y_sub, Y, num_examples*num_dimension);
		if((iter+1)%10==0) {
			float div_p_q[num_examples*num_examples];
			DivideDriver(P, Q, div_p_q, num_examples*num_examples);
			float log[num_examples*num_examples];	
			LogDriver(div_p_q, log, num_examples*num_examples);
			float C;
			ReduceSumDiver(log, &C, num_examples*num_examples);
			printf("iteration %d: error is %f\n",iter+1, C );
		}
		if(iter==100){
			float P_d[num_examples*num_examples];
			DivideDriver(P, 4, P_d, num_examples*num_examples);
			CopyDriver(P_d, P, num_examples*num_examples);
		}
	}
}


int main()
{
    load_mnist();
    int cnt = 0;
    for (int i=0; i<784; i++) {
		
        printf("%1.1f ", train_image[0][i]);
		if ((i+1) % 28 == 0) putchar('\n');
	}
}
