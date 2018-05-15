/*
Sequential Stochastic Gradient Descent
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include"definitions.h"
// Read data files (as preprocessed by code)
#define X_matrix "X_ent.txt" //size MxN
#define X_v_matrix "X_valida.txt" //size MxN
#define b_vector "b_bh.txt" //size Nx1
#define y_vector "y_train.txt" //size Mx1
#define y_v_vector "y_val.txt" //size Mx1


// FORTRAN (BLAS) function prototypes used for vector-matrix multiplication and scalar vector multiplication 
extern void dgemv_(char *transpose_a, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern void daxpy_(int *n, double *a, double *x, int *incx, double *y, int *incy);

int main(int argc, char const *argv[]) {

  // Declaration of structures to store data
  array_2d_T X, X_v, batch;
	array_1d_T y, y_v, b, g, y_b, rmse, rmse_v;

  //Get data dimensions from standard input
  int M=atoi(argv[1]);
  int N=atoi(argv[2]);
  int M_v=atoi(argv[3]);
  // Read batch size
  int batch_size=atoi(argv[4]);
  // Read number of iterations
  int iter=atoi(argv[5]);
  // Read learning rate value
  double lr=atof(argv[6]);

	int incx=1;
  double ALPHA, BETA;


// Assign memory space for out data structures
  X=malloc(sizeof(*X));
  X_v=malloc(sizeof(*X_v));
  y=malloc(sizeof(*y));
  y_v=malloc(sizeof(*y_v));
  b=malloc(sizeof(*b));
  g=malloc(sizeof(*g));
  batch=malloc(sizeof(*batch));
  y_b=malloc(sizeof(*y_b));
  rmse=malloc(sizeof(*rmse));
  rmse_v=malloc(sizeof(*rmse_v));
// Assign dimension values for data structures
	rows(X)=M;
	columns(X)=N;
  rows(X_v)=M_v;
	columns(X_v)=N;
  rows_vector(b)=N;
	rows_vector(y)=M;
  rows_vector(y_v)=M_v;
  rows_vector(g)=N;
  rows(batch)=batch_size;
  columns(batch)=N;
  rows_vector(y_b)=batch_size;
  rows_vector(rmse)=iter;
  rows_vector(rmse_v)=iter;

// Firr our structures with our data from .txt files
	values(X)=malloc(rows(X)*columns(X)*sizeof(double));
	fill_matrix(X,X_matrix);

  values(X_v)=malloc(rows(X_v)*columns(X_v)*sizeof(double));
  fill_matrix(X_v,X_v_matrix);

	values_vector(b)=malloc(N*sizeof(double));
	fill_vector(b,b_vector);

  values_vector(y)=malloc(M*sizeof(double));
  fill_vector(y,y_vector);

  values_vector(y_v)=malloc(M_v*sizeof(double));
  fill_vector(y_v,y_v_vector);

	values_vector(g)=malloc(N*sizeof(double));

  values(batch)=malloc(rows(batch)*columns(batch)*sizeof(double));

  values_vector(y_b)=malloc(batch_size*sizeof(double));

  values_vector(rmse)=malloc(iter*sizeof(double));
  values_vector(rmse_v)=malloc(iter*sizeof(double));

// Start  iterations
for(int it = 0; it < iter; it++){ 
// Re-fill our response variable vectors every iteration as they get modified in each one (Details below)
  fill_vector(y,y_vector);
  fill_vector(y_v,y_v_vector);
  
  // Fill batch matrix. This step is what makes this algorithm stochastic as it randomly chooses a fixed number of records to train on every iteration
  fill_batch(batch, X, y_b, y);

// Calculate prediction error: e = - X %*% b + y
// Because of how dgemv function was programmed, vector y is overwrittten by result e (which is why we re-fill the y vector every iteration)
  ALPHA = -1.0;
  BETA = 1.0;
	dgemv_("No transpose", &batch_size, &N, &ALPHA, values(batch), &batch_size, values_vector(b), &incx, &BETA, values_vector(y_b),&incx);
  dgemv_("No transpose", &M, &N, &ALPHA, values(X), &M, values_vector(b), &incx, &BETA, values_vector(y),&incx);
  dgemv_("No transpose", &M_v, &N, &ALPHA, values(X_v), &M_v, values_vector(b), &incx, &BETA, values_vector(y_v),&incx);

// Calculation of the training and validation errors
  double acum = 0;
  for(int i = 0; i < M; i++){
    acum += pow(values_vector(y)[i],2);
  }
  double acum_v = 0;
  for(int i = 0; i < M_v; i++){
    acum_v += pow(values_vector(y_v)[i],2);
  }
  value_vector(rmse,it) = acum/M;
  value_vector(rmse_v,it) = acum_v/M_v;
  printf("Iteration %d/%d RMSE train: %lf -- RMSE val: %lf \n", it+1, iter, value_vector(rmse,it), value_vector(rmse_v,it));
  printf("------------\n");
// Calculating the gradient (for descent direction): g = -X^t %*% e
  ALPHA = -1.0;
  BETA = 0.0;
  dgemv_("Transpose", &batch_size, &N, &ALPHA, values(batch), &batch_size, values_vector(y_b), &incx, &BETA, values_vector(g),&incx);
// now vector g holds the gradient's value

// Update coefficients: b = b - lr * g
  daxpy_(&N, &lr,values_vector(g), &incx, values_vector(b), &incx);
}

printf("----- Final Coefficients -----\n");
print_vector(b);

// Save RMSE train and validation scores for later analysis.
FILE *f = fopen("RMSE_SGD.txt", "w");
if (f == NULL)
{
    printf("Error opening file!\n");
    exit(1);
}
  fprintf(f, "iteration,rmse_t,rmse_v\n");
  for(int i=0; i<iter; i++){
    fprintf(f, "%d,%f,%f\n", i, value_vector(rmse,i),value_vector(rmse_v,i));
  }
fclose(f);

// Free up memory allocated to structures
  free(values(X));
  free(X);
  free(values_vector(b));
  free(b);
  free(values_vector(y));
  free(y);
  free(values_vector(g));
  free(g);
  free(values(X_v));
  free(X_v);
  free(values_vector(y_v));
  free(y_v);
  free(values(batch));
  free(batch);
  free(values_vector(y_b));
  free(y_b);

	return 0;

}
