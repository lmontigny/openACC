// Classical Gram-Schmidt Rob Farber
/*
This code fills a matrix with random numbers 
and then performs a Classical Gram-Schmidt orthogonalization. 
The orthogonality of the results is checked.

Source: http://www.drdobbs.com/parallel/the-openacc-execution-model/240006334?pgno=2
/*
#include <stdio.h>
#include <stdlib.h>
#include <accelmath.h>
#include <omp.h>
 
#ifndef ROWS
#define ROWS 640
#endif
#ifndef COLS
#define COLS 640
#endif
 
// The _OPENACC variable defined by the OpenACC compiler.
#ifdef _OPENACC
 
// Use the OpenACC implementation if _OPENACC is defined
void gramSchmidt(restrict float Q[][COLS], const int rows, const int cols) 
{
#pragma acc data copy(Q[0:rows][0:cols])
  for(int k=0; k < cols; k++) {
    double tmp = 0.;
#pragma acc parallel reduction(+:tmp)
    for(int i=0; i < rows; i++) tmp +=  (Q[i][k] * Q[i][k]);
    tmp = sqrt(tmp);
     
#pragma acc parallel loop
    for(int i=0; i < rows; i++) Q[i][k] /= tmp;
     
#pragma acc parallel loop
    for(int j=k+1; j < cols; j++) {
      tmp=0.;
      for(int i=0; i < rows; i++) tmp += Q[i][k] * Q[i][j];
      for(int i=0; i < rows; i++) Q[i][j] -= tmp * Q[i][k];
    }
  }
}
 
#else 
 
// If _OPENACC is not defined then use OMP/serial code for performance testing
void gramSchmidt(restrict float Q[][COLS], const int rows, const int cols) 
{
  for(int k=0; k < cols; k++) {
    double tmp = 0.;
#pragma omp parallel for reduction(+:tmp)
    for(int i=0; i < rows; i++) tmp +=  (Q[i][k] * Q[i][k]);
    tmp = sqrt(tmp);
 
#pragma omp parallel for
    for(int i=0; i < rows; i++) Q[i][k] /= tmp;
 
#pragma omp parallel for reduction(+:tmp)
    for(int j=k+1; j < cols; j++) {
      tmp=0.;
      for(int i=0; i < rows; i++) tmp += Q[i][k] * Q[i][j];
      for(int i=0; i < rows; i++) Q[i][j] -= tmp * Q[i][k];
    }
  }
}
 
#endif
 
//Allows loading of the matrices into octave/matlab for testing
void printOctave(char* var, float A[][COLS], int rows, int cols)
{
  //  if((rows*cols) > 1000) return;
  printf("%s=[\n",var);
  for(int i=0; i < rows; i++) { 
    for(int j=0; j < cols; j++) printf("%c%e", (j==0)?' ':',', A[i][j]);
    printf(";"); 
  }
  printf("];\n");
}
// checks that the results are orthonogal where they should be
void checkOctave(restrict float A[][COLS], int rows, int cols)
{
  int found_error=0;
  for(int c1=0; c1 < cols; c1++)
    for(int c2=c1; c2 < cols; c2++) {
      double sum=0.;
      for(int i=0; i < rows; i++) sum += A[i][c1] * A[i][c2];
      if(c1 == c2) { // should be near 1 (unit length)
    if(sum < 0.9) {
      printf("Failed unit length: %d %d %g\n", c1,c2,sum);
      found_error = 1;
      exit(1);
    }
      } else { // should be very small (orthogonal)
    if(sum > 0.1) {
      printf("Failed orthonogonal  %d %d %g\n", c1,c2,sum);
      found_error = 1;
      exit(1);
    }
      }
  }
  if(!found_error) printf("Check OK!\n");
}
 
//Simple driver using random values
int main(int argc, char *argv[])
{
  int rows=ROWS;
  int cols=COLS;
  float (*A)[COLS] = (float(*)[COLS])malloc(sizeof(float)*rows*cols);
 
  // fill matrix A with random numbers
  for(int i=0; i < rows; i++)
    for(int j=0; j < cols; j++)
      A[i][j] = (float)rand()/ (float)RAND_MAX;
  printf("Done with init!\n");
 
  //printOctave("A", A, rows, cols);
  double startTime = omp_get_wtime();
  gramSchmidt(A,rows, cols);
  double endTime = omp_get_wtime();
  printf("runtime %d %d matrix %g\n",rows, cols, (endTime-startTime));
  checkOctave(A, rows, cols);
  //printOctave("Q", A, rows, cols);
  free(A);
}
