// Classical Gram-Schmidt Rob Farber
/*
This code fills a matrix with random numbers 
and then performs a Classical Gram-Schmidt orthogonalization. 
The orthogonality of the results is checked.
Source: http://www.drdobbs.com/parallel/the-openacc-execution-model/240006334?pgno=4
*/

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
 
#ifdef _OPENACC
 
void gramSchmidt(restrict float Q[][COLS], const int rows, const int cols) 
{
  float Qt[cols][rows];
   
#pragma acc data create(Qt[cols][rows]) copy(Q[0:rows][0:cols])
  {
    //transpose Q in Qt
#pragma acc parallel loop
    for(int i=0; i < rows; i++) 
      for(int j=0; j < cols; j++) 
    Qt[j][i] = Q[i][j];
     
    for(int k=0; k < cols; k++) {
       
#pragma acc parallel
      {
    double tmp = 0.;
     
#pragma acc loop vector reduction(+:tmp)  
    for(int i=0; i < rows; i++) tmp +=  (Qt[k][i] * Qt[k][i]);
    tmp = sqrt(tmp);
     
#pragma acc loop vector 
    for(int i=0; i < rows; i++) Qt[k][i] /= tmp;
      }
       
#pragma acc parallel loop vector_length(128)
      for(int j=k+1; j < cols; j++) {
    double tmp=0.;
    for(int i=0; i < rows; i++) tmp += Qt[k][i] * Qt[j][i];
    for(int i=0; i < rows; i++) Qt[j][i] -= tmp * Qt[k][i];
      }
    }
 
#pragma acc parallel loop
    for(int i=0; i < rows; i++) 
      for(int j=0; j < cols; j++) 
    Q[i][j] = Qt[j][i];
  }
}
 
#else
 
// OMP/serial code for performance testing
void gramSchmidt(restrict float Q[][COLS], const int rows, const int cols) 
{
    float Qt[cols][rows];
 
#pragma omp parallel for
    for(int i=0; i < rows; i++) 
      for(int j=0; j < cols; j++) 
    Qt[j][i] = Q[i][j];
     
    for(int k=0; k < cols; k++) {
      double tmp = 0.;
#pragma omp parallel for reduction(+:tmp)
      for(int i=0; i < rows; i++) tmp +=  (Qt[k][i] * Qt[k][i]);
      tmp = sqrt(tmp);
       
#pragma omp parallel for
      for(int i=0; i < rows; i++) Qt[k][i] /= tmp;
       
#pragma omp parallel for reduction(+:tmp)
      for(int j=k+1; j < cols; j++) {
    double tmp=0.;
    for(int i=0; i < rows; i++) tmp += Qt[k][i] * Qt[j][i];
    for(int i=0; i < rows; i++) Qt[j][i] -= tmp * Qt[k][i];
      }
    }
     
#pragma omp parallel for
    for(int i=0; i < rows; i++) 
      for(int j=0; j < cols; j++) 
    Q[i][j] = Qt[j][i];
}
#endif
 
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
#ifndef NO_CHECK
  checkOctave(A, rows, cols);
  //printOctave("Q", A, rows, cols);
#endif
  free(A);
}
