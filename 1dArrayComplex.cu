//------------------------------------------------------------------------------
// compile with: nvcc -arch=sm_35 name.cu
//------------------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <cuComplex.h>
#include <complex.h>
 //-----------------------------------------------------------------------------
__global__ void vecAdd(double *A, double *B, cuDoubleComplex *C) {
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  *(C+j) = *(A+j) + *(B+j);
}
//------------------------------------------------------------------------------
int main() {
int N;
double *A, *B;
double _Complex *C;
double *Ag, *Bg;
cuDoubleComplex *Cg;
int j;
N = 4;
int *xd;
int *xdg;
int *xdn;
size_t sz = N*sizeof(double);
size_t sz_ = N*sizeof(double _Complex);
size_t szi = sizeof(int);

// allocates cpu memory
A = (double *)malloc(sz);
B = (double *)malloc(sz);
C = (double _Complex *)malloc(sz_);
xd = (int *)malloc(szi);
*xd = N;
xdn = (int *)malloc(szi);

printf("serial calc \n");
for (j = 0; j < N; j++) {
  A[j] = (double)j;  B[j] = 2*(double)j;
  *(C+j) = *(A+j) + *(B+j);
  printf("A= %f B= %f A+B= %f \n", *(A+j), *(B+j), *(C+j));
}

// allocates gpu memory
cudaMalloc(&xdg, szi);  // notice that the pointer to a pointer is sent to cudaMalloc
cudaMalloc(&Ag, sz);
cudaMalloc(&Bg, sz);
cudaMalloc(&Cg, sz_);

// copy data from cpu's memory to gpu's memory
cudaMemcpy(xdg, xd, szi, cudaMemcpyHostToDevice);
cudaMemcpy(Ag, A, sz, cudaMemcpyHostToDevice);
cudaMemcpy(Bg, B, sz, cudaMemcpyHostToDevice);
//cudaMemcpy(Cg, C, sz, cudaMemcpyHostToDevice);

dim3 blocksPerGrid(N/2,1,1);
// defines the No. of SMs to be used, for each dimension
dim3 threadsPerBloch(2,1,1);
// defines the No. of cores per SM to be used, for each dimension
// and runs the kernel in the gpu
vecAdd<<<blocksPerGrid, threadsPerBloch>>>(xdg, Ag, Bg, Cg);
// to wait for the gpu calc to end
cudaThreadSynchronize();

// copy data from gpu's memory to cpu's memory
cudaMemcpy(xdn, xdg, szi, cudaMemcpyDeviceToHost);
cudaMemcpy(A, Ag, sz, cudaMemcpyDeviceToHost);
cudaMemcpy(B, Bg, sz, cudaMemcpyDeviceToHost);
cudaMemcpy(C, Cg, sz, cudaMemcpyDeviceToHost);

printf("parallel calc \n");
for(j = 0; j < N; j++){
  printf("A= %f B= %f A+B+xd= %f \n", *(A+j), *(B+j), *(C+j));
}

// free gpu memory
cudaFree(Ag); cudaFree(Bg); cudaFree(Cg); cudaFree(xdg);
// free cpu memory
free(A); free(B); free(C); free(xd); free(xdn);

return 0;
}
//------------------------------------------------------------------------------
