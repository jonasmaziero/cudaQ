//------------------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
//------------------------------------------------------------------------------
__global__ void vecAdd(int *xd, float *Ag, float *Bg, float *Cg) {
  // this is a kernel, which state the computations the gpu shall do
  //int j = threadIdx.x;
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  *(Cg+j) = *(Ag+j) + *(Bg+j) + (*xd);
}
//------------------------------------------------------------------------------
int main() {
int N;
float *A, *B, *C;
float *Ag, *Bg, *Cg;
int j;
N = 4;
int *xd;
int *xdg;
int *xdn;
size_t sz = N*sizeof(float);
size_t szi = sizeof(int);

// allocates cpu memory
A = (float *)malloc(sz);
B = (float *)malloc(sz);
C = (float *)malloc(sz);
xd = (int *)malloc(szi);
*xd = N;
xdn = (int *)malloc(szi);

printf("serial calc \n");
for (j = 0; j < N; j++) {
  A[j] = (float)j;  B[j] = (float)j;
  *(C+j) = *(A+j) + *(B+j) + (*xd);
  printf("A= %f B= %f A+B+xd= %f \n", *(A+j), *(B+j), *(C+j));
}

// allocates gpu memory
cudaMalloc(&xdg, szi);
cudaMalloc(&Ag, sz);
cudaMalloc(&Bg, sz);
cudaMalloc(&Cg, sz);

// copy data from cpu's memory to gpu's memory
cudaMemcpy(xdg, xd, szi, cudaMemcpyHostToDevice);
cudaMemcpy(Ag, A, sz, cudaMemcpyHostToDevice);
cudaMemcpy(Bg, B, sz, cudaMemcpyHostToDevice);
//cudaMemcpy(Cg, C, sz, cudaMemcpyHostToDevice);

dim3 blocksPerGrid(N/2,1,1);
// defines the No. of SMs to be used, for each dimension
dim3 threadsPerBloch(2,1,1);
// defines the No. of cores per SM to be used, for each dimension
vecAdd<<<blocksPerGrid, threadsPerBloch>>>(xdg, Ag, Bg, Cg);
// runs the kernel in the gpu
cudaThreadSynchronize(); // to wait to the gpu calc to end

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
