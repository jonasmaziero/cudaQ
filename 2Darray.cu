//------------------------------------------------------------------------------
// compile with: nvcc -arch=sm_35 name.cu
//------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
//------------------------------------------------------------------------------
// the kernel
__global__ void matAdd(int *yd, float *Ag, float *Bg, float *Cg) {
  // reverse order of array and gpu idx, to gain speed
  int k = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  *(Cg+j*(*yd)+k) = *(Ag+j*(*yd)+k) + *(Bg+j*(*yd)+k);
}
//------------------------------------------------------------------------------
int main() {
int N, M;
int j, k;
N = 4;
M = 4;
float *A, *B, *C, *An, *Bn, *Cn;
float *Ag, *Bg, *Cg;
int *yd;
int *ydg;
int *ydn;
size_t sz = N*M*sizeof(float);
size_t szi = sizeof(int);

// allocates cpu memory
A = (float *)malloc(sz);
B = (float *)malloc(sz);
C = (float *)malloc(sz);
An = (float *)malloc(sz);
Bn = (float *)malloc(sz);
Cn = (float *)malloc(sz);
yd = (int *)malloc(szi);
*yd = M;
ydn = (int *)malloc(szi);

printf("serial calc \n");
for (j = 0; j < N; j++) {
for (k = 0; k < M; k++) {
  *(A+j*(*yd)+k) = 1.0;
  *(B+j*(*yd)+k) = 1.0;
  *(C+j*(*yd)+k) = *(A+j*(*yd)+k) + *(B+j*(*yd)+k);
printf("A= %f B= %f A+B= %f \n", *(A+j*(*yd)+k),*(B+j*(*yd)+k),*(C+j*(*yd)+k));
}
}

// allocates gpu memory
cudaMalloc(&ydg, szi);
cudaMalloc(&Ag, sz);
cudaMalloc(&Bg, sz);
cudaMalloc(&Cg, sz);

// copy data from cpu's memory to gpu's memory
cudaMemcpy(ydg, yd, szi, cudaMemcpyHostToDevice);
cudaMemcpy(Ag, A, sz, cudaMemcpyHostToDevice);
cudaMemcpy(Bg, B, sz, cudaMemcpyHostToDevice);
cudaMemcpy(Cg, C, sz, cudaMemcpyHostToDevice);

dim3 blocksPerGrid(N/2,M/2,1);
// defines the No. of SMs to be used, for each dimension
dim3 threadsPerBloch(2,2,1);
// defines the No. of cores per SM to be used, for each dimension
matAdd<<<blocksPerGrid, threadsPerBloch>>>(ydg, Ag, Bg, Cg);
// runs the kernel in the gpu
cudaThreadSynchronize(); // to wait to the gpu calc to end

// copy data from gpu's memory to cpu's memory
cudaMemcpy(ydn, ydg, szi, cudaMemcpyDeviceToHost);
cudaMemcpy(An, Ag, sz, cudaMemcpyDeviceToHost);
cudaMemcpy(Bn, Bg, sz, cudaMemcpyDeviceToHost);
cudaMemcpy(Cn, Cg, sz, cudaMemcpyDeviceToHost);

printf("parallel calc \n");
for (j = 0; j < N; j++) {
for (k = 0; k < M; k++) {
printf("An=%f,Bn=%f,An+Bn=%f\n",*(An+j*(*yd)+k),*(Bn+j*(*yd)+k),*(Cn+j*(*yd)+k));
}
}
for(j=0;j<N;j++){
  printf("An= %f Bn= %f An+Bn= %f \n", An[j], Bn[j], Cn[j]);
}

// free gpu memory
cudaFree(Ag); cudaFree(Bg); cudaFree(Cg); cudaFree(ydg);
// free cpu memory
free(A); free(B); free(C); free(An); free(Bn); free(Cn); free(yd); free(ydn);

return 0;
}
//------------------------------------------------------------------------------
