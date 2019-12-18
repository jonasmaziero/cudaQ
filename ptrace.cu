#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

__device__ void trace_subm(int j, int k, int *daG, int *dbG, double *AB, double *A){
  int l;
  for(l=0; l<(*dbG); l++){
       *(A+j*(*daG)+k) += *(AB+j*(*dbG)+l+k*(*dbG)+l);
  }
}

__global__ void ptrBp(int *daG, int *dbG, double *ABg, double *Ag) {
  int k = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  trace_subm(j, k, daG, dbG, ABg, Ag);
}

__global__ void AmB(int *nc, double *A, double *B, double *D) {
  // returns A-B
  int k = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  *(D+j*(*nc)+k) = *(A+j*(*nc)+k) - *(B+j*(*nc)+k)
}
//------------------------------------------------------------------------------
int main() {
size_t szi = sizeof(int);
int *nr, *nc;
nr = (int *)malloc(szi); nc = (int *)malloc(szi);
void array_display(int *, int *, double *);
int *da, *db, *d;
da = (int *)malloc(szi); db = (int *)malloc(szi); d = (int *)malloc(szi);
*da = 2; *db = 3; *d = (*da)*(*db);
double *AB;
size_t szAB = (*d)*(*d)*sizeof(double);
AB = (double *)malloc(szAB);
size_t szA = (*da)*(*da)*sizeof(double);

printf("mat gen \n");
void matAB(int *, double *);
matAB(d, AB);
*nr = *d; *nc = *d;
array_display(nr, nc, AB);

printf("serial calc \n");
double *A;
A = (double *)malloc(szA);
void ptrBs(int *, int *, double *, double *);
ptrBs(da, db, AB, A);
*nr = *da; *nc = *da;
array_display(nr, nc, A);

printf("parallel calc \n");
double *Ag, *ABg, *Ap;
Ap = (double *)malloc(szA);
int *daG, *dbG;
cudaMalloc(&daG, szi);
cudaMalloc(&dbG, szi);
cudaMalloc(&Ag, szA);
cudaMalloc(&ABg, szAB);
cudaMemcpy(daG, da, szi, cudaMemcpyHostToDevice);
cudaMemcpy(dbG, db, szi, cudaMemcpyHostToDevice);
cudaMemcpy(ABg, AB, szAB, cudaMemcpyHostToDevice);
dim3 blocksPerGrid(1,1);
dim3 threadsPerBloch(*da,*da);
ptrBp<<<blocksPerGrid, threadsPerBloch>>>(daG, dbG, ABg, Ag);
//cudaThreadSynchronize(); // to wait to the gpu calc to end
cudaDeviceSynchronize(); //mudei esta linha // pq?
cudaMemcpy(Ap, Ag, szA, cudaMemcpyDeviceToHost);
*nr = *da; *nc = *da;
array_display(nr, nc, Ap);

double dist;
//void distance_frobenius(int *, int *, double *, double *, double *);
//distance_frobenius(da, da, A, Ap, &dist);
void distance_frobenius_gpu(int *, int *, double *, double *, double *)
void distance_frobenius_gpu(nr, nc, A, B, &dist)
printf("||A-Ap|| = %lf \n",dist);

cudaFree(daG); cudaFree(dbG); cudaFree(Ag); cudaFree(ABg);
free(da); free(db); free(d); free(A); free(AB); free(Ap);
cudaDeviceReset();
return 0;
}
//------------------------------------------------------------------------------
void matAB(int *d, double *AB){
int j,k;
for (j = 0; j < (*d); j++) {
  for (k = 0; k < (*d); k++) {
    *(AB+j*(*d)+k) = 1.0;
  }
}
}
//------------------------------------------------------------------------------
void ptrBs(int *da, int *db, double *AB, double *A) {
int ja, ka, jb, jau, kau, jaau;
int d = (*da)*(*db);
for (ja = 0; ja < (*da); ja++) {
  jau = ja*(*db);
  jaau = ja*(*da);
  for (ka = ja; ka < (*da); ka++) {
    kau = ka*(*db);
    *(A+jaau+ka) = 0.0;
    for (jb = 0; jb < (*db); jb++) {
      *(A+jaau+ka) += *(AB+kau+jb+(jau+jb)*d);
    }
    if (ja != ka) {
      *(A+ka*(*da)+ja) = *(A+jaau+ka);
    }
  }
}
}
//------------------------------------------------------------------------------
void array_display(int *nr, int *nc, double *A){
  int j, k;
  for (j = 0; j < (*nr); j++) {
    for (k = 0; k < (*nc); k++) {
      printf("%lf \t",*(A+j*(*nc)+k));
    }
    printf("\n");
  }
}
//------------------------------------------------------------------------------
void norm_frobenius(int *nr, int *nc, double *A, double *norm){
  *norm = 0;
  for (int j = 0; j < (*nr); j++) {
    for (int k = 0; k < (*nc); k++) {
      *norm += pow(*(A+j*(*nc)+k),2) + pow(*(A+j*(*nc)+k),2);
      //*norm += pow(creal(*(A+j*(*nc)+k)),2) + pow(cimag(*(A+j*(*nc)+k)),2);
    }
  }
  *norm = sqrt(*norm);
}
//------------------------------------------------------------------------------
void distance_frobenius(int *nr, int *nc, double *A, double *B, double *distance){
  size_t szD = (*nr)*(*nc)*sizeof(double);
  double *D;
  D = (double *)malloc(szD);
  for (int j = 0; j < (*nr); j++) {
    for (int k = 0; k < (*nc); k++) {
      *(D+j*(*nc)+k) = *(A+j*(*nc)+k) - *(B+j*(*nc)+k);
    }
  }
  void norm_frobenius(int *, int *, double *, double *);
  norm_frobenius(nr, nc, D, distance);
}
//------------------------------------------------------------------------------
void distance_frobenius_gpu(int *nr, int *nc, double *A, double *B, double *distance){
  size_t szD = (*nr)*(*nc)*sizeof(double);
  double *Ag, *Bg, *Dg, *Dp;
  size_t szi = sizeof(int);
  int *ncG;
  Dp = (double *)malloc(szD);
  cudaMalloc(&Ag, szD);
  cudaMalloc(&Bg, szD);
  cudaMalloc(&Dg, szD);
  cudaMalloc(&ncG,szi);
  cudaMemcpy(Ag, A, szD, cudaMemcpyHostToDevice);
  cudaMemcpy(Bg, B, szD, cudaMemcpyHostToDevice);
  dim3 blocksPerGrid(1,1);
  dim3 threadsPerBloch(*nr,*nc);
  AmB<<<blocksPerGrid, threadsPerBloch>>>(ncG, Ag, Bg, Dg);
  cudaDeviceSynchronize();
  cudaMemcpy(Dp, Dg, szD, cudaMemcpyDeviceToHost);
  void norm_frobenius(int *, int *, double *, double *);
  norm_frobenius(nr, nc, Dp, distance);
}
//------------------------------------------------------------------------------
