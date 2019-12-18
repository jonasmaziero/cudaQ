#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuComplex.h>
#include <time.h>
#include <complex.h>
#include <sys/time.h>
#define PI 3.141592
//------------------------------------------------------------------------------
typedef struct{
  int m;
  int n;
  int da;
  int db;
  cuDoubleComplex *data;
}MAT;
//------------------------------------------------------------------------------

double cpuSecond(){
  //função para calcular o tempo de execução
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

//------------------------------------------------------------------------------
__device__ void tracoDevice(MAT ab, MAT a, int l, int m){
 MAT hab;
 hab.m = ab.db;
 hab.data = (cuDoubleComplex *)malloc(hab.m*sizeof(cuDoubleComplex));
 for (int k = 0; k < ab.db; k++) {
   hab.data[k] =  ab.data[ (l*ab.db+k)*ab.m+(m*ab.db+k) ];
   }
 for (int i = ab.db/2; i > 0; i>>=1) {
   for (int j = 0; j < i; j++) {
     hab.data[j] = cuCadd(hab.data[2*j], hab.data[2*j+1]);
  }
  a.data[l*ab.da+m] = hab.data[0];
}
free(hab.data);
}
__global__ void tracoParalelo(MAT ab, MAT a) {
 int i = blockIdx.x*blockDim.x+threadIdx.x;
 int j = blockIdx.y*blockDim.y+threadIdx.y;
 tracoDevice(ab,a,i,j);
}
//------------------------------------------------------------------------------
void cuTraco(MAT ab , MAT a) {
 MAT dab, da;
 dab.m = ab.m;
 dab.n = ab.n;
 dab.da = ab.da;
 dab.db = ab.db;
 cudaMalloc(&dab.data, dab.m*dab.n*sizeof(cuDoubleComplex));
 da.m = a.m;
 da.n = a.n;
 da.da = a.da;
 da.db = a.db;
 cudaMalloc(&da.data, da.m*da.n*sizeof(cuDoubleComplex));
 cudaMemcpy(dab.data, ab.data, dab.m*dab.n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
 dim3 grid(1,1);
 dim3 block(ab.da, ab.da);
 double siStart = cpuSecond();
 tracoParalelo<<<grid, block>>>(dab,da);
 cudaDeviceSynchronize();
 double siElaps = cpuSecond()- siStart;
 printf("\n" );
 printf("tempo GPU01 %f sec\n", siElaps );
 cudaMemcpy(a.data, da.data, da.m*da.n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost );
cudaFree(dab.data); cudaFree(da.data);
}

//------------------------------------------------------------------------------
//calcula o traço parcial de numeros complexos serial
void traco(MAT ab, MAT a, int l, int m) {
  MAT hab;
  hab.m = ab.db;
  hab.data = (cuDoubleComplex *)malloc(hab.m*sizeof(cuDoubleComplex));
  for (int k = 0; k < ab.db; k++) {
    hab.data[k] =  ab.data[ (l*ab.db+k)*ab.m+(m*ab.db+k) ];

  }
  for (int i = ab.db/2; i > 0; i>>=1) {
    for (int j = 0; j < i; j++) {
      hab.data[j] = cuCadd(hab.data[2*j], hab.data[2*j+1]);
   }
   a.data[l*ab.da+m] = hab.data[0];

  }
free(hab.data);
}

//------------------------------------------------------------------------------
__global__ void tracoDinamico(MAT ab ,MAT a) {
  extern __shared__ cuDoubleComplex q[];
  __syncthreads();
  q[threadIdx.x] = ab.data[ (blockIdx.x*ab.db+threadIdx.x)*ab.m + (blockIdx.y*ab.db+threadIdx.x) ];
  __syncthreads();
  int t = threadIdx.x;
    if (threadIdx.y==0) {
      __syncthreads();
      for (int i = blockDim.x/2; i > 0; i>>=1) {
          if (t<i) {
            __syncthreads();
            q[t] = cuCadd(q[2*t], q[2*t+1]);
            __syncthreads();
          }
        __syncthreads();
      }
      __syncthreads();
      if (t == 0) {
        __syncthreads();
        a.data[blockIdx.x*ab.da+blockIdx.y] = q[0];
        __syncthreads();
        //printf("%i\n", blockIdx.x*ab.da+blockIdx.y );
      //  printf("%lf + i%lf\n", cuCreal(a.data[blockIdx.x*ab.da+blockIdx.y]), cuCimag(a.data[blockIdx.x*ab.da+blockIdx.y]) );
      }
      __syncthreads();
    }
    __syncthreads();


}




//------------------------------------------------------------------------------
void cuTracoDevice(MAT ab, MAT a) {
  MAT dab, da;
  dab.m = ab.m;
  dab.n = ab.n;
  dab.da = ab.da;
  dab.db = ab.db;
  cudaMalloc(&dab.data, dab.m*dab.n*sizeof(cuDoubleComplex));

  da.m = a.m;
  da.n = a.n;
  da.da = a.da;
  da.db = a.db;
  cudaMalloc(&da.data, da.m*da.n*sizeof(cuDoubleComplex));
  cudaMemcpy(dab.data, ab.data, dab.m*dab.n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  dim3 grid(ab.da,ab.da);
  dim3 block(ab.db, 1);

  tracoDinamico<<<grid, block, ab.db*sizeof(cuDoubleComplex)>>>(dab,da);
  cudaDeviceSynchronize();

  cudaMemcpy(a.data, da.data, da.m*da.n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost );
cudaFree(dab.data); cudaFree(da.data);
}


//------------------------------------------------------------------------------
//função usada para informar se a soma foi feito corretamente
void norma(MAT a, MAT b) {
  MAT t;
  t.m = a.m;
  t.n = a.n;
  t.data = (cuDoubleComplex *)malloc(t.m*t.n*sizeof(cuDoubleComplex));
  for (int i = 0; i < a.m; i++) {
    for (int j = 0; j < a.n; j++) {
      t.data[i*a.m+j] = cuCsub(a.data[i*a.m+j],b.data[i*a.m+j]);
      //if (cuCabs(a.data[i*a.m+j]) !=cuCabs(b.data[i*a.m+j])) {

        //printf("%lf + i%lf // %lf + i%lf \n", cuCreal(a.data[i*a.m+j]), cuCimag(a.data[i*a.m+j]), cuCreal(b.data[i*a.m+j]), cuCimag(b.data[i*a.m+j]) );
      //}
        //printf("%lf + i%lf // %lf + i%lf \n", cuCreal(a.data[i*a.m+j]), cuCimag(a.data[i*a.m+j]), cuCreal(b.data[i*a.m+j]), cuCimag(b.data[i*a.m+j]) );
    }
  }
  double s = 0;
  for (int i = 0; i < a.m; i++) {
    for (int j = 0; j < a.n; j++) {
       s += pow(cuCabs( t.data[i*a.m+j]),2);
    }
  }
  printf("erro %f\n", sqrt(s));
}
//------------------------------------------------------------------------------

int main(int argc, char const *argv[]) {
srand(time(NULL));
double t1, t2;
MAT ab , a, c;
ab.da = 64;
ab.db = 64;
ab.m = ab.da*ab.db;
ab.n = ab.da*ab.db;
ab.data = (cuDoubleComplex *)malloc(ab.m*ab.n*sizeof(cuDoubleComplex));


a.m = ab.da;
a.n = ab.da;
a.data = (cuDoubleComplex *)malloc(a.m*a.n*sizeof(cuDoubleComplex));

c.m = ab.da;
c.n = ab.da;
c.data = (cuDoubleComplex *)malloc(c.m*c.n*sizeof(cuDoubleComplex));

for (int i = 0; i < ab.m; i++) {
  for (int j = 0; j < ab.n; j++) {
    ab.data[i*ab.m+j] =  make_cuDoubleComplex ((double)(rand()%10000)/10000 , (double)(rand()%10000)/10000 ) ;
    //printf("%lf + i%lf\t", cuCreal(ab.data[i*ab.m+j]), cuCimag(ab.data[i*ab.m+j]) );
  }
 //printf("\n");
}
//printf("\n" );

void array2Display(MAT ab); //array2Display(ab);

t1 = cpuSecond();
for (int i = 0; i < ab.da; i++) {
  for (int j = 0; j < ab.da; j++) {
    traco(ab,a,i,j);//calcula o traço serial
  }
}
t2 = cpuSecond();
printf("%f\n", t2-t1 );
//scuTraco(ab,b);  array2Display(b);
t1 = cpuSecond();
//cuTracoDevice(ab,c);
cuTraco(ab,c);
t2 = cpuSecond();
printf("%f\n", t2-t1 );
//array2Display(c);
//array2Display(a);

printf("erro serial -- dinamico\n" );
norma(a,c);
free(ab.data);
free(a.data);
//free(b.data);
free(c.data);

cudaDeviceReset();
  return 0;
}
//------------------------------------------------------------------------------


void array2Display(MAT a){
  for (int i = 0; i < a.m; i++) {
    for (int j = 0; j < a.n; j++) {
      printf("%lf + i%lf\t", cuCreal(a.data[i*a.m+j]), cuCimag(a.data[i*a.m+j]) );
    }
   printf("\n");
  }
  printf("\n" );
}
