#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include<time.h> //NECESSARIO PARA GERAR NUMEROS ALEATORIOS
#include <complex.h>

//comando para compilar --> nvcc -arch=sm_35 -rdc=true reducao.cu

double cpuSecond(){
  //função para calcular o tempo de execução
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

typedef struct{
  int m;
  double *data;
}MAT;

//------------------------------------------------------------------------------


__device__ void warpReduce(volatile double* sdata, int tid) {
sdata[tid] += sdata[tid + 32];
sdata[tid] += sdata[tid + 16];
sdata[tid] += sdata[tid + 8];
sdata[tid] += sdata[tid + 4];
sdata[tid] += sdata[tid + 2];
sdata[tid] += sdata[tid + 1];
}

__global__ void tracoDevice(MAT a,MAT b) {
  extern __shared__ double q[];
  int t = threadIdx.x;
  int w = blockDim.x*blockIdx.x+ threadIdx.x;
  q[threadIdx.x] = a.data[w];
  __syncthreads();
    for (int i = blockDim.x/2; i > 32; i>>=1) {
        if (t<i) {
          q[t] += q[t+i];
        }
      __syncthreads();
    }
    __syncthreads();

    if (t < 32) {
      warpReduce(q, t);
    }
    __syncthreads();

    if (t == 0) {

    b.data[blockIdx.x] = q[0];

    }
    __syncthreads();
}



/*__global__ void tracoDevice(MAT ab, MAT a) {
  extern __shared__ double q[];
  q[threadIdx.x] = ab.data[blockDim.x*blockIdx.x+threadIdx.x];
  __syncthreads();
  int t = threadIdx.x;
  for (int i = blockDim.x/2; i > 0; i>>=1) {
    if (t<i) {
      q[t] = q[2*t] + q[2*t+1];
    }
    __syncthreads();
  }
  if (t == 0) {
    a.data[blockIdx.x] = q[0];
  }
}*/


void tracoDeviceB(double _Complex *ab, double _Complex *a, int m, int n, int dda,int ddb) {
  MAT habR,habI;
  habR.m = dda*dda*ddb;
  habI.m = dda*dda*ddb;
  habR.data = (double *)malloc(habR.m*sizeof(double));
  habI.data = (double *)malloc(habI.m*sizeof(double));

  int dx = 0;
  for (int i = 0; i < dda; i++) {
    for (int j = 0; j < dda; j++) {
      for (int k = 0; k < ddb; k++) {
        habR.data[dx] = creal(*(ab+(i*ddb+k)*m+(j*ddb+k)));
        habI.data[dx] = cimag(*(ab+(i*ddb+k)*m+(j*ddb+k)));
        dx++;
      }
    }
  }

  MAT dabR, daR;
  dabR.m = habR.m;
  daR.m = dda*dda;
  cudaMalloc(&daR.data, daR.m*sizeof(double));
  cudaMalloc(&dabR.data, dabR.m*sizeof(double));
  cudaMemcpy(dabR.data, habR.data, habR.m*sizeof(double), cudaMemcpyHostToDevice);
  free(habR.data);
  dim3 grid(dda*dda);
  dim3 block(ddb);
  double t1, t2;
  t1 = cpuSecond();
  tracoDevice<<<grid,block, ddb*sizeof(double)>>>(dabR,daR);
  cudaDeviceSynchronize();
  t2 = cpuSecond() -t1;
  cudaFree(dabR.data);
  MAT haR;
  haR.m = daR.m;
  haR.data = (double *)malloc(haR.m*sizeof(double));
  cudaMemcpy(haR.data, daR.data, daR.m*sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(daR.data);

  MAT dabI, daI;
  dabI.m = habI.m;
  daI.m = dda*dda;
  cudaMalloc(&daI.data, daI.m*sizeof(double));
  cudaMalloc(&dabI.data, dabI.m*sizeof(double));
  cudaMemcpy(dabI.data, habI.data, habI.m*sizeof(double), cudaMemcpyHostToDevice);
  free(habI.data);
  dim3 grid1(dda*dda);
  dim3 block1(ddb);
  double t3,t4;
  t3 = cpuSecond();
  tracoDevice<<<grid1,block1, ddb*sizeof(double)>>>(dabI,daI);
  cudaDeviceSynchronize();
  t4 = cpuSecond() - t3;
  cudaFree(dabI.data);
  MAT haI;
  haI.m = daI.m;
  haI.data = (double *)malloc(haI.m*sizeof(double));
  cudaMemcpy(haI.data, daI.data, daI.m*sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(daI.data);
 printf("tempo GPU:  %f\n", t4+t2 );
  for (int i = 0; i < dda*dda; i++) {
    *(a+i) = haR.data[i] + I*haI.data[i];
  }

}

//------------------------------------------------------------------------------

void traco(MAT ab, MAT a, int da, int db) {
  for (int i = 0; i < da*da; i++) {
    for (int j = 0; j < db; j++) {
    a.data[i] += ab.data[j+i*db];
  }
}
}

void tracoHost(double _Complex *ab, double _Complex *a, int m, int n, int dda,int ddb) {
  MAT abR, abI;
  abR.m = dda*dda*ddb;
  abI.m = dda*dda*ddb;
  abR.data = (double *)malloc(abR.m*sizeof(double));
  abI.data = (double *)malloc(abI.m*sizeof(double ));

  int dx = 0;
  for (int i = 0; i < dda; i++) {
    for (int j = 0; j < dda; j++) {
      for (int k = 0; k < ddb; k++) {
        abR.data[dx] = creal(*(ab+(i*ddb+k)*m+(j*ddb+k)));
        abI.data[dx] = cimag(*(ab+(i*ddb+k)*m+(j*ddb+k)));
        dx++;
      }
    }
  }
  MAT aR;
  aR.m = dda*dda;
  aR.data = (double *)malloc(aR.m*sizeof(double));
  traco(abR,aR,dda,ddb);
  free(abR.data);
  MAT aI;
  aI.m = dda*dda;
  aI.data = (double *)malloc(aI.m*sizeof(double));
  traco(abI,aI,dda,ddb);
  free(abI.data);

  for (int i = 0; i < dda*dda; i++) {
    *(a+i) = aR.data[i] + I*aI.data[i];
  }

}

void erro(double _Complex *a, double _Complex *b, int da) {
  double s = 0;

  for (int i = 0; i < da; i++) {
    for (int j = 0; j < da; j++) {
      s += pow( cabs(*(a+i*da+j)) - cabs(*(b+i*da+j))  ,2);
    }
  }
  printf("erro %f\n", sqrt(s) );
}


int main(int argc, char const *argv[]) {

int da = pow(2,4); int db = pow(2,10);int m = da*db; int n = da*db;
double _Complex *ab = (double _Complex *)malloc(m*n*sizeof(double _Complex));
double _Complex *a = (double _Complex *)malloc(da*da*sizeof(double _Complex));
double _Complex *b = (double _Complex *)malloc(da*da*sizeof(double _Complex));

for (int i = 0; i < m; i++) {
  for (int j = 0; j < n; j++) {
    *(ab+i*m+j) = (double)(rand()%100)/100 + I*(double)(rand()%100)/100;
  }
}
printf("matriz gerada\n" );
/*for (int i = 0; i < m; i++) {
  for (int j = 0; j < n; j++) {
    printf("%f + I%f\t", creal(*(ab+i*m+j)), cimag(*(ab+i*m+j))  );
  }
  printf("\n" );
}
printf("\n" );
*/

double t1,t2;
t1 = cpuSecond();
tracoHost(ab,a,m, n,da, db);//traco seral
t2 = cpuSecond() - t1;

tracoDeviceB(ab,b,m,n,da,db);//traco paralelo

erro(a,b,da);
printf("tempo CPU: %f \n", t2 );
//printf("tempo GPU:  %f\n", t4 );
cudaDeviceReset();
free(ab);
free(a);
free(b);
  return 0;
}
