#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>


typedef struct {
  int n;
  int m;
  int da;
  int db;
  int *data;
}MAT;



double cpuSecond(){
  //função para calcular o tempo de execução
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void matriz(MAT a) {
  for (int i = 0; i < a.m; i++) {
    for (int j = 0; j < a.n; j++) {
      a.data[i*a.m+j] = (rand()%10);
    }
  }
}

void imprime(MAT a) {
  printf("\n" );
  for (int i = 0; i < a.m; i++) {
    for (int j = 0; j < a.n; j++) {
      printf("%i\t", a.data[i*a.m+j] );
    }
    printf("\n" );
  }
}

void tracoSerial(MAT ab, MAT a) {
  for (size_t i = 0; i < ab.da; i++) {
    for (int j = 0; j < ab.da; j++) {
      for (int k = 0; k < ab.db; k++) {
        a.data[i*ab.da+j] += ab.data[ (i*ab.db+k)*ab.m+(j*ab.db+k) ];
      }
    }
  }
}

void erro(MAT a, MAT b) { // the best way is using the norm of (a-b)=\sum_jk(a-b)_jk^2
  int s = 0;
  for (int i = 0; i < a.m; i++) {
    for (int j = 0; j < a.n; j++) {
      if (a.data[i*a.m+j] != b.data[i*a.m+j]) {
        s++;
      }
    }
  }
  printf("erro %i\n", s);
}
//------------------------------------------------------------------------------
//redução , uma soma por thread
__device__ void sum(MAT ab, MAT a, int i, int j) {
  for (int k = 0; k < ab.db; k++) {
    a.data[i*ab.da+j] += ab.data[(i*ab.db+k)*ab.m+(j*ab.db+k)]  ;
    //printf("i:%i --> %i \n",i*ab.da+j, ab.data[(i*ab.db+k)*ab.m+(j*ab.db+k)]  );
  }
}

__global__ void tracoDevice(MAT ab, MAT a) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  sum(ab,a,i,j);
  //printf("%i %i\n",i,j );
  __syncthreads();
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//redução uma soma por bloco
__global__ void sumD01(MAT a, MAT b) {

  extern __shared__ int q[];
  q[threadIdx.x] = a.data[ (blockIdx.x*a.db+threadIdx.x)*a.m + (blockIdx.y*a.db+threadIdx.x) ];
  __syncthreads();
  int t = threadIdx.x;
    if (threadIdx.y==0) {
      for (int i = blockDim.x/2; i > 0; i>>=1) {
          if (t<i) {
            q[t] += q[t+i];
          }
        __syncthreads();
      }
      __syncthreads();
      if (t == 0) {
        b.data[blockIdx.x*a.da+blockIdx.y] = q[0];
    //  printf("%i\n", q[0]);
  //   printf("teste\n" );
  printf("b.data[%i] = %i\n", blockIdx.x*a.da+blockIdx.y, q[0]);
      }
    }

}



//------------------------------------------------------------------------------

int main(int argc, char const *argv[]) {
  srand(time(NULL));// NECESSARIO PARA GERAR NUMEROS ALEATORIOS
    MAT hab, ha, ha1, ha2, dab, dab1, da, da1;
    //ponteiro matriz reduzida serial
    ha.m = 2;
    ha.db = 8;
    ha.n = ha.m;
    ha.da = ha.m;
    ha.data = (int *)malloc(ha.m*ha.n*sizeof(int));

    //ponteiro matriz reduzida paralelo
    ha1.m = ha.m;
    ha1.n = ha.m;
    ha1.data = (int *)malloc(ha1.m*ha1.n*sizeof(int));

    //ponteiro matriz reduzida redução
    ha2.m = ha.m;
    ha2.n = ha.m;
    ha2.data = (int *)malloc(ha2.m*ha2.n*sizeof(int));

    //ponteiro da matriz AB
    hab.da = ha.m;
    hab.db = ha.db;
    hab.m = hab.da*hab.db;
    hab.n = hab.m;
    hab.data = (int *)malloc(hab.m*hab.n*sizeof(int));

    //ponteiro matriz AB GPU
    dab.m = hab.m;
    dab.n = dab.m;
    dab.da = hab.da;
    dab.db = hab.db;
    cudaMalloc(&dab.data, dab.m*dab.n*sizeof(int));

    //ponteiro matriz AB GPU 01
    dab1.m = hab.m;
    dab1.n = dab.m;
    dab1.da = hab.da;
    dab1.db = hab.db;
    cudaMalloc(&dab1.data, dab1.m*dab1.n*sizeof(int));

    //ponteiro matriz reduzida paralelo
    da.m = ha.m;
    da.n = ha.n;
    cudaMalloc(&da.data, da.m*da.n*sizeof(int));

    //ponteiro matriz reduzida redução
    da1.m = ha.m;
    da1.n = ha.n;
    cudaMalloc(&da.data, da1.m*da1.n*sizeof(int));

    //gera matriz aleatoria
    matriz(hab);
    //imprime(hab);
    printf("matriz criada\n" );

//------------------------------------------------------------------------------
    double siStart = cpuSecond();

      tracoSerial(hab, ha);//traço serial

    double siElaps = cpuSecond()- siStart;
    printf("\n" );
    printf("tempo CPU %f sec\n", siElaps );
    printf("\n" );
   //imprime(ha);
//------------------------------------------------------------------------------
    printf("transferindo dados CPU --> GPU\n" );
      cudaMemcpy(dab.data, hab.data, dab.m*dab.n*sizeof(int), cudaMemcpyHostToDevice);
    printf("dados transferido\n" );

    dim3 grid(1,1);
    dim3 block(ha1.m/grid.x, ha1.n/grid.y);

    double sssiStart = cpuSecond();
      tracoDevice<<<grid, block>>>(dab,da);
      cudaDeviceSynchronize();
    double sssiElaps = cpuSecond()- sssiStart;
    printf("\n" );
    printf("tempo GPU01 %f sec\n", sssiElaps );
    printf("transferindo dados GPU --> CPU\n" );

    cudaMemcpy(ha1.data, da.data, ha1.m*ha1.n*sizeof(int), cudaMemcpyDeviceToHost);
    printf("dados trasferidos\n" );
  // imprime(ha1);

//------------------------------------------------------------------------------
    cudaMemcpy(dab1.data, hab.data, dab1.m*dab1.n*sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid1(ha.da,ha.da);
    dim3 block1(ha.db,1);

    double aasiStart = cpuSecond();

      sumD01<<<grid1,block1, hab.db*sizeof(int)>>>(dab1, da1);
      cudaDeviceSynchronize();
    double aasiElaps = cpuSecond()- aasiStart;
    printf("\n" );
    printf("tempo GPU02 %f sec\n", aasiElaps );

    cudaMemcpy(ha2.data, da1.data, ha.m*ha.n*sizeof(int), cudaMemcpyDeviceToHost);

/*printf("erro paralelo\n" );
erro(ha,ha1);
printf("\n" );
printf("erro redução\n" );
erro(ha,ha2);
*/
imprime(ha2);


free(ha.data);
free(hab.data);
free(ha1.data);
free(ha2.data);
cudaFree(dab.data);
cudaFree(dab1.data);
cudaFree(da.data);
cudaFree(da1.data);


cudaDeviceReset();
  return 0;
}
