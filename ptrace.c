#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//codigo para calcular o traço serial e paralelo comparalos e medir o tempo de execução
// author: Lucas Friedrich
//------------------------------------------------------------------------------


double cpuSecond(){
  //função para calcular o tempo de execução
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


//------------------------------------------------------------------------------


void funcErro( int *A, int *B, int d) {
  //input --> matriz A , matriz B, dimansao d
  //output --> informa se a matriz A e B sao iguais
  int dif = 0;
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) {
      dif = *(A + i*d + j) - *(B + i*d + j);
      if (dif !=0) {
        printf("erro--> %i serial--> %i paralelo--> %i\n", dif, *(B + i*d + j), *(A + i*d + j) );
      }
    }
  }
  printf("--------------------------------fim teste---------------------------\n" );
}


//------------------------------------------------------------------------------


__global__ void Tparcial(int *AB, int *A, int const da, int const db ) {
    // retorna --> tr{AB} = <0b|AB|0b> + <1b|AB|1b>
  int k = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int soma = 0, dy = 0,dx = 0, dt = 0;
  dx = db*j;
  dt = db*db*da*k;
    for (int i = 0; i < db; i++) {
      dy = ( da*db + 1 )*i;
      soma += *(AB + dy + dt + dx);
    }
  *(A + k*da + j) = soma;
}


//------------------------------------------------------------------------------


void TparcialB(int *AB, int *B, int da, int db) {
  //input --> matriz qualquer
  //output --> traço da matriz sobre B
  int dy = 0, dx = 0, dt = 0, soma = 0;
  for (int k = 0; k < da; k++) {
    dt = (db*db)*da*k;
    for (int j = 0; j < da; j++) {
      dx = db*j;
      for (int i = 0; i < db; i++) {
        dy = (da*db + 1)*i;
        soma += *(AB + dy + dx + dt);
      }
      *(B + k*da + j) = soma;
      soma = 0;
    }
  }
}

__global__ void Array(int *D, int const d_dy) {
  //função que gera  uma matriz qualquer
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
    *(D + i*d_dy + j ) =  1;
}


//------------------------------------------------------------------------------

void test( ) {

  FILE *serial= fopen("serial2.dat", "w");
  FILE *paralelo = fopen("paralelo2.dat", "w");

  for (int n = 1; n <= 7; n++) {
      int v = pow(2,n);
      int da = v ,  db = v ;
      int d = da*db;
      int *AB = (int *)malloc(d*d*sizeof(int));
      int *DAB; cudaMalloc(&DAB, d*d*sizeof(int));

      dim3 l1(d,d);
      dim3 l2(1,1);

      Array<<<l1,l2>>>(DAB, d);//cria uma matriz qualquer
      cudaDeviceSynchronize();
      int *B =(int *)malloc(da*da*sizeof(int));

      void  TparcialB(int *AB, int *B, int da, int db);

        double siStart = cpuSecond();
/*--->*/TparcialB(AB, B, da, db );//calcula o traço serial
        double siElaps = cpuSecond()- siStart;

        printf("\n" );
        printf("tempo serial %f sec\n", siElaps );
        fprintf(serial, "%f %f\n", (double)v, (double)siElaps );



      int *D_AB ; cudaMalloc(&D_AB, d*d*sizeof(int));
      int *D_B; cudaMalloc(&D_B, da*da*sizeof(int));

      cudaMemcpy(D_AB, AB, d*d*sizeof(int), cudaMemcpyHostToDevice);

      dim3 t1(da/2,da/2);
      dim3 t2(2,2);
      int *A = (int *)malloc(da*da*sizeof(int));
      double iStart = cpuSecond();
      Tparcial<<<t1,t2>>>(D_AB, D_B, da, db);//calcula o traço paralelo
      cudaDeviceSynchronize();
      double iElaps = cpuSecond()- iStart;
      printf("\n" );
      printf("tempo paralelo %f sec\n", iElaps );
      fprintf(paralelo, "%f %f\n",(double)v, (double)iElaps );
      cudaMemcpy(A, D_B, da*da*sizeof(int), cudaMemcpyDeviceToHost);

      void funcErro( int *A, int *B, int d);
        funcErro(A, B, da);//compara o traço serial com paralelo para saber se a erro
}

}
//------------------------------------------------------------------------------
int main(int argc, char const *argv[]) {
    void test();
    test();
  return 0;
}
