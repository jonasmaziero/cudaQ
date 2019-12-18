// gcc main.c rpvg.c MT19937_64.c -lm
//------------------------------------------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "MT19937_64.h"
//------------------------------------------------------------------------------------------------------------------------------------
void rpvg_test(){
  unsigned long long int seed;
  srand(time(NULL)); seed = rand();
  void init_genrand64();  init_genrand64(seed);
  int d = 3, j;
  void rpv_zhsl();  double rpv[d];
  for (j = 0; j < d; ++j){
    rpv[j] = 0.0;  //printf("%f \n", rpv[j]);
  }
  rpv_zhsl(&d, rpv);
  for (j = 0; j < d; ++j){
    printf("%f \n", rpv[j]);
  }
  //print('normalization',np.sum(rpv))'''
  /*ns = 10**4
  ni = 40
  delta = 1.0/ni
  avg_rpv = np.zeros(d)
  ct = np.zeros((ni,d))
  for j in range(1, ns):
    rpv = rpv_zhsl(d)
    avg_rpv = avg_rpv + rpv
    for k in range(0, d):
      if rpv[k] == 1.0:
        rpv[k] = 1.0 - 1/(10**10)
      for l in range(0, ni):
        if rpv[k] >= l*delta and rpv[k] < (l+1)*delta:
          ct[l][k] = ct[l][k] + 1
  avg_rpv = avg_rpv/ns
  if ( d < 5 ):
    print('avg_rpv = ', avg_rpv)
  x = np.zeros(ni);  y1 = np.zeros(ni);  y2 = np.zeros(ni);  y3 = np.zeros(ni)
  for l in range(0, ni):
    x[l] = l*delta;  y1[l] = ct[l][0]/ns;  y2[l] = ct[l][1]/ns;  y3[l] = ct[l][2]/ns
  import matplotlib.pyplot as plt;  plt.plot(x,y1,label='p0');  plt.plot(x,y2,label='p1');  plt.plot(x,y3,label='p2')
  axes = plt.gca();  axes.set_xlim([0,1]);  axes.set_ylim([0,0.1])
  plt.xlabel('pj');  plt.ylabel('');  plt.legend();  plt.show()*/
}
//------------------------------------------------------------------------------------------------------------------------------------
void rpv_zhsl(int *d, double *rpv){
  double genrand64_real1();
  double rn[(*d)], norm;
  int j;
  for (j = 0; j < ((*d)-1); ++j){
    rn[j] = genrand64_real1();  //printf("%f \n", rn[j]);
  }
  rpv[0] = 1.0 - pow(rn[0],1.0/((*d)-1.0));
  norm = rpv[0];
  if ((*d) > 2){
    for (j = 1;  j < (*d)-1; ++j){
      rpv[j] = (1.0 - pow(rn[j],1.0/((*d)-j-1)))*(1.0-norm);
      norm = norm + rpv[j];
    }
  }
  rpv[(*d)-1] = 1.0 - norm;
}
//------------------------------------------------------------------------------------------------------------------------------------
