#ifndef _TRANS_FIELD_GUASS_ROUND_
#define _TRANS_FIELD_GUASS_ROUND_

#include "constants.h"

#ifndef _GPUCODE
  #include <math.h>
  #define CLGLOBAL
#endif

typedef struct{
  double sigma;
  double Delta_x;
  double Delta_y;
}transv_field_gauss_round_data;

_CUDA_HOST_DEVICE_
void get_transv_field_gauss_round(CLGLOBAL transv_field_gauss_round_data* data, 
                                  double x, double y, double* Ex, double* Ey){    
  double r2, temp;
  
  r2 = (x-data->Delta_x)*(x-data->Delta_x)+(y-data->Delta_y)*(y-data->Delta_y);
  if (r2<1e-20) temp = sqrt(r2)/(2.*PI*EPSILON_0*data->sigma); //linearised
  else          temp = (1-exp(-0.5*r2/(data->sigma*data->sigma)))/(2.*PI*EPSILON_0*r2);
  
  (*Ex) = temp * (x-data->Delta_x);
  (*Ey) = temp * (y-data->Delta_y);
}

#endif
