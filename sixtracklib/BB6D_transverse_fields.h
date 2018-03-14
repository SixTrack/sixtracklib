#ifndef _BB6D_TRANSVERSE_FIELDS_
#define _BB6D_TRANSVERSE_FIELDS_

#ifndef _GPUCODE
  #include <math.h>
  #define CLGLOBAL
#endif

//From sixtraclib
#include "BB_transverse_field_gauss_ellip.h"
#include "BB_transverse_field_gauss_round.h"
#include "constants.h"

void get_Ex_Ey_Gx_Gy_gauss(double x, double  y, 
    double sigma_x, double sigma_y, double min_sigma_diff,
    double* Ex_ptr, double* Ey_ptr, double* Gx_ptr, double* Gy_ptr, bool* flag_4D){
        
    double Ex, Ey, Gx, Gy;
    
    if (fabs(sigma_x-sigma_y)< min_sigma_diff){
        
                
        transv_field_gauss_round_data data;
        data.sigma = 0.5*(sigma_x+sigma_y);
        data.Delta_x = 0.;
        data.Delta_y = 0.;
        
        get_transv_field_gauss_round(&data, x, y, &Ex, &Ey);
        if (!flag_4D) {
          Gx = 1/(2.*(x*x+y*y))*(y*Ey-x*Ex+1./(2*PI*EPSILON_0*data.sigma*data.sigma)
                            *x*x*exp(-(x*x+y*y)/(2.*data.sigma*data.sigma)));
          Gy = 1./(2*(x*x+y*y))*(x*Ex-y*Ey+1./(2*PI*EPSILON_0*data.sigma*data.sigma)
                            *y*y*exp(-(x*x+y*y)/(2.*data.sigma*data.sigma)));
       }
    }
    else{
        transv_field_gauss_ellip_data data;
        data.sigma_x = sigma_x;
        data.sigma_y = sigma_y;
        data.Delta_x = 0.;
        data.Delta_y = 0.;
        
        get_transv_field_gauss_ellip(&data, x, y, &Ex, &Ey);

        double Sig_11 = sigma_x*sigma_x;
        double Sig_33 = sigma_y*sigma_y;
        if (!flag_4D) {
          Gx =-1./(2*(Sig_11-Sig_33))*(x*Ex+y*Ey+1./(2*PI*EPSILON_0)*\
                    (sigma_y/sigma_x*exp(-x*x/(2*Sig_11)-y*y/(2*Sig_33))-1.));
          Gy =1./(2*(Sig_11-Sig_33))*(x*Ex+y*Ey+1./(2*PI*EPSILON_0)*\
                    (sigma_x/sigma_y*exp(-x*x/(2*Sig_11)-y*y/(2*Sig_33))-1.));
        }
    }
                    
    *Ex_ptr = Ex;
    *Ey_ptr = Ey;
    if (!flag_4D) {
      *Gx_ptr = Gx;
      *Gy_ptr = Gy;  
    } 

}


#endif
