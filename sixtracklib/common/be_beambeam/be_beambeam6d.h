#ifndef SIXTRL_COMMON_BE_BEAMBEAM_BE_BEAMBEAM6D_H__
#define SIXTRL_COMMON_BE_BEAMBEAM_BE_BEAMBEAM6D_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_defines.h"
    #include "sixtracklib/common/impl/beam_elements_defines.h"
    #include "sixtracklib/common/impl/buffer_type.h"
    #include "transverse_field_gauss_ellip.h"
    #include "transverse_field_gauss_round.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T*        NS(beambeam6d_real_ptr_t);
typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const*  NS(beambeam6d_real_const_ptr_t);

typedef struct NS(BeamBeam6D)
{
    SIXTRL_UINT64_T                           size      SIXTRL_ALIGN( 8 );
    NS(beambeam6d_real_ptr_t) SIXTRL_RESTRICT data      SIXTRL_ALIGN( 8 );
}
NS(BeamBeam6D);

SIXTRL_FN SIXTRL_STATIC  SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_preset)( NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam );

#if !defined( _GPUCODE )

SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const data_size );

SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const  data_size,
    NS(beambeam6d_real_ptr_t) SIXTRL_RESTRICT input_data );

SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT orig );

#endif /* !defined( _GPUCODE ) */

SIXTRL_FN SIXTRL_STATIC NS(beambeam6d_real_const_ptr_t)
NS(BeamBeam6D_get_const_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC NS(beambeam6d_real_ptr_t) NS(BeamBeam6D_get_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC SIXTRL_UINT64_T NS(BeamBeam6D_get_data_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT beam_beam );

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====             Implementation of inline functions                ===== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #if !defined( _GPUCODE ) || defined( __CUDACC__ )
    #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) || defined( __CUDACC__ ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_preset)( NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam )
{
    if( beam_beam != SIXTRL_NULLPTR )
    {
        beam_beam->size = ( SIXTRL_UINT64_T )0u;
        beam_beam->data   = SIXTRL_NULLPTR;
    }

    return beam_beam;
}

SIXTRL_INLINE NS(beambeam6d_real_const_ptr_t)
NS(BeamBeam6D_get_const_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT beam_beam )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    return beam_beam->data;
}

SIXTRL_INLINE NS(beambeam6d_real_ptr_t)
NS(BeamBeam6D_get_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam )
{
    return ( NS(beambeam6d_real_ptr_t) )NS(BeamBeam6D_get_const_data)( beam_beam );
}

SIXTRL_INLINE SIXTRL_UINT64_T NS(BeamBeam6D_get_data_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT beam_beam )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    return beam_beam->size;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */


// GIANNI: to be moved around

#define mysign(a) (((a) >= 0) - ((a) < 0))

typedef struct{
    double sphi;
    double cphi;
    double tphi;
    double salpha;
    double calpha;
}BB6D_boost_data;

typedef struct{
    double Sig_11_0;
    double Sig_12_0;
    double Sig_13_0;
    double Sig_14_0;
    double Sig_22_0;
    double Sig_23_0;
    double Sig_24_0;
    double Sig_33_0;
    double Sig_34_0;
    double Sig_44_0;
}BB6D_Sigmas;

typedef struct{
    double q_part;
    BB6D_boost_data parboost;
    BB6D_Sigmas Sigmas_0_star;
    double min_sigma_diff;
    double threshold_singular;
    long int N_slices;

    double delta_x; 
    double delta_y; 
    double x_CO; 
    double px_CO; 
    double y_CO; 
    double py_CO; 
    double sigma_CO; 
    double delta_CO; 
    double Dx_sub; 
    double Dpx_sub; 
    double Dy_sub; 
    double Dpy_sub; 
    double Dsigma_sub; 
    double Ddelta_sub;         
    long int enabled;

    double* N_part_per_slice;
    double* x_slices_star;
    double* y_slices_star;
    double* sigma_slices_star;
}BB6D_data;

void BB6D_boost(BB6D_boost_data* data,
                double* x_star, double* px_star, 
                double* y_star, double* py_star,
                double* sigma_star, double*  delta_star){
    
    double sphi = data->sphi;
    double cphi = data->cphi;
    double tphi = data->tphi;
    double salpha = data->salpha;
    double calpha = data->calpha;
    

    double x = *x_star;
    double px = *px_star;
    double y = *y_star;
    double py = *py_star ;              
    double sigma = *sigma_star;
    double delta = *delta_star ; 
    
    double h = delta + 1. - sqrt((1.+delta)*(1.+delta)-px*px-py*py);

    
    double px_st = px/cphi-h*calpha*tphi/cphi;
    double py_st = py/cphi-h*salpha*tphi/cphi;
    double delta_st = delta -px*calpha*tphi-py*salpha*tphi+h*tphi*tphi;

    double pz_st = sqrt((1.+delta_st)*(1.+delta_st)-px_st*px_st-py_st*py_st);
    double hx_st = px_st/pz_st;
    double hy_st = py_st/pz_st;
    double hsigma_st = 1.-(delta_st+1)/pz_st;

    double L11 = 1.+hx_st*calpha*sphi;
    double L12 = hx_st*salpha*sphi;
    double L13 = calpha*tphi;

    double L21 = hy_st*calpha*sphi;
    double L22 = 1.+hy_st*salpha*sphi;
    double L23 = salpha*tphi;

    double L31 = hsigma_st*calpha*sphi;
    double L32 = hsigma_st*salpha*sphi;
    double L33 = 1./cphi;

    double x_st = L11*x + L12*y + L13*sigma;
    double y_st = L21*x + L22*y + L23*sigma;
    double sigma_st = L31*x + L32*y + L33*sigma;  
    
    *x_star = x_st;
    *px_star = px_st;
    *y_star = y_st;
    *py_star = py_st;
    *sigma_star = sigma_st;
    *delta_star = delta_st;
    
}

void BB6D_inv_boost(BB6D_boost_data* data,
                double* x, double* px, 
                double* y, double* py,
                double* sigma, double*  delta){
    
    double sphi = data->sphi;
    double cphi = data->cphi;
    double tphi = data->tphi;
    double salpha = data->salpha;
    double calpha = data->calpha;
    
    double x_st = *x;
    double px_st = *px;
    double y_st = *y;
    double py_st = *py ;              
    double sigma_st = *sigma;
    double delta_st = *delta ; 
    
    double pz_st = sqrt((1.+delta_st)*(1.+delta_st)-px_st*px_st-py_st*py_st);
    double hx_st = px_st/pz_st;
    double hy_st = py_st/pz_st;
    double hsigma_st = 1.-(delta_st+1)/pz_st;

    double Det_L = 1./cphi + (hx_st*calpha + hy_st*salpha-hsigma_st*sphi)*tphi;

    double Linv_11 = (1./cphi + salpha*tphi*(hy_st-hsigma_st*salpha*sphi))/Det_L;
    double Linv_12 = (salpha*tphi*(hsigma_st*calpha*sphi-hx_st))/Det_L;
    double Linv_13 = -tphi*(calpha - hx_st*salpha*salpha*sphi + hy_st*calpha*salpha*sphi)/Det_L;

    double Linv_21 = (calpha*tphi*(-hy_st + hsigma_st*salpha*sphi))/Det_L;
    double Linv_22 = (1./cphi + calpha*tphi*(hx_st-hsigma_st*calpha*sphi))/Det_L;
    double Linv_23 = -tphi*(salpha - hy_st*calpha*calpha*sphi + hx_st*calpha*salpha*sphi)/Det_L;

    double Linv_31 = -hsigma_st*calpha*sphi/Det_L;
    double Linv_32 = -hsigma_st*salpha*sphi/Det_L;
    double Linv_33 = (1. + hx_st*calpha*sphi + hy_st*salpha*sphi)/Det_L;

    double x_i = Linv_11*x_st + Linv_12*y_st + Linv_13*sigma_st;
    double y_i = Linv_21*x_st + Linv_22*y_st + Linv_23*sigma_st;
    double sigma_i = Linv_31*x_st + Linv_32*y_st + Linv_33*sigma_st;

    double h = (delta_st+1.-pz_st)*cphi*cphi;

    double px_i = px_st*cphi+h*calpha*tphi;
    double py_i = py_st*cphi+h*salpha*tphi;

    double delta_i = delta_st + px_i*calpha*tphi + py_i*salpha*tphi - h*tphi*tphi;

    
    *x = x_i;
    *px = px_i;
    *y = y_i;
    *py = py_i;
    *sigma = sigma_i;
    *delta = delta_i;
    
}

void BB6D_propagate_Sigma_matrix(BB6D_Sigmas* data,
        double S, double threshold_singular, long int handle_singularities,
        double* Sig_11_hat_ptr, double* Sig_33_hat_ptr, 
        double* costheta_ptr, double* sintheta_ptr,
        double* dS_Sig_11_hat_ptr, double* dS_Sig_33_hat_ptr, 
        double* dS_costheta_ptr, double* dS_sintheta_ptr){
            
    double Sig_11_0 = data->Sig_11_0;
    double Sig_12_0 = data->Sig_12_0;
    double Sig_13_0 = data->Sig_13_0;
    double Sig_14_0 = data->Sig_14_0;
    double Sig_22_0 = data->Sig_22_0;
    double Sig_23_0 = data->Sig_23_0;
    double Sig_24_0 = data->Sig_24_0;
    double Sig_33_0 = data->Sig_33_0;
    double Sig_34_0 = data->Sig_34_0;
    double Sig_44_0 = data->Sig_44_0;
    
    // Propagate sigma matrix
    double Sig_11 = Sig_11_0 + 2.*Sig_12_0*S+Sig_22_0*S*S;
    double Sig_33 = Sig_33_0 + 2.*Sig_34_0*S+Sig_44_0*S*S;
    double Sig_13 = Sig_13_0 + (Sig_14_0+Sig_23_0)*S+Sig_24_0*S*S;
    
    double Sig_12 = Sig_12_0 + Sig_22_0*S;
    double Sig_14 = Sig_14_0 + Sig_24_0*S;
    double Sig_22 = Sig_22_0 + 0.*S;
    double Sig_23 = Sig_23_0 + Sig_24_0*S;
    double Sig_24 = Sig_24_0 + 0.*S;
    double Sig_34 = Sig_34_0 + Sig_44_0*S;
    double Sig_44 = Sig_44_0 + 0.*S;
    
    double R = Sig_11-Sig_33;
    double W = Sig_11+Sig_33;
    double T = R*R+4*Sig_13*Sig_13;
    
    //evaluate derivatives
    double dS_R = 2.*(Sig_12_0-Sig_34_0)+2*S*(Sig_22_0-Sig_44_0);
    double dS_W = 2.*(Sig_12_0+Sig_34_0)+2*S*(Sig_22_0+Sig_44_0);
    double dS_Sig_13 = Sig_14_0 + Sig_23_0 + 2*Sig_24_0*S;
    double dS_T = 2*R*dS_R+8.*Sig_13*dS_Sig_13;
    
    double Sig_11_hat, Sig_33_hat, costheta, sintheta, dS_Sig_11_hat, 
           dS_Sig_33_hat, dS_costheta, dS_sintheta, cos2theta, dS_cos2theta;
           
    double signR = mysign(R);
    
    //~ printf("handle: %ld\n",handle_singularities);
    
    if (T<threshold_singular && handle_singularities){
        
        double a = Sig_12-Sig_34;
        double b = Sig_22-Sig_44;
        double c = Sig_14+Sig_23;
        double d = Sig_24;
        
        double sqrt_a2_c2 = sqrt(a*a+c*c);
        
        if (sqrt_a2_c2*sqrt_a2_c2*sqrt_a2_c2 < threshold_singular){
        //equivalent to: if np.abs(c)<threshold_singular and np.abs(a)<threshold_singular:
            
            if (fabs(d)> threshold_singular){
                cos2theta = fabs(b)/sqrt(b*b+4*d*d);
                }
            else{
                cos2theta = 1.;
                } // Decoupled beam
            
            costheta = sqrt(0.5*(1.+cos2theta));
            sintheta = mysign(b)*mysign(d)*sqrt(0.5*(1.-cos2theta));
            
            dS_costheta = 0.;
            dS_sintheta = 0.;
            
            Sig_11_hat = 0.5*W;
            Sig_33_hat = 0.5*W;
            
            dS_Sig_11_hat = 0.5*dS_W;
            dS_Sig_33_hat = 0.5*dS_W;
        }
        else{
            //~ printf("I am here\n");
            //~ printf("a=%.2e c=%.2e\n", a, c);
            sqrt_a2_c2 = sqrt(a*a+c*c); //repeated?
            cos2theta = fabs(2.*a)/(2*sqrt_a2_c2);
            costheta = sqrt(0.5*(1.+cos2theta));
            sintheta = mysign(a)*mysign(c)*sqrt(0.5*(1.-cos2theta));
            
            dS_cos2theta = mysign(a)*(0.5*b/sqrt_a2_c2-a*(a*b+2.*c*d)/(2.*sqrt_a2_c2*sqrt_a2_c2*sqrt_a2_c2));
            
            dS_costheta = 1./(4.*costheta)*dS_cos2theta;
            if (fabs(sintheta)>threshold_singular){ 
            //equivalent to: if np.abs(c)>threshold_singular:
                dS_sintheta = -1./(4.*sintheta)*dS_cos2theta;
            }
            else{
                dS_sintheta = d/(2.*a);
            }
                
            Sig_11_hat = 0.5*W;
            Sig_33_hat = 0.5*W;
            
            dS_Sig_11_hat = 0.5*dS_W + mysign(a)*sqrt_a2_c2;
            dS_Sig_33_hat = 0.5*dS_W - mysign(a)*sqrt_a2_c2;
        }
    }
    else{
        
        double sqrtT = sqrt(T);
        cos2theta = signR*R/sqrtT;
        costheta = sqrt(0.5*(1.+cos2theta));
        sintheta = signR*mysign(Sig_13)*sqrt(0.5*(1.-cos2theta));

        //in sixtrack this line seems to be different different
        // sintheta = -mysign((Sig_11-Sig_33))*np.sqrt(0.5*(1.-cos2theta))

        Sig_11_hat = 0.5*(W+signR*sqrtT);
        Sig_33_hat = 0.5*(W-signR*sqrtT);

        dS_cos2theta = signR*(dS_R/sqrtT - R/(2*sqrtT*sqrtT*sqrtT)*dS_T);
        dS_costheta = 1./(4.*costheta)*dS_cos2theta;
        
        if (fabs(sintheta)<threshold_singular && handle_singularities){
        //equivalent to to np.abs(Sig_13)<threshold_singular
            dS_sintheta = (Sig_14+Sig_23)/R;
        }
        else{
            dS_sintheta = -1./(4.*sintheta)*dS_cos2theta;
        }
            
        dS_Sig_11_hat = 0.5*(dS_W + signR*0.5/sqrtT*dS_T);
        dS_Sig_33_hat = 0.5*(dS_W - signR*0.5/sqrtT*dS_T);
    }
    
    *Sig_11_hat_ptr = Sig_11_hat;
    *Sig_33_hat_ptr = Sig_33_hat;
    *costheta_ptr = costheta;
    *sintheta_ptr = sintheta;
    *dS_Sig_11_hat_ptr = dS_Sig_11_hat;
    *dS_Sig_33_hat_ptr = dS_Sig_33_hat;
    *dS_costheta_ptr = dS_costheta;
    *dS_sintheta_ptr = dS_sintheta;
    
}

void get_Ex_Ey_Gx_Gy_gauss(double x, double  y, 
    double sigma_x, double sigma_y, double min_sigma_diff,
    double* Ex_ptr, double* Ey_ptr, double* Gx_ptr, double* Gy_ptr){
        
    double Ex, Ey, Gx, Gy;
    
    if (fabs(sigma_x-sigma_y)< min_sigma_diff){
        
                
        transv_field_gauss_round_data data;
        data.sigma = 0.5*(sigma_x+sigma_y);
        data.Delta_x = 0.;
        data.Delta_y = 0.;
        
        get_transv_field_gauss_round(&data, x, y, &Ex, &Ey);

        Gx = 1/(2.*(x*x+y*y))*(y*Ey-x*Ex+1./(2*MYPI*EPSILON_0*data.sigma*data.sigma)
                            *x*x*exp(-(x*x+y*y)/(2.*data.sigma*data.sigma)));
        Gy = 1./(2*(x*x+y*y))*(x*Ex-y*Ey+1./(2*MYPI*EPSILON_0*data.sigma*data.sigma)
                            *y*y*exp(-(x*x+y*y)/(2.*data.sigma*data.sigma)));
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
        
        Gx =-1./(2*(Sig_11-Sig_33))*(x*Ex+y*Ey+1./(2*MYPI*EPSILON_0)*\
                    (sigma_y/sigma_x*exp(-x*x/(2*Sig_11)-y*y/(2*Sig_33))-1.));
        Gy =1./(2*(Sig_11-Sig_33))*(x*Ex+y*Ey+1./(2*MYPI*EPSILON_0)*\
                    (sigma_x/sigma_y*exp(-x*x/(2*Sig_11)-y*y/(2*Sig_33))-1.));
    }
                    
    *Ex_ptr = Ex;
    *Ey_ptr = Ey;
    *Gx_ptr = Gx;
    *Gy_ptr = Gy;    

}




#endif /* SIXTRL_COMMON_BE_BEAMBEAM_BE_BEAMBEAM6D_H__ */

/* sixtracklib/common/be_beambeam/be_beambeam6d.h */
