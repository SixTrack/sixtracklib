#ifndef SIXTRL_COMMON_BE_BEAMBEAM_BE_BEAMBEAM6D_H__
#define SIXTRL_COMMON_BE_BEAMBEAM_BE_BEAMBEAM6D_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/be_beambeam/gauss_fields.h"
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

typedef struct{
    SIXTRL_REAL_T q_part;
    SIXTRL_REAL_T N_part;
    SIXTRL_REAL_T sigma_x;
    SIXTRL_REAL_T sigma_y;
    SIXTRL_REAL_T beta_s;
    SIXTRL_REAL_T min_sigma_diff;
    SIXTRL_REAL_T Delta_x;
    SIXTRL_REAL_T Delta_y;
    SIXTRL_REAL_T Dpx_sub;
    SIXTRL_REAL_T Dpy_sub;
    SIXTRL_INT64_T enabled;
}BB4D_data;


typedef struct{
    SIXTRL_REAL_T sphi;
    SIXTRL_REAL_T cphi;
    SIXTRL_REAL_T tphi;
    SIXTRL_REAL_T salpha;
    SIXTRL_REAL_T calpha;
}BB6D_boost_data;

typedef struct{
    SIXTRL_REAL_T Sig_11_0;
    SIXTRL_REAL_T Sig_12_0;
    SIXTRL_REAL_T Sig_13_0;
    SIXTRL_REAL_T Sig_14_0;
    SIXTRL_REAL_T Sig_22_0;
    SIXTRL_REAL_T Sig_23_0;
    SIXTRL_REAL_T Sig_24_0;
    SIXTRL_REAL_T Sig_33_0;
    SIXTRL_REAL_T Sig_34_0;
    SIXTRL_REAL_T Sig_44_0;
}BB6D_Sigmas;

typedef struct{
    SIXTRL_REAL_T q_part;
    BB6D_boost_data parboost;
    BB6D_Sigmas Sigmas_0_star;
    SIXTRL_REAL_T min_sigma_diff;
    SIXTRL_REAL_T threshold_singular;
    SIXTRL_INT64_T N_slices;

    SIXTRL_REAL_T delta_x;
    SIXTRL_REAL_T delta_y;
    SIXTRL_REAL_T x_CO;
    SIXTRL_REAL_T px_CO;
    SIXTRL_REAL_T y_CO;
    SIXTRL_REAL_T py_CO;
    SIXTRL_REAL_T sigma_CO;
    SIXTRL_REAL_T delta_CO;
    SIXTRL_REAL_T Dx_sub;
    SIXTRL_REAL_T Dpx_sub;
    SIXTRL_REAL_T Dy_sub;
    SIXTRL_REAL_T Dpy_sub;
    SIXTRL_REAL_T Dsigma_sub;
    SIXTRL_REAL_T Ddelta_sub;
    SIXTRL_INT64_T enabled;

    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* N_part_per_slice;
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* x_slices_star;
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* y_slices_star;
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* sigma_slices_star;
}BB6D_data;


SIXTRL_FN SIXTRL_STATIC  SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam );

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

SIXTRL_FN SIXTRL_STATIC void
BB6D_boost(SIXTRL_BE_DATAPTR_DEC BB6D_boost_data* data,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* x_star, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* px_star,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* y_star, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* py_star,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* sigma_star, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T*  delta_star );

SIXTRL_FN SIXTRL_STATIC void BB6D_inv_boost(
        SIXTRL_BE_DATAPTR_DEC BB6D_boost_data* data,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* x, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* px,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* y, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* py,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* sigma, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T*  delta);

SIXTRL_FN SIXTRL_STATIC void BB6D_propagate_Sigma_matrix(
        SIXTRL_BE_DATAPTR_DEC BB6D_Sigmas* data,
        SIXTRL_REAL_T S, SIXTRL_REAL_T threshold_singular,
        SIXTRL_UINT64_T handle_singularities,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* Sig_11_hat_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* Sig_33_hat_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* costheta_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* sintheta_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* dS_Sig_11_hat_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* dS_Sig_33_hat_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* dS_costheta_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* dS_sintheta_ptr);

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
NS(BeamBeam6D_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam )
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

#if !defined(mysign)
	#define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

SIXTRL_INLINE void BB6D_boost(
        SIXTRL_BE_DATAPTR_DEC BB6D_boost_data* data,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* x_star, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* px_star,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* y_star, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* py_star,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* sigma_star,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* delta_star){

    SIXTRL_REAL_T sphi = data->sphi;
    SIXTRL_REAL_T cphi = data->cphi;
    SIXTRL_REAL_T tphi = data->tphi;
    SIXTRL_REAL_T salpha = data->salpha;
    SIXTRL_REAL_T calpha = data->calpha;


    SIXTRL_REAL_T x = *x_star;
    SIXTRL_REAL_T px = *px_star;
    SIXTRL_REAL_T y = *y_star;
    SIXTRL_REAL_T py = *py_star ;
    SIXTRL_REAL_T sigma = *sigma_star;
    SIXTRL_REAL_T delta = *delta_star ;

    SIXTRL_REAL_T h = delta + 1. - sqrt((1.+delta)*(1.+delta)-px*px-py*py);


    SIXTRL_REAL_T px_st = px/cphi-h*calpha*tphi/cphi;
    SIXTRL_REAL_T py_st = py/cphi-h*salpha*tphi/cphi;
    SIXTRL_REAL_T delta_st = delta -px*calpha*tphi-py*salpha*tphi+h*tphi*tphi;

    SIXTRL_REAL_T pz_st = sqrt((1.+delta_st)*(1.+delta_st)-px_st*px_st-py_st*py_st);
    SIXTRL_REAL_T hx_st = px_st/pz_st;
    SIXTRL_REAL_T hy_st = py_st/pz_st;
    SIXTRL_REAL_T hsigma_st = 1.-(delta_st+1)/pz_st;

    SIXTRL_REAL_T L11 = 1.+hx_st*calpha*sphi;
    SIXTRL_REAL_T L12 = hx_st*salpha*sphi;
    SIXTRL_REAL_T L13 = calpha*tphi;

    SIXTRL_REAL_T L21 = hy_st*calpha*sphi;
    SIXTRL_REAL_T L22 = 1.+hy_st*salpha*sphi;
    SIXTRL_REAL_T L23 = salpha*tphi;

    SIXTRL_REAL_T L31 = hsigma_st*calpha*sphi;
    SIXTRL_REAL_T L32 = hsigma_st*salpha*sphi;
    SIXTRL_REAL_T L33 = 1./cphi;

    SIXTRL_REAL_T x_st = L11*x + L12*y + L13*sigma;
    SIXTRL_REAL_T y_st = L21*x + L22*y + L23*sigma;
    SIXTRL_REAL_T sigma_st = L31*x + L32*y + L33*sigma;

    *x_star = x_st;
    *px_star = px_st;
    *y_star = y_st;
    *py_star = py_st;
    *sigma_star = sigma_st;
    *delta_star = delta_st;

}

SIXTRL_INLINE void BB6D_inv_boost(
        SIXTRL_BE_DATAPTR_DEC BB6D_boost_data* data,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* x, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* px,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* y, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* py,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* sigma, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T*  delta)
{

    SIXTRL_REAL_T sphi = data->sphi;
    SIXTRL_REAL_T cphi = data->cphi;
    SIXTRL_REAL_T tphi = data->tphi;
    SIXTRL_REAL_T salpha = data->salpha;
    SIXTRL_REAL_T calpha = data->calpha;

    SIXTRL_REAL_T x_st = *x;
    SIXTRL_REAL_T px_st = *px;
    SIXTRL_REAL_T y_st = *y;
    SIXTRL_REAL_T py_st = *py ;
    SIXTRL_REAL_T sigma_st = *sigma;
    SIXTRL_REAL_T delta_st = *delta ;

    SIXTRL_REAL_T pz_st = sqrt((1.+delta_st)*(1.+delta_st)-px_st*px_st-py_st*py_st);
    SIXTRL_REAL_T hx_st = px_st/pz_st;
    SIXTRL_REAL_T hy_st = py_st/pz_st;
    SIXTRL_REAL_T hsigma_st = 1.-(delta_st+1)/pz_st;

    SIXTRL_REAL_T Det_L = 1./cphi + (hx_st*calpha + hy_st*salpha-hsigma_st*sphi)*tphi;

    SIXTRL_REAL_T Linv_11 = (1./cphi + salpha*tphi*(hy_st-hsigma_st*salpha*sphi))/Det_L;
    SIXTRL_REAL_T Linv_12 = (salpha*tphi*(hsigma_st*calpha*sphi-hx_st))/Det_L;
    SIXTRL_REAL_T Linv_13 = -tphi*(calpha - hx_st*salpha*salpha*sphi + hy_st*calpha*salpha*sphi)/Det_L;

    SIXTRL_REAL_T Linv_21 = (calpha*tphi*(-hy_st + hsigma_st*salpha*sphi))/Det_L;
    SIXTRL_REAL_T Linv_22 = (1./cphi + calpha*tphi*(hx_st-hsigma_st*calpha*sphi))/Det_L;
    SIXTRL_REAL_T Linv_23 = -tphi*(salpha - hy_st*calpha*calpha*sphi + hx_st*calpha*salpha*sphi)/Det_L;

    SIXTRL_REAL_T Linv_31 = -hsigma_st*calpha*sphi/Det_L;
    SIXTRL_REAL_T Linv_32 = -hsigma_st*salpha*sphi/Det_L;
    SIXTRL_REAL_T Linv_33 = (1. + hx_st*calpha*sphi + hy_st*salpha*sphi)/Det_L;

    SIXTRL_REAL_T x_i = Linv_11*x_st + Linv_12*y_st + Linv_13*sigma_st;
    SIXTRL_REAL_T y_i = Linv_21*x_st + Linv_22*y_st + Linv_23*sigma_st;
    SIXTRL_REAL_T sigma_i = Linv_31*x_st + Linv_32*y_st + Linv_33*sigma_st;

    SIXTRL_REAL_T h = (delta_st+1.-pz_st)*cphi*cphi;

    SIXTRL_REAL_T px_i = px_st*cphi+h*calpha*tphi;
    SIXTRL_REAL_T py_i = py_st*cphi+h*salpha*tphi;

    SIXTRL_REAL_T delta_i = delta_st + px_i*calpha*tphi + py_i*salpha*tphi - h*tphi*tphi;


    *x = x_i;
    *px = px_i;
    *y = y_i;
    *py = py_i;
    *sigma = sigma_i;
    *delta = delta_i;

}

SIXTRL_INLINE void BB6D_propagate_Sigma_matrix(SIXTRL_BE_DATAPTR_DEC BB6D_Sigmas* data,
        SIXTRL_REAL_T S, SIXTRL_REAL_T threshold_singular, SIXTRL_UINT64_T handle_singularities,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* Sig_11_hat_ptr, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* Sig_33_hat_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* costheta_ptr, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* sintheta_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* dS_Sig_11_hat_ptr, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* dS_Sig_33_hat_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* dS_costheta_ptr, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* dS_sintheta_ptr){

    SIXTRL_REAL_T Sig_11_0 = data->Sig_11_0;
    SIXTRL_REAL_T Sig_12_0 = data->Sig_12_0;
    SIXTRL_REAL_T Sig_13_0 = data->Sig_13_0;
    SIXTRL_REAL_T Sig_14_0 = data->Sig_14_0;
    SIXTRL_REAL_T Sig_22_0 = data->Sig_22_0;
    SIXTRL_REAL_T Sig_23_0 = data->Sig_23_0;
    SIXTRL_REAL_T Sig_24_0 = data->Sig_24_0;
    SIXTRL_REAL_T Sig_33_0 = data->Sig_33_0;
    SIXTRL_REAL_T Sig_34_0 = data->Sig_34_0;
    SIXTRL_REAL_T Sig_44_0 = data->Sig_44_0;

    // Propagate sigma matrix
    SIXTRL_REAL_T Sig_11 = Sig_11_0 + 2.*Sig_12_0*S+Sig_22_0*S*S;
    SIXTRL_REAL_T Sig_33 = Sig_33_0 + 2.*Sig_34_0*S+Sig_44_0*S*S;
    SIXTRL_REAL_T Sig_13 = Sig_13_0 + (Sig_14_0+Sig_23_0)*S+Sig_24_0*S*S;

    SIXTRL_REAL_T Sig_12 = Sig_12_0 + Sig_22_0*S;
    SIXTRL_REAL_T Sig_14 = Sig_14_0 + Sig_24_0*S;
    SIXTRL_REAL_T Sig_22 = Sig_22_0 + 0.*S;
    SIXTRL_REAL_T Sig_23 = Sig_23_0 + Sig_24_0*S;
    SIXTRL_REAL_T Sig_24 = Sig_24_0 + 0.*S;
    SIXTRL_REAL_T Sig_34 = Sig_34_0 + Sig_44_0*S;
    SIXTRL_REAL_T Sig_44 = Sig_44_0 + 0.*S;

    SIXTRL_REAL_T R = Sig_11-Sig_33;
    SIXTRL_REAL_T W = Sig_11+Sig_33;
    SIXTRL_REAL_T T = R*R+4*Sig_13*Sig_13;

    //evaluate derivatives
    SIXTRL_REAL_T dS_R = 2.*(Sig_12_0-Sig_34_0)+2*S*(Sig_22_0-Sig_44_0);
    SIXTRL_REAL_T dS_W = 2.*(Sig_12_0+Sig_34_0)+2*S*(Sig_22_0+Sig_44_0);
    SIXTRL_REAL_T dS_Sig_13 = Sig_14_0 + Sig_23_0 + 2*Sig_24_0*S;
    SIXTRL_REAL_T dS_T = 2*R*dS_R+8.*Sig_13*dS_Sig_13;

    SIXTRL_REAL_T Sig_11_hat, Sig_33_hat, costheta, sintheta, dS_Sig_11_hat,
           dS_Sig_33_hat, dS_costheta, dS_sintheta, cos2theta, dS_cos2theta;

    SIXTRL_REAL_T signR = mysign(R);

    //~ printf("handle: %ld\n",handle_singularities);

    if (T<threshold_singular && handle_singularities){

        SIXTRL_REAL_T a = Sig_12-Sig_34;
        SIXTRL_REAL_T b = Sig_22-Sig_44;
        SIXTRL_REAL_T c = Sig_14+Sig_23;
        SIXTRL_REAL_T d = Sig_24;

        SIXTRL_REAL_T sqrt_a2_c2 = sqrt(a*a+c*c);

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

        SIXTRL_REAL_T sqrtT = sqrt(T);
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


#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRL_COMMON_BE_BEAMBEAM_BE_BEAMBEAM6D_H__ */

/* sixtracklib/common/be_beambeam/be_beambeam6d.h */
