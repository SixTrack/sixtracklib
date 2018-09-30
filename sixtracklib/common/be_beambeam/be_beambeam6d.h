#ifndef SIXTRL_COMMON_BE_BEAMBEAM_BE_BEAMBEAM6D_H__
#define SIXTRL_COMMON_BE_BEAMBEAM_BE_BEAMBEAM6D_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_defines.h"
    #include "sixtracklib/common/impl/beam_elements_defines.h"
    #include "sixtracklib/common/impl/buffer_type.h"
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




#endif /* SIXTRL_COMMON_BE_BEAMBEAM_BE_BEAMBEAM6D_H__ */

/* sixtracklib/common/be_beambeam/be_beambeam6d.h */
