#ifndef SIXTRACKLIB_COMMON_IMPL_FADDEEVA_MIT_H__
#define SIXTRACKLIB_COMMON_IMPL_FADDEEVA_MIT_H__

#if !defined( SIXTRL_NO_INCLUDE )
    #include "sixtracklib/_impl/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDE ) */

#if !defined( _GPUCODE ) && defined ( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined ( __cplusplus ) */

SIXTRL_HOST_FN SIXTRL_STATIC int NS(Faddeeva_calculate_w_mit)(
    SIXTRL_REAL_T* SIXTRL_RESTRICT out_re,
    SIXTRL_REAL_T* SIXTRL_RESTRICT out_im,
    SIXTRL_REAL_T const re, SIXTRL_REAL_T const im,
    SIXTRL_REAL_T rel_error,
    int const use_continued_fractions );

#if !defined( _GPUCODE ) && defined ( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined ( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_FADDEEVA_MIT_H__ */

/* end: sixtracklib/common/impl/faddeeva_mit.h */
