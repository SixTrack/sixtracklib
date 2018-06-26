#ifndef SIXTRLIB_COMMON_IMPL_FADDEEVA_ERROR_FUNCTION_H__
#define SIXTRLIB_COMMON_IMPL_FADDEEVA_ERROR_FUNCTION_H__

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/_impl/definitions.h"

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

SIXTRL_FN int NS(Faddeeva_calculate_w_cern335)( 
    SIXTRL_REAL_T* SIXTRL_RESTRICT re, SIXTRL_REAL_T* SIXTRL_RESTRICT im,
    SIXTRL_REAL_T const in_re, SIXTRL_REAL_T const in_im );

SIXTRL_FN int NS(Faddeeva_calculate_w_mit)( 
    SIXTRL_REAL_T* SIXTRL_RESTRICT re, SIXTRL_REAL_T* SIXTRL_RESTRICT im,
    SIXTRL_REAL_T const in_re, SIXTRL_REAL_T const in_im, 
    SIXTRL_REAL_T rel_error, 
    int const use_continued_fractions );

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRLIB_COMMON_IMPL_FADDEEVA_ERROR_FUNCTION_H__ */

/* end: sixtracklib/common/impl/faddeeva.h */
