#ifndef SIXTRACKLIB_COMMON_ALIGNMENT_H__
#define SIXTRACKLIB_COMMON_ALIGNMENT_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/alignment_impl.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_HOST_FN SIXTRL_UINT64_T NS(Alignment_calculate_commonN)(
    SIXTRL_UINT64_T const* SIXTRL_RESTRICT numbers,
    SIXTRL_UINT64_T const num_of_operands );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_ALIGNMENT_H__ */

/* end: sixtracklib/common/alignment.h */
