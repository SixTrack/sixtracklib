#ifndef SIXTRACKLIB_COMMON_ALIGNMENT_H__
#define SIXTRACKLIB_COMMON_ALIGNMENT_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/impl/alignment_impl.h"

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

#endif /* !defined( _GPUCODE ) */
    
SIXTRL_UINT64_T NS(Alignment_calculate_commonN)(
    SIXTRL_UINT64_T const* SIXTRL_RESTRICT numbers, 
    SIXTRL_UINT64_T const num_of_operands );

#if !defined( _GPUCODE )

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */
    
#endif /* !defined( _GPUCODE ) */
    
#endif /* SIXTRACKLIB_COMMON_ALIGNMENT_H__ */

/* end: sixtracklib/common/alignment.h */
