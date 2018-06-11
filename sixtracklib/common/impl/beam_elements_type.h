#ifndef SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENTS_TYPE_H__
#define SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENTS_TYPE_H__

#if !defined( _GPUCODE )

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/blocks.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */
    
typedef struct NS(Drift)
{
    SIXTRL_REAL_T length __attribute__(( aligned( 8 ) ));
}
NS(Drift);   

/* ------------------------------------------------------------------------- */

typedef struct NS(DriftExact)
{
    SIXTRL_REAL_T length __attribute__(( aligned( 8 ) ));
}
NS(DriftExact);

/* ------------------------------------------------------------------------- */

typedef struct NS(MultiPole)
{
    SIXTRL_REAL_T   length  __attribute__(( aligned( 8 ) ));
    SIXTRL_REAL_T   hxl     __attribute__(( aligned( 8 ) ));
    SIXTRL_REAL_T   hyl     __attribute__(( aligned( 8 ) ));
    SIXTRL_INT64_T  order   __attribute__(( aligned( 8 ) ));
    
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
        SIXTRL_RESTRICT bal __attribute__(( aligned( 8 ) ));
}
NS(MultiPole);

/* ------------------------------------------------------------------------- */
    
#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */
    

#endif /* SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENTS_TYPE_H__ */

/* end: sixtracklib/common/impl/beam_elements_type.h */
