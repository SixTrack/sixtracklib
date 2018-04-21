#ifndef SIXTRACKLIB_COMMON_VALUES_H__
#define SIXTRACKLIB_COMMON_VALUES_H__

#include "sixtracklib/_impl/definitions.h"

#if !defined( _GPUCODE )

#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defiend( _GPUCODE ) */
    
typedef union NS(CommonValues)
{
    SIXTRL_REAL_T   f64;
    SIXTRL_INT64_T  i64;
    SIXTRL_UINT64_T u64;
    SIXTRL_FLOAT_T  f32[ 2 ];
    SIXTRL_INT8_T   i8[ 8 ];
    SIXTRL_UINT8_T  u8[ 8 ];
}
NS(value_t) __attribute__ ((aligned (16)));

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_BASELINE_VALUES_H__ */

/* end: sixtracklib/common/values.h */
