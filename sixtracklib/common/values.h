#ifndef SIXTRACKLIB_BASELINE_VALUES_H__
#define SIXTRACKLIB_BASELINE_VALUES_H__

#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef union CommonValues
{
    double      f64;
    int64_t     i64;
    uint64_t    u64;
    float       f32[ 2 ];
    int8_t      i8[ 8 ];
    uint8_t     u8[ 8 ];
}
value_t;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* SIXTRACKLIB_BASELINE_VALUES_H__ */

/* end: sixtracklib/baseline/values.h */
