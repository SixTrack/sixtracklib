#ifndef SIXTRACKLIB_TESTLIB_COMMON_BUFFER_BINARY_ARRAY_BUFFER_C99_H__
#define SIXTRACKLIB_TESTLIB_COMMON_BUFFER_BINARY_ARRAY_BUFFER_C99_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/internal/binary_array_buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <limits.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_STATIC SIXTRL_FN void NS(BinaryArray_print_out)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj );

#if !defined( GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(BinaryArray_print)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj );

#endif /* !defined( GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */


#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#if !defined( _GPUCODE )

void NS(BinaryArray_print_out)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj )
{
    NS(BinaryArray_print)( stdout, obj );
}

#else

void NS(BinaryArray_print_out)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj )
{
    NS(BinaryArray_print)( stdout, obj );
}

#endif /* !defined _GPUCODE */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_TESTLIB_COMMON_BUFFER_BINARY_ARRAY_BUFFER_C99_H__ */
/* end: tests/sixtracklib/testlib/common/buffer/binary_array_buffer.h */
