#ifndef SIXTRACKLIB_TESTLIB_COMMON_BUFFER_H__
#define SIXTRACKLIB_TESTLIB_COMMON_BUFFER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/internal/buffer_object_defines.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(Object);

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Buffer_object_typeid_to_string)(
    NS(object_type_id_t) const type_id,
    char* SIXTRL_RESTRICT type_str,
    NS(buffer_size_t) const max_type_str_length );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Buffer_object_print_typeid)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    NS(object_type_id_t) const type_id );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Buffer_object_print_out_typeid)(
    NS(object_type_id_t) const type_id );


SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Buffer_object_print)(
    SIXTRL_ARGPTR_DEC FILE* fp,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const struct NS(Object) *const
        SIXTRL_RESTRICT obj );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Buffer_object_print_out)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const struct NS(Object) *const
        SIXTRL_RESTRICT obj );


#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_TESTLIB_COMMON_BUFFER_H__ */

/* end: tests/sixtracklib/testlib/common/buffer.h */
