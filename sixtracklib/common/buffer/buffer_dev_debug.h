#ifndef SIXTRACKLIB_COMMON_BUFFER_BUFFER_DEV_DEBUG_H__
#define SIXTRACKLIB_COMMON_BUFFER_BUFFER_DEV_DEBUG_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_is_in_developer_debug_mode)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC void NS(Buffer_enable_developer_debug_mode)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC void NS(Buffer_disable_developer_debug_mode)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *******             Implementation of inline functions            ******* */
/* ************************************************************************* */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE bool NS(Buffer_is_in_developer_debug_mode)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return ( ( buffer != SIXTRL_NULLPTR ) &&
        ( ( buffer->datastore_flags & SIXTRL_BUFFER_DEVELOPMENT_DEBUG_MODE ) ==
            SIXTRL_BUFFER_DEVELOPMENT_DEBUG_MODE ) );
}

SIXTRL_INLINE void NS(Buffer_enable_developer_debug_mode)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    if( buffer != SIXTRL_NULLPTR )
    {
        buffer->datastore_flags |= SIXTRL_BUFFER_DEVELOPMENT_DEBUG_MODE;
    }

    return;
}

SIXTRL_INLINE void NS(Buffer_disable_developer_debug_mode)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    if( buffer != SIXTRL_NULLPTR )
    {
        buffer->datastore_flags &= ~( SIXTRL_BUFFER_DEVELOPMENT_DEBUG_MODE );
    }

    return;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BUFFER_BUFFER_DEV_DEBUG_H__ */