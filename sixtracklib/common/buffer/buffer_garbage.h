#ifndef SIXTRACKLIB_COMMON_BUFFER_BUFFER_GARBAGE_RANGE_H__
#define SIXTRACKLIB_COMMON_BUFFER_BUFFER_GARBAGE_RANGE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <limits.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/buffer_garbage_defines.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

typedef struct NS(BufferGarbage)
{
    NS(buffer_addr_t)    begin_addr       SIXTRL_ALIGN( 8u );
    NS(buffer_addr_t)    size             SIXTRL_ALIGN( 8u );
}
NS(BufferGarbage);

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_GARBAGE_ARGPTR_DEC NS(BufferGarbage)*
NS(BufferGarbage_preset)(
    SIXTRL_BUFFER_GARBAGE_ARGPTR_DEC NS(BufferGarbage)*
        SIXTRL_RESTRICT garbage_range );

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_t) NS(BufferGarbage_get_begin_addr)(
    SIXTRL_BUFFER_GARBAGE_ARGPTR_DEC const NS(BufferGarbage)
        *const SIXTRL_RESTRICT garbage_range );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(BufferGarbage_get_size)(
    SIXTRL_BUFFER_GARBAGE_ARGPTR_DEC const NS(BufferGarbage)
        *const SIXTRL_RESTRICT garbage_range );

SIXTRL_FN SIXTRL_STATIC void NS(BufferGarbage_set_begin_addr)(
    SIXTRL_BUFFER_GARBAGE_ARGPTR_DEC NS(BufferGarbage)*
        SIXTRL_RESTRICT garbage_range,
    NS(buffer_addr_t) const begin_addr );

SIXTRL_FN SIXTRL_STATIC void NS(BufferGarbage_set_size)(
    SIXTRL_BUFFER_GARBAGE_ARGPTR_DEC NS(BufferGarbage)*
        SIXTRL_RESTRICT garbage_range,
    NS(buffer_size_t) const range_size );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* *
 * *****         Implementation of inline functions and methods        ***** *
 * ************************************************************************* */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */


SIXTRL_INLINE SIXTRL_BUFFER_GARBAGE_ARGPTR_DEC NS(BufferGarbage)*
NS(BufferGarbage_preset)( SIXTRL_BUFFER_GARBAGE_ARGPTR_DEC NS(BufferGarbage)*
    SIXTRL_RESTRICT garbage_range )
{
    NS(BufferGarbage_set_begin_addr)( garbage_range, 0 );
    NS(BufferGarbage_set_size)( garbage_range, 0u );

    return garbage_range;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_addr_t) NS(BufferGarbage_get_begin_addr)(
    SIXTRL_BUFFER_GARBAGE_ARGPTR_DEC const NS(BufferGarbage)
        *const SIXTRL_RESTRICT garbage_range )
{
    return ( garbage_range != SIXTRL_NULLPTR )
        ? garbage_range->begin_addr : ( NS(buffer_addr_t) )0u;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(BufferGarbage_get_size)(
    SIXTRL_BUFFER_GARBAGE_ARGPTR_DEC const NS(BufferGarbage)
        *const SIXTRL_RESTRICT garbage_range )
{
    return ( garbage_range != SIXTRL_NULLPTR )
        ? garbage_range->size : ( NS(buffer_size_t) )0u;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(BufferGarbage_set_begin_addr)(
    SIXTRL_BUFFER_GARBAGE_ARGPTR_DEC NS(BufferGarbage)*
        SIXTRL_RESTRICT garbage_range,
    NS(buffer_addr_t) const begin_addr )
{
    if( garbage_range != SIXTRL_NULLPTR )
    {
        garbage_range->begin_addr = begin_addr;
    }

    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(BufferGarbage_set_size)(
    SIXTRL_BUFFER_GARBAGE_ARGPTR_DEC NS(BufferGarbage)*
        SIXTRL_RESTRICT garbage_range,
    NS(buffer_size_t) const range_size )
{
    if( garbage_range != SIXTRL_NULLPTR )
    {
        garbage_range->size = range_size;
    }

    return;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BUFFER_BUFFER_GARBAGE_RANGE_H__ */

/* end: sixtracklib/common/buffer/buffer_garbage.h */
