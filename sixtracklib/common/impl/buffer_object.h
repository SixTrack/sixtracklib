#ifndef SIXTRACKLIB_COMMON_IMPL_BUFFER_OBJECT_H__
#define SIXTRACKLIB_COMMON_IMPL_BUFFER_OBJECT_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <limits.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/impl/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

struct NS(Object);
struct NS(Buffer);

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC SIXTRL_DATAPTR_DEC unsigned char const*
NS(Object_get_const_begin_ptr)(
    SIXTRL_ARGPTR_DEC const struct NS(Object) *const SIXTRL_RESTRICT object );

SIXTRL_FN SIXTRL_STATIC SIXTRL_DATAPTR_DEC unsigned char*
NS(Object_get_begin_ptr)(
    SIXTRL_ARGPTR_DEC struct NS(Object)* SIXTRL_RESTRICT object );

SIXTRL_FN SIXTRL_STATIC  void NS(Object_set_begin_ptr)(
    SIXTRL_ARGPTR_DEC struct NS(Object)* SIXTRL_RESTRICT object,
    SIXTRL_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin_ptr );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(Object) const*
NS(Buffer_get_const_objects_begin)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(Object) const*
NS(Buffer_get_const_objects_end)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(Object)*
NS(Buffer_get_objects_begin)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(Object)*
NS(Buffer_get_objects_end)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* *
 * *****         Implementation of inline functions and methods        ***** *
 * ************************************************************************* */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

SIXTRL_INLINE SIXTRL_ARGPTR_DEC SIXTRL_DATAPTR_DEC unsigned char const*
NS(Object_get_const_begin_ptr)(
    SIXTRL_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT object )
{
    typedef SIXTRL_DATAPTR_DEC unsigned char const* ptr_to_raw_t;
    return ( ptr_to_raw_t )( uintptr_t )NS(Object_get_begin_addr)( object );
}

SIXTRL_INLINE SIXTRL_DATAPTR_DEC unsigned char* NS(Object_get_begin_ptr)(
    NS(Object)* SIXTRL_RESTRICT object )
{
    return ( SIXTRL_DATAPTR_DEC unsigned char*
        )NS(Object_get_const_begin_ptr)( object );
}

SIXTRL_INLINE void NS(Object_set_begin_ptr)(
    SIXTRL_ARGPTR_DEC  NS(Object)* SIXTRL_RESTRICT object,
    SIXTRL_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin_ptr )
{
    typedef NS(buffer_addr_t) address_t;
    NS(Object_set_begin_addr)( object, ( address_t )( uintptr_t )begin_ptr );
    return;
}

/* ========================================================================= */

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(Object) const*
NS(Buffer_get_const_objects_begin)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buf )
{
    typedef SIXTRL_ARGPTR_DEC NS(Object) const*     ptr_to_obj_t;
    typedef uintptr_t                               uptr_t;
    return ( ptr_to_obj_t )( uptr_t )NS(Buffer_get_objects_begin_addr)( buf );
}

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(Object) const*
NS(Buffer_get_const_objects_end)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buf )
{
    typedef NS(Object) const* ptr_to_obj_t;
    return ( ptr_to_obj_t )( uintptr_t )NS(Buffer_get_objects_end_addr)( buf );
}

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(Object)* NS(Buffer_get_objects_begin)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef SIXTRL_ARGPTR_DEC NS(Object)* ptr_to_obj_t;
    return ( ptr_to_obj_t )NS(Buffer_get_const_objects_begin)( buffer);
}

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(Object)* NS(Buffer_get_objects_end)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef SIXTRL_ARGPTR_DEC NS(Object)* ptr_to_obj_t;
    return ( ptr_to_obj_t )NS(Buffer_get_const_objects_end)( buffer);
}

/* ========================================================================= */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_BUFFER_OBJECT_H__ */

/* end: sixtracklib/common/impl/buffer_object.h */
