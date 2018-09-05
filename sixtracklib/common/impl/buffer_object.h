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

SIXTRL_FN SIXTRL_STATIC  NS(Object)* NS(Object_preset)(
    struct NS(Object)* SIXTRL_RESTRICT object );

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_t) NS(Object_get_begin_addr)(
    const struct NS(Object) *const SIXTRL_RESTRICT object );

SIXTRL_FN SIXTRL_STATIC void NS(Object_set_begin_addr)(
    struct NS(Object)* SIXTRL_RESTRICT object,
    NS(buffer_addr_t) const begin_addr );

SIXTRL_FN SIXTRL_STATIC  NS(object_type_id_t) NS(Object_get_type_id)(
    const struct NS(Object) *const SIXTRL_RESTRICT object );

SIXTRL_FN SIXTRL_STATIC  void NS(Object_set_type_id)(
    struct NS(Object)* SIXTRL_RESTRICT object,
    NS(object_type_id_t) const type_id );

SIXTRL_FN SIXTRL_STATIC  NS(buffer_size_t) NS(Object_get_size)(
    const struct NS(Object) *const SIXTRL_RESTRICT object );

SIXTRL_FN SIXTRL_STATIC  void NS(Object_set_size)(
    struct NS(Object)* SIXTRL_RESTRICT object, NS(buffer_size_t) const size );

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

SIXTRL_INLINE NS(Object)* NS(Object_preset)(
    NS(Object)* SIXTRL_RESTRICT object )
{
    if( object != SIXTRL_NULLPTR )
    {
        NS(Object_set_begin_addr)( object, 0u );
        NS(Object_set_type_id)( object, 0u );
        NS(Object_set_size)( object, 0u );
    }

    return object;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_addr_t) NS(Object_get_begin_addr)(
    const NS(Object) *const SIXTRL_RESTRICT object )
{
    #if !defined( NDEBUG )
    typedef unsigned char const*    ptr_to_raw_t;
    #endif /* !defined( NDEBUG ) */

    typedef NS(buffer_addr_t)       address_t;

    SIXTRL_ASSERT(
        (   sizeof( ptr_to_raw_t ) >= sizeof( address_t ) ) ||
        ( ( sizeof( ptr_to_raw_t ) == 4u ) &&
          ( sizeof( address_t    ) == 8u ) &&
          ( ( ( object != SIXTRL_NULLPTR ) &&
              ( ( ( address_t )NS(ManagedBuffer_get_limit_offset_max)() >
                object->begin_addr ) ) ) ||
            (   object == SIXTRL_NULLPTR ) ) ) );

    return ( object != SIXTRL_NULLPTR ) ? object->begin_addr : ( address_t )0u;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(Object_set_begin_addr)(
    NS(Object)* SIXTRL_RESTRICT object, NS(buffer_addr_t) const begin_addr )
{
    #if !defined( NDEBUG )
    typedef unsigned char const*    ptr_to_raw_t;
    typedef NS(buffer_addr_t)       address_t;

    SIXTRL_ASSERT(
        (   sizeof( ptr_to_raw_t ) >= sizeof( address_t ) ) ||
        ( ( sizeof( ptr_to_raw_t ) == 4u ) &&
          ( sizeof( address_t    ) == 8u ) &&
          ( ( address_t )NS(ManagedBuffer_get_limit_offset_max)() >
              begin_addr ) ) );
    #endif /* !defined( NDEBUG ) */

    if( object != SIXTRL_NULLPTR ) object->begin_addr = begin_addr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(object_type_id_t) NS(Object_get_type_id)(
    const NS(Object) *const SIXTRL_RESTRICT object )
{
    return ( object != SIXTRL_NULLPTR )
        ? object->type_id : ( NS(object_type_id_t ) )0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(Object_set_type_id)(
    NS(Object)* SIXTRL_RESTRICT object, NS(object_type_id_t) const type_id )
{
    if( object != SIXTRL_NULLPTR ) object->type_id = type_id;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(Object_get_size)(
    const NS(Object) *const SIXTRL_RESTRICT object )
{
    return ( object != SIXTRL_NULLPTR )
        ? object->size : ( NS(buffer_size_t) )0u;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(Object_set_size)(
    NS(Object)* SIXTRL_RESTRICT object, NS(buffer_size_t) const size )
{
    if( object != SIXTRL_NULLPTR )
    {
        object->size = size;
    }

    return;
}

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
