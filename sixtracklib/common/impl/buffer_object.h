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
    #include "sixtracklib/common/impl/buffer_defines.h"
    #include "sixtracklib/common/impl/buffer_object_defines.h"
    #include "sixtracklib/common/impl/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

typedef struct NS(Object)
{
    NS(buffer_addr_t)    begin_addr       SIXTRL_ALIGN( 8u );
    NS(object_type_id_t) type_id          SIXTRL_ALIGN( 8u );
    NS(buffer_size_t)    size             SIXTRL_ALIGN( 8u );
}
NS(Object);

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
NS(Object_preset)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT ob );

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_t) NS(Object_get_begin_addr)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT ob );

SIXTRL_FN SIXTRL_STATIC void NS(Object_set_begin_addr)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT object,
    NS(buffer_addr_t) const begin_addr );

SIXTRL_FN SIXTRL_STATIC  NS(object_type_id_t) NS(Object_get_type_id)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT ob );

SIXTRL_FN SIXTRL_STATIC  void NS(Object_set_type_id)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT object,
    NS(object_type_id_t) const type_id );

SIXTRL_FN SIXTRL_STATIC  NS(buffer_size_t) NS(Object_get_size)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT ob );

SIXTRL_FN SIXTRL_STATIC  void NS(Object_set_size)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT ob,
    NS(buffer_size_t) const size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_OBJ_DATAPTR_DEC unsigned char const*
NS(Object_get_const_begin_ptr)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT ob );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_OBJ_DATAPTR_DEC unsigned char*
NS(Object_get_begin_ptr)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT ob );

SIXTRL_FN SIXTRL_STATIC void NS(Object_set_begin_ptr)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT ob,
    SIXTRL_BUFFER_OBJ_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin_ptr );

/* ========================================================================= */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* *
 * *****         Implementation of inline functions and methods        ***** *
 * ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/impl/managed_buffer_minimal.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

SIXTRL_INLINE SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* NS(Object_preset)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT object )
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
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT object )
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
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT object,
    NS(buffer_addr_t) const begin_addr )
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
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT object )
{
    return ( object != SIXTRL_NULLPTR )
        ? object->type_id : ( NS(object_type_id_t ) )0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(Object_set_type_id)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT object,
    NS(object_type_id_t) const type_id )
{
    if( object != SIXTRL_NULLPTR ) object->type_id = type_id;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(Object_get_size)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT object )
{
    return ( object != SIXTRL_NULLPTR )
        ? object->size : ( NS(buffer_size_t) )0u;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(Object_set_size)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT object,
    NS(buffer_size_t) const size )
{
    if( object != SIXTRL_NULLPTR )
    {
        object->size = size;
    }

    return;
}

SIXTRL_INLINE SIXTRL_BUFFER_OBJ_DATAPTR_DEC unsigned char const*
NS(Object_get_const_begin_ptr)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT object )
{
    typedef SIXTRL_BUFFER_OBJ_DATAPTR_DEC unsigned char const* ptr_to_raw_t;
    return ( ptr_to_raw_t )( uintptr_t )NS(Object_get_begin_addr)( object );
}

SIXTRL_INLINE SIXTRL_BUFFER_OBJ_DATAPTR_DEC unsigned char*
NS(Object_get_begin_ptr)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT object )
{
    return ( SIXTRL_BUFFER_OBJ_DATAPTR_DEC unsigned char*
        )NS(Object_get_const_begin_ptr)( object );
}

SIXTRL_INLINE void NS(Object_set_begin_ptr)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT object,
    SIXTRL_BUFFER_OBJ_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin_ptr )
{
    typedef NS(buffer_addr_t) address_t;
    NS(Object_set_begin_addr)( object, ( address_t )( uintptr_t )begin_ptr );
    return;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_BUFFER_OBJECT_H__ */

/* end: sixtracklib/common/impl/buffer_object.h */
