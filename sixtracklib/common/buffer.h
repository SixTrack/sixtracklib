#ifndef SIXTRACKLIB_COMMON_BUFFER_H__
#define SIXTRACKLIB_COMMON_BUFFER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <limits.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/_impl/modules.h"
    #include "sixtracklib/common/impl/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ------------------------------------------------------------------------- */

struct NS(Object);

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

/* ------------------------------------------------------------------------- */

struct NS(BufferGarbage);

SIXTRL_FN SIXTRL_STATIC struct NS(BufferGarbage)* NS(BufferGarbage_preset)(
    struct NS(BufferGarbage)* SIXTRL_RESTRICT garbage_range );

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_t) NS(BufferGarbage_get_begin_addr)(
    const struct NS(BufferGarbage) *const SIXTRL_RESTRICT garbage_range );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(BufferGarbage_get_size)(
    const struct NS(BufferGarbage) *const SIXTRL_RESTRICT garbage_range );

SIXTRL_FN SIXTRL_STATIC void NS(BufferGarbage_set_begin_addr)(
    struct NS(BufferGarbage)* SIXTRL_RESTRICT garbage_range,
    NS(buffer_addr_t) const begin_addr );

SIXTRL_FN SIXTRL_STATIC void NS(BufferGarbage_set_size)(
    struct NS(BufferGarbage)* SIXTRL_RESTRICT garbage_range,
    NS(buffer_size_t) const range_size );

/* ------------------------------------------------------------------------- */

struct NS(Buffer);

SIXTRL_FN SIXTRL_STATIC  NS(Buffer)* NS(Buffer_preset)(
   NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_diff_t)
    NS(Buffer_get_limit_offset_max)( void );

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_diff_t)
    NS(Buffer_get_limit_offset_min)( void );

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_check_addr_arithmetic)(
    NS(buffer_addr_t) const addr, NS(buffer_addr_diff_t) const offset,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_t) NS(Buffer_perform_addr_shift)(
    NS(buffer_addr_t) const addr, NS(buffer_addr_diff_t) const offset,
    NS(buffer_size_t) const slot_size );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Buffer_get_slot_size)(
   const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Buffer_get_slot_based_length)(
    NS(buffer_size_t) const in_length, NS(buffer_size_t) const slot_size );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC NS(buffer_flags_t) NS(Buffer_get_flags)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_owns_datastore)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_uses_datastore)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_has_datastore)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_allow_modify_datastore_contents)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_allow_clear)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_allow_append_objects)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_allow_delete_objects)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_allow_remapping)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_allow_resize)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_uses_mempool_datastore)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_uses_special_opencl_datastore)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_uses_special_cuda_datastore)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_flags_t)
NS(Buffer_get_datastore_special_flags)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC void NS(Buffer_set_datastore_special_flags)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_flags_t) const flags );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_t) NS(Buffer_get_datastore_begin_addr)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Buffer_get_size)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Buffer_get_capacity)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Buffer_get_header_size)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_t) NS(Buffer_get_data_begin_addr)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_t) NS(Buffer_get_data_end_addr)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_t) NS(Buffer_get_objects_begin_addr)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_t) NS(Buffer_get_objects_end_addr)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Buffer_get_section_header_size)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Buffer_get_slots_extent)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Buffer_get_num_of_slots)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Buffer_get_objects_extent)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC  NS(buffer_size_t) NS(Buffer_get_num_of_objects)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Buffer_get_dataptrs_extent)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC  NS(buffer_size_t) NS(Buffer_get_num_of_dataptrs)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Buffer_get_garbage_extent)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC  NS(buffer_size_t)
NS(Buffer_get_num_of_garbage_ranges)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_reserve)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const new_max_num_slots,
    NS(buffer_size_t) const new_max_num_objects,
    NS(buffer_size_t) const new_max_num_dataptrs,
    NS(buffer_size_t) const new_max_num_garbage_elems );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_remap)(
    NS(Buffer)* SIXTRL_RESTRICT buffer );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if !defined( _GPUCODE )

SIXTRL_HOST_FN NS(Buffer)* NS(Buffer_new)(
    NS(buffer_size_t) const buffer_capacity );

SIXTRL_HOST_FN NS(Buffer)* NS(Buffer_new_detailed)(
    NS(buffer_size_t)  const initial_max_num_objects,
    NS(buffer_size_t)  const initial_max_num_slots,
    NS(buffer_size_t)  const initial_max_num_dataptrs,
    NS(buffer_size_t)  const initial_max_num_garbage_elements,
    NS(buffer_flags_t) const buffer_flags );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC void NS(Buffer_free)(
    NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_clear)(
    NS(Buffer)* SIXTRL_RESTRICT buffer );

#if !defined( _GPUCODE )

SIXTRL_HOST_FN SIXTRL_STATIC void NS(Buffer_delete)(
    NS(Buffer)* SIXTRL_RESTRICT buffer );

#endif /* !defined( _GPUCODE ) */


#endif /* !defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *******             Implementation of inline functions            ******* */
/* ************************************************************************* */

/* For plain C/Cxx */

#if !defined( SIXTRL_NO_INCLUDES)
    #include "sixtracklib/common/impl/buffer_type.h"
    #include "sixtracklib/common/impl/buffer_generic.h"

    #if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
               ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )
        #include "sixtracklib/opencl/buffer.h"
    #endif /* OpenCL */

    #if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
               ( SIXTRACKLIB_ENABLE_MODULE_CUDA == 1 )
        #include "sixtracklib/cuda/buffer.h"
    #endif /* Cuda */

#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

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
              ( ( ( address_t )NS(Buffer_get_limit_offset_max)() >
                object->begin_addr ) ) ) ||
            (   object == SIXTRL_NULLPTR ) ) ) );

    return ( object != SIXTRL_NULLPTR ) ? object->begin_addr : ( address_t )0u;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(Object_set_begin_addr)(
    NS(Object)* SIXTRL_RESTRICT object, NS(buffer_addr_t) const begin_addr )
{
    typedef unsigned char const*    ptr_to_raw_t;
    typedef NS(buffer_addr_t)       address_t;

    SIXTRL_ASSERT(
        (   sizeof( ptr_to_raw_t ) >= sizeof( address_t ) ) ||
        ( ( sizeof( ptr_to_raw_t ) == 4u ) &&
          ( sizeof( address_t    ) == 8u ) &&
          ( ( address_t )NS(Buffer_get_limit_offset_max)() >
              begin_addr ) ) );

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

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(BufferGarbage)* NS(BufferGarbage_preset)(
    struct NS(BufferGarbage)* SIXTRL_RESTRICT garbage_range )
{
    NS(BufferGarbage_set_begin_addr)( garbage_range, 0 );
    NS(BufferGarbage_set_size)( garbage_range, 0u );

    return garbage_range;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_addr_t) NS(BufferGarbage_get_begin_addr)(
    const struct NS(BufferGarbage) *const SIXTRL_RESTRICT garbage_range )
{
    return ( garbage_range != SIXTRL_NULLPTR )
        ? garbage_range->begin_addr : ( NS(buffer_addr_t) )0u;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(BufferGarbage_get_size)(
    const struct NS(BufferGarbage) *const SIXTRL_RESTRICT garbage_range )
{
    return ( garbage_range != SIXTRL_NULLPTR )
        ? garbage_range->size : ( NS(buffer_size_t) )0u;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(BufferGarbage_set_begin_addr)(
    struct NS(BufferGarbage)* SIXTRL_RESTRICT garbage_range,
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
    struct NS(BufferGarbage)* SIXTRL_RESTRICT garbage_range,
    NS(buffer_size_t) const range_size )
{
    if( garbage_range != SIXTRL_NULLPTR )
    {
        garbage_range->size = range_size;
    }

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(Buffer)* NS(Buffer_preset)( NS(Buffer)* SIXTRL_RESTRICT buf )
{
    if( buf != SIXTRL_NULLPTR )
    {
        buf->data_addr        = ( NS(buffer_addr_t)  )0u;
        buf->data_size        = ( NS(buffer_size_t)  )0u;
        buf->header_size      = NS(BUFFER_DEFAULT_HEADER_SIZE);
        buf->data_capacity    = ( NS(buffer_size_t)  )0u;

        buf->slot_length      = NS(BUFFER_DEFAULT_SLOT_SIZE);

        buf->object_addr      = ( NS(buffer_addr_t)  )0u;
        buf->num_objects      = ( NS(buffer_size_t)  )0u;

        buf->datastore_flags  = ( NS(buffer_flags_t) )0;
        buf->datastore_addr   = ( NS(buffer_addr_t)  )0u;
    }

    return buf;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_addr_diff_t) NS(Buffer_get_limit_offset_max)()
{
    #if defined( _GPUCODE )
         #if defined( __OPENCL_VERSION__ ) &&
             defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) &&
             ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )

            SIXTRL_ASSERT( sizeof( NS(buffer_addr_diff_t) >=
                           sizeof( ptr_to_raw_t ) );

            SIXTRL_STATIC_VAR addr_diff_t const LIMIT_OFFSET_MAX = LONG_MAX;

        #elif defined( __CUDACC__ ) &&
              defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) &&
              ( SIXTRACKLIB_ENABLE_MODULE_CUDA == 1 )

            SIXTRL_ASSERT( sizeof( addr_diff_t ) >= sizeof( long long int ) );
            SIXTRL_STATIC_VAR addr_diff_t const LIMIT_OFFSET_MAX = NPP_MAX_64S;

        #endif /* defined( __OPENCL_VERSION__ ) */
    #elif defined( __cplusplus )
        using addr_diff_t = NS(buffer_addr_diff_t);

        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::digits >= 63u );
        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::is_signed  );
        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::is_integer );

        SIXTRL_STATIC_VAR addr_diff_t const LIMIT_OFFSET_MAX =
             std::numeric_limits< addr_diff_t >::max();

    #else
        SIXTRL_ASSERT( sizeof( NS(buffer_addr_diff_t) ) >=
                       sizeof( long long int ) );

        SIXTRL_STATIC_VAR NS(buffer_addr_diff_t) const
            LIMIT_OFFSET_MAX = ( NS(buffer_addr_diff_t) )LLONG_MAX;

    #endif /* defined( _GPUCODE ) */

    return LIMIT_OFFSET_MAX;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_addr_diff_t) NS(Buffer_get_limit_offset_min)( void )
{
    #if defined( _GPUCODE )
         #if defined( __OPENCL_VERSION__ ) &&
             defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) &&
             ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )

            SIXTRL_ASSERT( sizeof( NS(buffer_addr_diff_t) >=
                           sizeof( ptr_to_raw_t ) );

            SIXTRL_STATIC_VAR addr_diff_t const LIMIT_OFFSET_MIN = LONG_MIN;

        #elif defined( __CUDACC__ ) &&
              defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) &&
              ( SIXTRACKLIB_ENABLE_MODULE_CUDA == 1 )

            SIXTRL_ASSERT( sizeof( addr_diff_t ) >= sizeof( long long int ) );
            SIXTRL_STATIC_VAR addr_diff_t const LIMIT_OFFSET_MIN = NPP_MIN_64S;

        #endif /* defined( __OPENCL_VERSION__ ) */
    #elif defined( __cplusplus )
        using addr_diff_t = NS(buffer_addr_diff_t);

        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::digits >= 63u );
        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::is_signed  );
        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::is_integer );

        SIXTRL_STATIC_VAR addr_diff_t const LIMIT_OFFSET_MIN =
             std::numeric_limits< addr_diff_t >::min();

    #else
        SIXTRL_ASSERT( sizeof( NS(buffer_addr_diff_t) ) >=
                       sizeof( long long int ) );

        SIXTRL_STATIC_VAR NS(buffer_addr_diff_t) const
            LIMIT_OFFSET_MIN = ( NS(buffer_addr_diff_t) )LLONG_MIN;

    #endif /* defined( _GPUCODE ) */

    return LIMIT_OFFSET_MIN;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE bool NS(Buffer_check_addr_arithmetic)(
    NS(buffer_addr_t) const addr,
    NS(buffer_addr_diff_t) const offset,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_addr_t)       address_t;
    typedef NS(buffer_addr_diff_t)  addr_diff_t;

    SIXTRL_ASSERT( sizeof( address_t ) == sizeof( addr_diff_t) );
    SIXTRL_ASSERT( sizeof( address_t ) >= 8u );
    SIXTRL_ASSERT( slot_size > 0u );

    addr_diff_t const LIMIT_OFFSET_MAX = NS(Buffer_get_limit_offset_max)();

    return (
        ( addr != ( address_t )0u ) && ( ( addr % slot_size ) == 0u ) &&
        ( ( ( offset >= 0 ) &&
            ( ( ( ( address_t )offset ) % slot_size ) == 0u ) &&
            ( ( LIMIT_OFFSET_MAX - ( address_t )offset ) >= addr ) ) ||
          ( ( offset <  0 ) &&
            ( ( ( ( address_t )-offset ) % slot_size ) == 0u ) &&
            ( addr >= ( address_t )-offset ) ) ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_addr_t) NS(Buffer_perform_addr_shift)(
    NS(buffer_addr_t) const addr,
    NS(buffer_addr_diff_t) const offset,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_ASSERT( NS(Buffer_check_addr_arithmetic)(
        addr, offset, slot_size) );

    #if !defined( NDEBUG )
    ( void )slot_size;
    #endif /* !defined( NDEBUG ) */

    return addr + offset;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_slot_size)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return ( buffer != SIXTRL_NULLPTR )
        ? buffer->slot_length : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_slot_based_length)(
    NS(buffer_size_t) const in_length, NS(buffer_size_t) const slot_size )
{
    NS(buffer_size_t) consolidated_length = in_length;

    SIXTRL_STATIC_VAR NS(buffer_size_t) const ZERO = ( NS(buffer_size_t) )0u;

    if( ( slot_size != ZERO ) && ( ( in_length % slot_size ) != 0u ) )
    {
        consolidated_length /= slot_size;
        consolidated_length += 1u;
        consolidated_length *= slot_size;
    }

    return consolidated_length;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN bool NS(Buffer_check_for_flags)(
    const NS(Buffer)  *const SIXTRL_RESTRICT buffer,
    NS(buffer_flags_t) const flags_to_check )
{
    return ( ( buffer != SIXTRL_NULLPTR ) &&
        ( ( buffer->datastore_flags & flags_to_check ) == flags_to_check ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_flags_t) NS(Buffer_get_flags)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return ( buffer != SIXTRL_NULLPTR)
        ? buffer->datastore_flags : NS(BUFFER_FLAGS_NONE);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE bool NS(Buffer_owns_datastore)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_check_for_flags)( buffer, NS(BUFFER_OWNS_DATASTORE) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE bool NS(Buffer_uses_datastore)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return ( ( NS(Buffer_has_datastore)( buffer ) ) &&
        ( NS(Buffer_check_for_flags)(
            buffer, NS(BUFFER_USES_DATASTORE) ) ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE bool NS(Buffer_has_datastore)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return ( ( buffer != SIXTRL_NULLPTR ) &&
             ( buffer->datastore_addr != ( NS(buffer_addr_t) )0u ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE bool NS(Buffer_allow_modify_datastore_contents)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return (
        ( NS(Buffer_allow_clear)( buffer ) ) &&
        ( ( NS(Buffer_allow_append_objects)( buffer ) ) ||
          ( NS(Buffer_allow_delete_objects)( buffer ) ) ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE bool NS(Buffer_allow_clear)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_check_for_flags)( buffer,
        NS(BUFFER_DATASTORE_ALLOW_CLEAR) | NS(BUFFER_USES_DATASTORE) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE bool NS(Buffer_allow_append_objects)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_check_for_flags)( buffer,
        NS(BUFFER_DATASTORE_ALLOW_APPENDS) | NS(BUFFER_USES_DATASTORE) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE bool NS(Buffer_allow_delete_objects)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_check_for_flags)( buffer,
        NS(BUFFER_DATASTORE_ALLOW_DELETES)  | NS(BUFFER_USES_DATASTORE) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE bool NS(Buffer_allow_remapping)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_check_for_flags)( buffer,
        NS(BUFFER_DATASTORE_ALLOW_REMAPPING) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE bool NS(Buffer_allow_resize)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_check_for_flags)( buffer,
        NS(BUFFER_DATASTORE_ALLOW_RESIZE) |
        NS(BUFFER_USES_DATASTORE) | NS(BUFFER_OWNS_DATASTORE) );
}

SIXTRL_INLINE bool NS(Buffer_uses_mempool_datastore)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_check_for_flags)( buffer,
        NS(BUFFER_USES_DATASTORE) |
        NS(BUFFER_DATASTORE_MEMPOOL) );
}

SIXTRL_INLINE bool NS(Buffer_uses_special_opencl_datastore)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_check_for_flags)( buffer,
        NS(BUFFER_USES_DATASTORE) |
        NS(BUFFER_DATASTORE_OPENCL) );
}

SIXTRL_INLINE bool NS(Buffer_uses_special_cuda_datastore)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_check_for_flags)( buffer,
        NS(BUFFER_USES_DATASTORE) |
        NS(BUFFER_DATASTORE_CUDA) );
}

SIXTRL_INLINE NS(buffer_flags_t)
NS(Buffer_get_datastore_special_flags)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    typedef NS(buffer_flags_t) flags_t;

    flags_t   flags = buffer->datastore_flags;
    flags &=  NS(BUFFER_DATASTORE_SPECIAL_FLAGS_MASK);
    flags >>= NS(BUFFER_DATASTORE_SPECIAL_FLAGS_BITS);

    return flags;
}

SIXTRL_INLINE void NS(Buffer_set_datastore_special_flags)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_flags_t) const flags )
{
    typedef NS(buffer_flags_t) flags_t;

    flags_t const temp = ( flags & (
        NS(BUFFER_DATASTORE_SPECIAL_FLAGS_MASK) >>
        NS(BUFFER_DATASTORE_SPECIAL_FLAGS_BITS) ) ) <<
            NS(BUFFER_DATASTORE_SPECIAL_FLAGS_BITS);

    buffer->datastore_flags &= ~( NS(BUFFER_DATASTORE_SPECIAL_FLAGS_MASK) );
    buffer->datastore_flags |= temp;

    return;
}

SIXTRL_INLINE NS(buffer_addr_t) NS(Buffer_get_datastore_begin_addr)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    typedef NS(buffer_addr_t) address_t;

    return ( NS(Buffer_has_datastore)( buffer ) )
         ? ( buffer->datastore_addr )
         : ( address_t )0u;
}

/* -------------------------------------------------------------------------- */

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_size)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return ( buffer != SIXTRL_NULLPTR )
        ? ( buffer->data_size ) : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_capacity)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return ( buffer != SIXTRL_NULLPTR )
        ? ( buffer->data_capacity ) : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_header_size)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return ( buffer != SIXTRL_NULLPTR )
        ? ( buffer->header_size ) : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_addr_t) NS(Buffer_get_data_begin_addr)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    NS(buffer_addr_t) const begin_addr = ( buffer != SIXTRL_NULLPTR )
        ? ( buffer->data_addr ) : ( NS(buffer_addr_t ) )0u;

    SIXTRL_ASSERT(
        ( NS(Buffer_get_slot_size)( buffer ) > ( NS(buffer_size_t) )0u ) &&
        ( ( begin_addr % NS(Buffer_get_slot_size)( buffer ) ) == 0u ) );

    return begin_addr;
}

SIXTRL_INLINE NS(buffer_addr_t) NS(Buffer_get_data_end_addr)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    NS(buffer_addr_t) end_addr = NS(Buffer_get_data_begin_addr)( buffer );

    if( end_addr != ( NS(buffer_size_t) )0u )
    {
        end_addr += NS(Buffer_get_size)( buffer );
    }

    SIXTRL_ASSERT(
        ( NS(Buffer_get_slot_size)( buffer ) > ( NS(buffer_size_t) )0u ) &&
        ( ( end_addr % NS(Buffer_get_slot_size)( buffer ) ) == 0u ) );

    return end_addr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */


SIXTRL_INLINE NS(buffer_addr_t) NS(Buffer_get_objects_begin_addr)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    NS(buffer_addr_t) const begin_addr = ( buffer != SIXTRL_NULLPTR )
        ? ( buffer->object_addr ) : ( NS(buffer_addr_t ) )0u;

    SIXTRL_ASSERT(
        ( NS(Buffer_get_slot_size)( buffer ) > ( NS(buffer_size_t) )0u ) &&
        ( ( begin_addr % NS(Buffer_get_slot_size)( buffer ) ) == 0u ) );

    return begin_addr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_addr_t) NS(Buffer_get_objects_end_addr)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) address_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    address_t end_addr = NS(Buffer_get_objects_begin_addr)( buffer );

    if( end_addr != ZERO_SIZE )
    {
        buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );
        buf_size_t const obj_size  = NS(Buffer_get_slot_based_length)(
            sizeof( NS(Object) ), slot_size );

        SIXTRL_ASSERT( buffer    != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( slot_size != ZERO_SIZE );
        SIXTRL_ASSERT( obj_size  != ZERO_SIZE );

        end_addr += obj_size * buffer->num_objects;

        SIXTRL_ASSERT( ( end_addr % slot_size ) == 0u );
        SIXTRL_ASSERT( end_addr < NS(Buffer_get_data_end_addr)( buffer ) );
    }

    return end_addr;
}

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Buffer_get_section_extent)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id )
{
    typedef NS(buffer_size_t) buf_size_t;
    buf_size_t extent = ( buf_size_t )0u;

    if( NS(Buffer_uses_datastore)( buffer ) )
    {
        #if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
            SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1
        if( NS(Buffer_uses_special_opencl_datastore)( buffer ) )
        {
            extent = NS(Buffer_get_section_extent_opencl)(
                buffer, header_slot_id);
        }
        else
        #endif /* SIXTRACKLIB_ENABLE_MODULE_OPENCL */

        #if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
            SIXTRACKLIB_ENABLE_MODULE_CUDA == 1
        if( NS(Buffer_uses_special_cuda_datastore)( buffer ) )
        {
            extent = NS(Buffer_get_section_extent_cuda)(
                buffer, header_slot_id );
        }
        else
        #endif /* SIXTRACKLIB_ENABLE_MODULE_CUDA */

        {
            extent = NS(Buffer_get_section_extent_generic)(
                buffer, header_slot_id );
        }
    }

    return extent;
}

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Buffer_get_section_num_elements)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id )
{
    typedef NS(buffer_size_t) buf_size_t;
    buf_size_t num_elements = ( buf_size_t )0u;

    if( NS(Buffer_uses_datastore)( buffer ) )
    {
        #if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
            SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1
        if( NS(Buffer_uses_special_opencl_datastore)( buffer ) )
        {
            num_elements = NS(Buffer_get_section_num_elements_opencl)(
                buffer, header_slot_id);
        }
        else
        #endif /* SIXTRACKLIB_ENABLE_MODULE_OPENCL */

        #if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
            SIXTRACKLIB_ENABLE_MODULE_CUDA == 1
        if( NS(Buffer_uses_special_cuda_datastore)( buffer ) )
        {
            num_elements = NS(Buffer_get_section_num_elements_cuda)(
                buffer, header_slot_id );
        }
        else
        #endif /* SIXTRACKLIB_ENABLE_MODULE_CUDA */

        {
            num_elements = NS(Buffer_get_section_num_entities_generic)(
                buffer, header_slot_id );
        }
    }

    return num_elements;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_section_header_size)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) address_t;

    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    return 2u * NS(Buffer_get_slot_based_length)(
        sizeof( address_t ), slot_size );
}

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_slots_extent)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    SIXTRL_STATIC_VAR NS(buffer_size_t) const
        SLOTS_HEADER_SLOT_ID = ( NS(buffer_size_t) )3u;

    return NS(Buffer_get_section_extent)( buffer, SLOTS_HEADER_SLOT_ID );
}

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_num_of_slots)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    SIXTRL_STATIC_VAR NS(buffer_size_t) const
        SLOTS_HEADER_SLOT_ID = ( NS(buffer_size_t) )3u;

    return NS(Buffer_get_section_num_elements)(
        buffer, SLOTS_HEADER_SLOT_ID );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_objects_extent)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    SIXTRL_STATIC_VAR NS(buffer_size_t) const
        OBJECTS_HEADER_SLOT_ID = ( NS(buffer_size_t) )4u;

    return NS(Buffer_get_section_extent)( buffer, OBJECTS_HEADER_SLOT_ID );
}

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_num_of_objects)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    SIXTRL_STATIC_VAR NS(buffer_size_t) const
        OBJ_HEADER_SLOT_ID = ( NS(buffer_size_t) )4u;

    NS(buffer_size_t) const num_objects = ( buffer != SIXTRL_NULLPTR )
        ? ( buffer->num_objects ) : ( NS(buffer_size_t) )0u;

    NS(buffer_size_t) const cmp_num_objects =
        NS(Buffer_get_section_num_elements)( buffer, OBJ_HEADER_SLOT_ID );

    SIXTRL_ASSERT( num_objects == cmp_num_objects );

    return num_objects;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_dataptrs_extent)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    SIXTRL_STATIC_VAR NS(buffer_size_t) const
        DATAPTRS_HEADER_SLOT_ID = ( NS(buffer_size_t) )5u;

    return NS(Buffer_get_section_extent)( buffer, DATAPTRS_HEADER_SLOT_ID );
}

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_num_of_dataptrs)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    SIXTRL_STATIC_VAR NS(buffer_size_t) const
        DATAPTRS_HEADER_SLOT_ID = ( NS(buffer_size_t) )5u;

    return NS(Buffer_get_section_num_elements)(
        buffer, DATAPTRS_HEADER_SLOT_ID );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_garbage_extent)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    SIXTRL_STATIC_VAR NS(buffer_size_t) const
        GARBAGE_HEADER_SLOT_ID = ( NS(buffer_size_t) )6u;

    return NS(Buffer_get_section_extent)( buffer, GARBAGE_HEADER_SLOT_ID );
}

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_num_of_garbage_ranges)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    SIXTRL_STATIC_VAR NS(buffer_size_t) const
        GARBAGE_HEADER_SLOT_ID = ( NS(buffer_size_t) )6u;

    return NS(Buffer_get_section_num_elements)(
        buffer, GARBAGE_HEADER_SLOT_ID );
}

/* ========================================================================= */

SIXTRL_INLINE int NS(Buffer_reserve)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_elems )
{
    int success = -1;

    if( NS(Buffer_uses_datastore)( buffer ) )
    {
        #if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
             SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1

        if( NS(Buffer_uses_special_opencl_datastore)( buffer ) )
        {
            success = NS(Buffer_reserve_opencl)( buffer, max_num_objects,
                max_num_slots, max_num_dataptrs, max_num_garbage_elems );
        }
        else
        #endif /* SIXTRACKLIB_ENABLE_MODULE_OPENCL */

        #if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
             SIXTRACKLIB_ENABLE_MODULE_CUDA == 1

        if( NS(Buffer_uses_special_cuda_datastore)( buffer ) )
        {
            success = NS(Buffer_reserve_cuda)( buffer, max_num_objects,
                max_num_slots, max_num_dataptrs, max_num_garbage_elems );
        }
        else
        #endif /* SIXTRACKLIB_ENABLE_MODULE_CUDA */

        {
            success = NS(Buffer_reserve_generic)( buffer, max_num_objects,
                max_num_slots, max_num_dataptrs, max_num_garbage_elems );
        }
    }

    return success;
}

SIXTRL_INLINE int NS(Buffer_remap)( NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    int success = -1;

    if( ( NS(Buffer_has_datastore)( buffer ) ) &&
        ( NS(Buffer_allow_remapping)( buffer ) ) )
    {
        #if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
             SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1

        if( NS(Buffer_uses_special_opencl_datastore)( buffer ) )
        {
            success = NS(Buffer_remap_opencl)( buffer );
        }
        else
        #endif /* SIXTRACKLIB_ENABLE_MODULE_OPENCL */

        #if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
             SIXTRACKLIB_ENABLE_MODULE_CUDA == 1

        if( NS(Buffer_uses_special_cuda_datastore)( buffer ) )
        {
            success = NS(Buffer_remap_cuda)( buffer );
        }
        else
        #endif /* SIXTRACKLIB_ENABLE_MODULE_CUDA */

        {
            success = NS(Buffer_remap_generic)( buffer );
        }
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE int NS(Buffer_clear)( NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    int success = -1;

    if( ( NS(Buffer_has_datastore)( buffer ) ) &&
        ( NS(Buffer_allow_clear)(   buffer ) ) )
    {
        #if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
             SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1

        if( NS(Buffer_uses_special_opencl_datastore)( buffer ) )
        {
            success = NS(Buffer_clear_opencl)( buffer );
        }
        else
        #endif /* SIXTRACKLIB_ENABLE_MODULE_OPENCL */

        #if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
            SIXTRACKLIB_ENABLE_MODULE_CUDA == 1

        if( NS(Buffer_uses_special_cuda_datastore)( buffer ) )
        {
            success = NS(Buffer_clear_cuda)( buffer );
        }
        else
        #endif /* SIXTRACKLIB_ENABLE_MODULE_CUDA */

        {
            success = NS(Buffer_clear_generic)( buffer );
        }
    }

    return success;
}

SIXTRL_INLINE void NS(Buffer_free)( NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    if( NS(Buffer_owns_datastore)( buffer ) )
    {
        #if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
             SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1

        if( NS(Buffer_uses_special_opencl_datastore)( buffer ) )
        {
            NS(Buffer_free_opencl)( buffer );
        }
        else
        #endif /* SIXTRACKLIB_ENABLE_MODULE_OPENCL */

        #if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
             SIXTRACKLIB_ENABLE_MODULE_CUDA == 1

        if( NS(Buffer_uses_special_cuda_datastore)( buffer ) )
        {
            NS(Buffer_free_cuda)( buffer );
        }
        else
        #endif /* SIXTRACKLIB_ENABLE_MODULE_CUDA */

        {
            NS(Buffer_free_generic)( buffer );
        }
    }

    return;
}

#if !defined( _GPUCODE )

SIXTRL_INLINE void NS(Buffer_delete)( NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    NS(Buffer_free)( buffer );
    free( buffer );
    buffer = SIXTRL_NULLPTR;

    return;
}

#endif /* !defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BUFFER_H__ */

/* end: sixtracklib/common/buffer.h */
