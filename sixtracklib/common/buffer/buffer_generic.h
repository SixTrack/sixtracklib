#ifndef SIXTRKL_COMMON_BUFFER_BUFFER_GENERIC_H__
#define SIXTRKL_COMMON_BUFFER_BUFFER_GENERIC_H__

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
    #include "sixtracklib/common/buffer/buffer_object.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(Buffer_get_const_data_begin)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(Buffer_get_const_data_end)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC  SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(Buffer_get_data_begin)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC  SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(Buffer_get_data_end)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_clear_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    bool const set_data_to_zero );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_reset_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_reset_detailed_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const initial_max_num_objects,
    NS(buffer_size_t) const initial_max_num_slots,
    NS(buffer_size_t) const initial_max_num_dataptrs,
    NS(buffer_size_t) const initial_max_num_garbage_elems );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_needs_remapping_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_remap_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_refresh_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_reserve_capacity_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const new_buffer_capacity );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_reserve_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_ranges );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC void NS(Buffer_free_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
NS(Buffer_add_object_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const void *const SIXTRL_RESTRICT object_handle,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const  object_size,
    NS(object_type_id_t)                const  type_id,
    NS(buffer_size_t)                   const  num_obj_dataptrs,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT offsets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT sizes,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT counts );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_init_from_data)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const max_data_buffer_length );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_init_on_flat_memory)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const buffer_capacity );

SIXTRL_FN SIXTRL_STATIC  int NS(Buffer_init_on_flat_memory_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const buffer_capacity,
    NS(buffer_size_t) const initial_max_num_objects,
    NS(buffer_size_t) const initial_max_num_slots,
    NS(buffer_size_t) const initial_max_num_dataptrs,
    NS(buffer_size_t) const initial_max_num_garbage_elems );

/* ========================================================================= */

#if !defined( _GPUCODE )

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(Object) const*
NS(Buffer_get_const_object)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const object_index );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(Object)*
NS(Buffer_get_object)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const object_index );


SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(Object) const*
NS(Buffer_get_const_objects_begin)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(Object) const*
NS(Buffer_get_const_objects_end)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(Object)*
NS(Buffer_get_objects_begin)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(Object)*
NS(Buffer_get_objects_end)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

#endif /* !defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* *
 * *****         Implementation of inline functions and methods        ***** *
 * ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer/mem_pool.h"
    #include "sixtracklib/common/buffer/managed_buffer.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(Buffer_get_const_data_begin)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buf )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* ptr_to_raw_t;
    return ( ptr_to_raw_t )( uintptr_t )NS(Buffer_get_data_begin_addr)( buf );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(Buffer_get_const_data_end)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buf )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* ptr_to_raw_t;
    return ( ptr_to_raw_t )( uintptr_t )NS(Buffer_get_data_end_addr)( buf );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(Buffer_get_data_begin)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_to_raw_t;
    return ( ptr_to_raw_t )NS(Buffer_get_const_data_begin)( buffer );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC unsigned char* NS(Buffer_get_data_end)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_to_raw_t;
    return ( ptr_to_raw_t )NS(Buffer_get_const_data_end)( buffer );
}

/* ========================================================================= */

SIXTRL_INLINE int NS(Buffer_clear_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    bool const set_data_to_zero )
{
    int success = -1;

    typedef NS(buffer_size_t)                buf_size_t;
    typedef unsigned char                    raw_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC raw_t* ptr_to_raw_t;

    if( ( NS(Buffer_has_datastore)( buffer ) ) &&
        ( NS(Buffer_allow_clear)( buffer ) ) )
    {
        buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

        ptr_to_raw_t begin = ( ptr_to_raw_t )( uintptr_t
            )NS(Buffer_get_data_begin_addr)( buffer );

        NS(ManagedBuffer_clear)( begin, set_data_to_zero, slot_size );
        buffer->num_objects = ( buf_size_t )0u;
        success = 0;
    }

    return success;
}

/* ========================================================================= */

SIXTRL_INLINE int NS(Buffer_reset_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_reset_detailed_generic)( buffer, 0u, 0u, 0u, 0u );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Buffer_reset_detailed_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_ranges )
{
    int success = -1;

    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) address_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_to_raw_t;

    ptr_to_raw_t begin = ( ptr_to_raw_t )( uintptr_t
        )NS(Buffer_get_data_begin_addr)( buffer );

    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    SIXTRL_ASSERT( slot_size > 0u );
    SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size ) == 0u );
    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)( begin, slot_size ) );

    if( ( NS(Buffer_uses_datastore)( buffer ) ) &&
        ( NS(Buffer_allow_modify_datastore_contents)( buffer ) ) &&
        ( NS(Buffer_allow_clear)( buffer ) ) )
    {
        buf_size_t const capacity = NS(Buffer_get_capacity)( buffer );
        buf_size_t buffer_size    = ( buf_size_t )0u;

        success = NS(ManagedBuffer_init)( begin, &buffer_size, max_num_objects,
            max_num_slots, max_num_dataptrs, max_num_garbage_ranges,
                capacity, slot_size );

        if( success == 0 )
        {
            SIXTRL_ASSERT( begin != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( buffer_size <= capacity );

            SIXTRL_ASSERT( buffer_size >
                NS(ManagedBuffer_get_section_header_length)(
                    begin, slot_size ) );

            buffer->data_addr     = ( address_t )( uintptr_t )begin;
            buffer->data_size     = buffer_size;
            buffer->data_capacity = capacity;

            buffer->num_objects   = ( buf_size_t )0u;
            buffer->object_addr   = ( address_t )( uintptr_t
                )NS(ManagedBuffer_get_const_objects_index_begin)(
                    begin, slot_size );
        }
    }

    return success;
}

/* ========================================================================= */

SIXTRL_INLINE bool NS(Buffer_needs_remapping_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
//     SIXTRL_ASSERT( NS(Buffer_has_datastore)( buffer ) );
    SIXTRL_ASSERT( NS(Buffer_allow_remapping)( buffer ) );

    return NS(ManagedBuffer_needs_remapping)(
            NS(Buffer_get_const_data_begin)( buffer ),
            NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE int NS(Buffer_remap_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    int success = -1;

    typedef NS(buffer_size_t )        buf_size_t;
    typedef NS(buffer_addr_t )        address_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_to_raw_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC address_t*     ptr_to_addr_t;

    if( NS(Buffer_allow_remapping)( buffer ) )
    {
        ptr_to_raw_t begin = ( ptr_to_raw_t )( uintptr_t
            )NS(Buffer_get_data_begin_addr)( buffer );

        ptr_to_addr_t ptr_header   = ( ptr_to_addr_t )begin;
        buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

        SIXTRL_ASSERT( ptr_header != SIXTRL_NULLPTR );

        if( ( ptr_header[ 0 ] != ( address_t )0 ) &&
            ( ptr_header[ 0 ] == ( address_t )( uintptr_t )ptr_header ) )
        {
            success = 0;
        }
        else if( ( ptr_header[ 0 ] != ( address_t )0 ) &&
                 ( slot_size > 0u ) && ( begin != SIXTRL_NULLPTR ) )
        {
            SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size ) == 0u );
            success = NS(ManagedBuffer_remap)( begin, slot_size );

            if( success == 0 )
            {
                SIXTRL_STATIC_VAR buf_size_t const OBJS_ID = ( buf_size_t )4u;

                SIXTRL_ASSERT( ptr_header[ 0 ] ==
                    ( address_t )( uintptr_t )ptr_header );

                buffer->data_addr   = ptr_header[ 0 ];
                buffer->data_size   = ptr_header[ 1 ];
                buffer->header_size = ptr_header[ 2 ];

                buffer->object_addr = ( address_t )( uintptr_t
                    )NS(ManagedBuffer_get_ptr_to_section_data)(
                        begin, OBJS_ID, slot_size );

                buffer->num_objects =
                    NS(ManagedBuffer_get_section_num_entities)(
                        begin, OBJS_ID, slot_size );
            }
        }
    }

    return success;
}

SIXTRL_INLINE int NS(Buffer_refresh_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_to_raw_t;

    int success = -1;
    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE  = ( buf_size_t )0u;

    buf_size_t const slot_s = NS(Buffer_get_slot_size)( buffer );

    ptr_to_raw_t ptr = ( ptr_to_raw_t )( uintptr_t
        )NS(Buffer_get_data_begin_addr)( buffer );

    if( ( buffer != SIXTRL_NULLPTR ) && ( ptr != SIXTRL_NULLPTR ) &&
        ( slot_s > ZERO_SIZE ) &&
        ( ZERO_SIZE == ( ( ( uintptr_t )ptr ) % slot_s ) ) )
    {
        success = 0;

        if( NS(ManagedBuffer_needs_remapping)( ptr, slot_s ) )
        {
            if( NS(Buffer_allow_remapping)( buffer ) )
            {
                success = NS(ManagedBuffer_remap)( ptr, slot_s );
            }
            else
            {
                success = -1;
            }
        }

        if( success == 0 )
        {
            SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID = ( buf_size_t )4u;

            SIXTRL_ASSERT( ( uintptr_t )ptr ==
                           ( uintptr_t )buffer->data_addr );
            SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)( ptr, slot_s ) );

            buffer->data_size =
                NS(ManagedBuffer_get_buffer_length)( ptr, slot_s );

            if( buffer->data_capacity < buffer->data_size )
            {
                buffer->data_capacity = buffer->data_size;
            }

            buffer->object_addr = ( NS(buffer_addr_t) )( uintptr_t
                )NS(ManagedBuffer_get_const_objects_index_begin)(
                    ptr, slot_s );

            buffer->num_objects = NS(ManagedBuffer_get_section_num_entities)(
                ptr, OBJECTS_ID, slot_s );
        }
    }

    return success;
}

/* ========================================================================= */

SIXTRL_INLINE int NS(Buffer_reserve_capacity_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const new_buffer_capacity )
{
    int success = -1;

    typedef NS(buffer_addr_t)                address_t;
    typedef NS(buffer_size_t)                buf_size_t;

    typedef unsigned char                    raw_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC raw_t* ptr_to_raw_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    if( ( buffer != SIXTRL_NULLPTR ) && ( slot_size != ZERO_SIZE ) &&
        ( NS(Buffer_owns_datastore)( buffer ) ) &&
        ( NS(Buffer_allow_resize)( buffer ) ) )
    {
        if( NS(Buffer_get_capacity)( buffer ) < new_buffer_capacity )
        {
            address_t const datastore_addr =
                NS(Buffer_get_datastore_begin_addr)( buffer );

            if( NS(Buffer_uses_mempool_datastore)( buffer ) )
            {
                #if !defined( _GPUCODE )

                typedef NS(MemPool)     mem_pool_t;
                typedef mem_pool_t*     ptr_to_mem_pool_t;
                typedef NS(AllocResult) alloc_result_t;

                ptr_to_mem_pool_t mem_pool = ( ptr_to_mem_pool_t )(
                    uintptr_t )datastore_addr;

                ptr_to_raw_t old_begin = ( ptr_to_raw_t )( uintptr_t
                   )NS(MemPool_get_begin_pos)( mem_pool );

                address_t const old_base_addr =
                    ( address_t )( uintptr_t )old_begin;

                buf_size_t bytes_missing = new_buffer_capacity;

                buf_size_t const current_capacity =
                    NS(MemPool_get_size)( mem_pool );

                bytes_missing -= ( bytes_missing >= current_capacity )
                    ? current_capacity : bytes_missing;

                SIXTRL_ASSERT( current_capacity ==
                    NS(Buffer_get_capacity)( buffer ) );

                SIXTRL_ASSERT( ( old_base_addr % slot_size ) == ZERO_SIZE );
                SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
                    old_begin, slot_size ) );

                if( bytes_missing > ZERO_SIZE )
                {
                    bool const changed_base_addr = NS(MemPool_reserve_aligned)(
                        mem_pool, new_buffer_capacity, slot_size );

                    if( NS(MemPool_get_remaining_bytes)( mem_pool ) >=
                        bytes_missing )
                    {
                        ptr_to_raw_t new_begin = ( ptr_to_raw_t )( uintptr_t
                                )NS(MemPool_get_begin_pos)( mem_pool );

                        address_t const new_begin_addr = ( address_t )(
                            uintptr_t )new_begin;

                        SIXTRL_ASSERT( new_begin_addr != ( address_t )0 );

                        if( ( new_begin_addr == old_base_addr ) &&
                            ( !changed_base_addr ) )
                        {
                            success = 0;
                        }
                        else
                        {
                            success = NS(ManagedBuffer_remap)(
                                new_begin, slot_size );
                        }

                        SIXTRL_ASSERT( ( !NS(ManagedBuffer_needs_remapping)(
                                new_begin, slot_size ) ) || ( success != 0 ) );

                        if( success == 0 )
                        {
                            alloc_result_t result = NS(MemPool_append_aligned)(
                                mem_pool, bytes_missing, slot_size );

                            SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
                                new_begin, slot_size ) );

                            if( NS(AllocResult_valid)( &result ) )
                            {
                                SIXTRL_STATIC_VAR raw_t Z = ( raw_t )0;

                                SIXTRL_STATIC_VAR buf_size_t const
                                    OBJECTS_ID = ( buf_size_t )4u;

                                buffer->object_addr = ( address_t )( uintptr_t
                                    )NS(ManagedBuffer_get_const_ptr_to_section)(
                                        new_begin, OBJECTS_ID, slot_size );

                                buffer->data_capacity =
                                    NS(MemPool_get_size)( mem_pool );

                                buffer->data_addr = new_begin_addr;

                                SIXTRACKLIB_SET_VALUES( raw_t,
                                    NS(AllocResult_get_pointer)( &result ),
                                    NS(AllocResult_get_length)( &result ), Z );
                            }
                        }
                    }
                }
                else
                {
                    success = 0;
                }

                #else  /* !defined( _GPUCODE ) */

                if( new_buffer_capacity <= NS(Buffer_get_capacity)( buffer ) )
                {
                    success = 0;
                }

                #endif /* !defined( _GPUCODE ) */
            }
        }
    }

    return success;
}

SIXTRL_INLINE int NS(Buffer_reserve_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_ranges )
{
    typedef NS(buffer_size_t)         buf_size_t;
    typedef NS(buffer_addr_t)         address_t;
    typedef unsigned char             raw_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC raw_t* ptr_to_raw_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const SLOTS_ID     = ( buf_size_t )3u;
    SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID   = ( buf_size_t )4u;
    SIXTRL_STATIC_VAR buf_size_t const DATAPTRS_ID  = ( buf_size_t )5u;
    SIXTRL_STATIC_VAR buf_size_t const GARBAGE_ID   = ( buf_size_t )6u;
    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE    = ( buf_size_t )0u;

    ptr_to_raw_t begin = ( ptr_to_raw_t )( uintptr_t
        )NS(Buffer_get_data_begin_addr)( buffer );

    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    SIXTRL_ASSERT(!NS(ManagedBuffer_needs_remapping)( begin, slot_size ) );

    if( ( NS(Buffer_uses_datastore)( buffer ) ) &&
        ( NS(Buffer_allow_modify_datastore_contents)( buffer ) ) &&
        ( ( max_num_objects != NS(ManagedBuffer_get_section_max_num_entities)(
                begin, OBJECTS_ID, slot_size ) ) ||
          ( max_num_slots != NS(ManagedBuffer_get_section_max_num_entities)(
                begin, SLOTS_ID, slot_size ) ) ||
          ( max_num_dataptrs != NS(ManagedBuffer_get_section_max_num_entities)(
                begin, DATAPTRS_ID, slot_size ) ) ||
          ( max_num_garbage_ranges !=
            NS(ManagedBuffer_get_section_max_num_entities)(
                begin, GARBAGE_ID, slot_size ) ) ) )
    {
        buf_size_t current_capacity = NS(Buffer_get_capacity)( buffer );

        buf_size_t const requ_buffer_capacity =
            NS(ManagedBuffer_calculate_buffer_length)( begin,
                max_num_objects, max_num_slots, max_num_dataptrs,
                    max_num_garbage_ranges, slot_size );

        bool needs_resizing = ( current_capacity <  requ_buffer_capacity );

        success = ( !needs_resizing ) ? 0 : -1;

        if( needs_resizing )
        {
            success = NS(Buffer_reserve_capacity_generic)(
                buffer, requ_buffer_capacity);

            if( success == 0 )
            {
                current_capacity = NS(Buffer_get_capacity)( buffer );
                needs_resizing   = ( current_capacity  < requ_buffer_capacity );
            }
        }

        if( !needs_resizing )
        {
            buf_size_t cur_buffer_size = ZERO_SIZE;
            ptr_to_raw_t new_begin = ( ptr_to_raw_t )( uintptr_t
                )NS(Buffer_get_data_begin_addr)( buffer );

            SIXTRL_ASSERT( success == 0 );

            success = NS(ManagedBuffer_reserve)( new_begin, &cur_buffer_size,
                max_num_objects, max_num_slots, max_num_dataptrs,
                    max_num_garbage_ranges, current_capacity, slot_size );

            if( success == 0 )
            {
                SIXTRL_STATIC_VAR buf_size_t const OBJS_ID = ( buf_size_t )4u;

                SIXTRL_ASSERT( buffer->data_capacity >= cur_buffer_size );

                buffer->data_size = cur_buffer_size;

                buffer->object_addr = ( address_t )( uintptr_t
                    )NS(ManagedBuffer_get_const_objects_index_begin)(
                        new_begin, slot_size );

                buffer->num_objects =
                    NS(ManagedBuffer_get_section_num_entities)(
                        new_begin, OBJS_ID, slot_size );
            }
        }
    }

    return success;
}

/* ========================================================================= */

SIXTRL_INLINE void NS(Buffer_free_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    if( ( NS(Buffer_has_datastore)(  buffer ) ) &&
        ( NS(Buffer_owns_datastore)( buffer ) ) )
    {
        typedef NS(buffer_addr_t)   address_t;
        typedef NS(buffer_flags_t)  flags_t;

        address_t const datastore_addr =
            NS(Buffer_get_datastore_begin_addr)( buffer );

        flags_t buffer_flags =
                NS(Buffer_get_datastore_special_flags)( buffer );

        if( NS(Buffer_uses_mempool_datastore)( buffer ) )
        {
            #if !defined( _GPUCODE )
            typedef NS(MemPool)                   mem_pool_t;
            typedef SIXTRL_ARGPTR_DEC mem_pool_t* ptr_to_mem_pool_t;

            ptr_to_mem_pool_t ptr_mem_pool =
                ( ptr_to_mem_pool_t )( uintptr_t )datastore_addr;

            NS(MemPool_free)( ptr_mem_pool );
            free( ptr_mem_pool );
            #endif /* !defined( _GPUCODE ) */

            buffer_flags = ( flags_t )0u;
        }

        NS(Buffer_set_datastore_special_flags)( buffer, buffer_flags );
        NS(Buffer_preset)( buffer );
    }

    return;
}

/* ========================================================================= */

SIXTRL_INLINE SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
NS(Buffer_add_object_generic)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const void *const SIXTRL_RESTRICT ptr_to_object,
    NS(buffer_size_t)    const object_size,
    NS(object_type_id_t) const type_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const num_obj_dataptr,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT dataptr_offsets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT dataptr_sizes,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT dataptr_counts )
{
    typedef NS(Object)                              object_t;
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef NS(buffer_addr_t)                       address_t;
    typedef unsigned char                           raw_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC raw_t*        ptr_to_raw_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC address_t*    ptr_to_addr_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC object_t*  ptr_to_object_t;

    #if defined( _GPUCODE )
    typedef SIXTRL_ARGPTR_DEC unsigned char const*  cpy_raw_ptr_t;
    #endif /* defined( _GPUCODE ) */

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR address_t  const ZERO_ADDR = ( address_t  )0u;

    SIXTRL_BUFFER_OBJ_ARGPTR_DEC object_t* result_object = SIXTRL_NULLPTR;
    int success = -1;

    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    if( ( slot_size     != ZERO_SIZE      ) &&
        ( object_size   >  ZERO_SIZE      ) &&
        ( ptr_to_object != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_allow_append_objects)( buffer ) ) )
    {
        SIXTRL_STATIC_VAR buf_size_t const SLOTS_ID     = ( buf_size_t )3u;
        SIXTRL_STATIC_VAR buf_size_t const OBJS_ID      = ( buf_size_t )4u;
        SIXTRL_STATIC_VAR buf_size_t const DATAPTRS_ID  = ( buf_size_t )5u;
        SIXTRL_STATIC_VAR buf_size_t const GARBAGE_ID   = ( buf_size_t )6u;

        ptr_to_raw_t begin = ( ptr_to_raw_t )( uintptr_t
            )NS(Buffer_get_data_begin_addr)( buffer );

        buf_size_t const cur_num_slots =
            NS(ManagedBuffer_get_section_num_entities)(
                begin, SLOTS_ID, slot_size );

        buf_size_t max_num_slots =
            NS(ManagedBuffer_get_section_max_num_entities)(
                begin, SLOTS_ID, slot_size );

        buf_size_t const cur_num_objects =
            NS(ManagedBuffer_get_section_num_entities)(
                begin, OBJS_ID, slot_size );

        buf_size_t max_num_objects =
            NS(ManagedBuffer_get_section_max_num_entities)(
                begin, OBJS_ID, slot_size );

        buf_size_t const cur_num_dataptrs =
            NS(ManagedBuffer_get_section_num_entities)(
                begin, DATAPTRS_ID, slot_size );

        buf_size_t max_num_dataptrs =
            NS(ManagedBuffer_get_section_max_num_entities)(
                begin, DATAPTRS_ID, slot_size );

        buf_size_t const max_num_garbage_ranges =
            NS(ManagedBuffer_get_section_max_num_entities)(
                begin, GARBAGE_ID, slot_size );

        buf_size_t const requ_num_objects  = cur_num_objects  + 1u;
        buf_size_t const requ_num_dataptrs = cur_num_dataptrs + num_obj_dataptr;

        buf_size_t const obj_handle_size =
            NS(ManagedBuffer_get_slot_based_length)( object_size, slot_size );

        buf_size_t const additional_num_slots =
            NS(ManagedBuffer_predict_required_num_slots)( begin, object_size,
                num_obj_dataptr, dataptr_sizes, dataptr_counts, slot_size );

        buf_size_t const additional_num_slots_size =
            additional_num_slots * slot_size;

        buf_size_t const requ_num_slots = cur_num_slots + additional_num_slots;

        if( ( requ_num_objects  > max_num_objects  ) ||
            ( requ_num_slots    > max_num_slots    ) ||
            ( requ_num_dataptrs > max_num_dataptrs ) )
        {
            success = NS(Buffer_reserve_generic)( buffer,
                requ_num_objects, requ_num_slots,
                    requ_num_dataptrs, max_num_garbage_ranges );

            begin = ( ptr_to_raw_t )( uintptr_t
                )NS(Buffer_get_data_begin_addr)( buffer );
        }
        else
        {
            success = 0;
        }

        SIXTRL_ASSERT( ( success != 0 ) || ( requ_num_objects <=
                        NS(ManagedBuffer_get_section_max_num_entities)(
                            begin, OBJS_ID, slot_size ) ) );

        SIXTRL_ASSERT( ( success != 0 ) || ( requ_num_slots <=
                NS(ManagedBuffer_get_section_max_num_entities)(
                    begin, SLOTS_ID, slot_size ) ) );

        SIXTRL_ASSERT( ( success != 0 ) || ( requ_num_dataptrs <=
                NS(ManagedBuffer_get_section_max_num_entities)(
                    begin, DATAPTRS_ID, slot_size ) ) );

        SIXTRL_ASSERT( ( success != 0 ) || ( max_num_garbage_ranges <=
                NS(ManagedBuffer_get_section_max_num_entities)(
                    begin, GARBAGE_ID, slot_size ) ) );

        /* ----------------------------------------------------------------- */

        if( success == 0 )
        {

            buf_size_t const current_slots_size =
                NS(ManagedBuffer_get_section_size)(
                    begin, SLOTS_ID, slot_size );

            ptr_to_raw_t dest_slots = (
                ( ptr_to_raw_t )NS(ManagedBuffer_get_ptr_to_section)(
                    begin, SLOTS_ID, slot_size ) ) + current_slots_size;

            ptr_to_raw_t stored_obj_begin = dest_slots;

            if( object_size > ZERO_SIZE )
            {
                ptr_to_raw_t dest_obj_info = ( ( ptr_to_raw_t
                    )NS(ManagedBuffer_get_ptr_to_section)(
                        begin, OBJS_ID, slot_size ) );

                object_t obj_info;
                NS(Object_preset_priv)( &obj_info );

                if( stored_obj_begin != SIXTRL_NULLPTR )
                {
                    SIXTRL_ASSERT( ( ( ( uintptr_t )stored_obj_begin ) %
                        slot_size ) == 0 );

                    #if !defined( _GPUCODE )

                    SIXTRACKLIB_COPY_VALUES( unsigned char, stored_obj_begin,
                        ptr_to_object, object_size );

                    #else

                    SIXTRACKLIB_COPY_VALUES( unsigned char, stored_obj_begin,
                        ( cpy_raw_ptr_t )( uintptr_t )ptr_to_object,
                            object_size );

                    #endif /* !defined( _GPUCODE ) */
                }
                else
                {
                    success = -1;
                }

                NS(Object_set_type_id_priv)( &obj_info, type_id );
                NS(Object_set_begin_ptr_priv)( &obj_info, stored_obj_begin );
                NS(Object_set_size_priv)( &obj_info, additional_num_slots_size );

                if( ( success == 0 ) && ( dest_obj_info != SIXTRL_NULLPTR ) )
                {
                    dest_obj_info = dest_obj_info +
                        NS(ManagedBuffer_get_section_size)(
                            begin, OBJS_ID, slot_size );

                    SIXTRL_ASSERT( ( ( ( uintptr_t )dest_obj_info )
                        % slot_size ) == ZERO_SIZE );

                    #if !defined( _GPUCODE )

                    SIXTRACKLIB_COPY_VALUES( unsigned char,
                        dest_obj_info, &obj_info, sizeof( object_t ) );

                    #else

                    SIXTRACKLIB_COPY_VALUES( unsigned char, dest_obj_info,
                        ( cpy_raw_ptr_t )( uintptr_t )&obj_info,
                            sizeof( object_t )  );

                    #endif /* !defined( _GPUCODE ) */

                    result_object = ( ptr_to_object_t )dest_obj_info;
                }
                else if( success == 0 )
                {
                    success = -1;
                }
            }

            if( success == 0 )
            {
                dest_slots = dest_slots + obj_handle_size;
            }

            if( ( success == 0 ) && ( num_obj_dataptr > ZERO_SIZE ) )
            {
                buf_size_t const current_dataptrs_size =
                    NS(ManagedBuffer_get_section_size)(
                        begin, DATAPTRS_ID, slot_size );

                buf_size_t ii = ZERO_SIZE;

                ptr_to_raw_t dest_dataptrs = ( ( ptr_to_raw_t
                    )NS(ManagedBuffer_get_ptr_to_section)(
                        begin, DATAPTRS_ID, slot_size ) ) +
                    current_dataptrs_size;

                ptr_to_addr_t out_it = ( ptr_to_addr_t )dest_dataptrs;
                raw_t const  z_value = ( raw_t )0u;

                SIXTRL_ASSERT( ( ( ( uintptr_t )dest_dataptrs )
                    % slot_size ) == ZERO_SIZE );

                for( ; ii < num_obj_dataptr ; ++ii, ++out_it )
                {
                    buf_size_t const attr_size =
                        dataptr_counts[ ii ] * dataptr_sizes[ ii ];

                    buf_size_t const attr_extent =
                        NS(ManagedBuffer_get_slot_based_length)(
                            attr_size, slot_size );

                    ptr_to_raw_t ptr_attr_slot =
                        stored_obj_begin + dataptr_offsets[ ii ];

                    address_t const attr_slot_addr =
                        ( address_t )( uintptr_t )ptr_attr_slot;

                    address_t const source_addr = ( attr_slot_addr != ZERO_ADDR )
                        ? *( ( ptr_to_addr_t )ptr_attr_slot ) : ZERO_ADDR;

                    if( source_addr != ZERO_ADDR )
                    {
                        ptr_to_raw_t ptr_attr_src =
                            ( ptr_to_raw_t )( uintptr_t )source_addr;

                        SIXTRACKLIB_COPY_VALUES(
                            raw_t, dest_slots, ptr_attr_src, attr_size );

                        if( attr_extent > attr_size )
                        {
                            buf_size_t const to_fill = attr_extent - attr_size;

                            SIXTRACKLIB_SET_VALUES( raw_t,
                                dest_slots + attr_size, to_fill, z_value );
                        }
                    }
                    else
                    {
                        SIXTRACKLIB_SET_VALUES(
                            raw_t, dest_slots, attr_extent, z_value );
                    }

                    *( ( ptr_to_addr_t )ptr_attr_slot ) =
                        ( address_t )( uintptr_t )dest_slots;

                    *out_it    = attr_slot_addr;
                    dest_slots = dest_slots + attr_extent;
                }

                SIXTRL_ASSERT( requ_num_dataptrs ==
                    ( cur_num_dataptrs + num_obj_dataptr ) );

                NS(ManagedBuffer_set_section_num_entities)( begin, DATAPTRS_ID,
                        requ_num_dataptrs, slot_size );
            }

            if( success == 0 )
            {
                NS(ManagedBuffer_set_section_num_entities)(
                    begin, SLOTS_ID, requ_num_slots, slot_size );

                NS(ManagedBuffer_set_section_num_entities)(
                    begin, OBJS_ID, requ_num_objects, slot_size );

                buffer->num_objects = requ_num_objects;
            }
            else
            {
                result_object = SIXTRL_NULLPTR;
            }
        }
    }

    return result_object;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Buffer_init_from_data)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const max_data_buffer_length )
{
    typedef NS(buffer_size_t)        buf_size_t;
    typedef NS(buffer_addr_t)        address_t;
    typedef unsigned char            raw_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC raw_t* ptr_to_raw_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE  = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID = ( buf_size_t )4u;

    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    if( ( buffer != SIXTRL_NULLPTR ) && ( slot_size > ZERO_SIZE ) &&
        ( begin  != SIXTRL_NULLPTR ) )
    {
        buf_size_t in_length = ZERO_SIZE;

        SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size ) == ZERO_SIZE );

        if( NS(ManagedBuffer_needs_remapping)( begin, slot_size ) )
        {
            if( NS(ManagedBuffer_remap)( begin, slot_size ) != 0 )
            {
                return success;
            }
        }

        SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)( begin, slot_size ) );
        in_length = NS(ManagedBuffer_get_buffer_length)( begin, slot_size );

        SIXTRL_ASSERT( in_length <= max_data_buffer_length );

        if( !NS(Buffer_uses_datastore)( buffer ) )
        {
            buffer->data_addr     = ( address_t )( uintptr_t )begin;
            buffer->data_size     = in_length;
            buffer->data_capacity = max_data_buffer_length;

            buffer->object_addr = ( address_t )( uintptr_t
                )NS(ManagedBuffer_get_const_objects_index_begin)(
                    begin, slot_size );

            buffer->num_objects = NS(ManagedBuffer_get_section_num_entities)(
                begin, OBJECTS_ID, slot_size );

            buffer->datastore_flags =
                SIXTRL_BUFFER_USES_DATASTORE |
                SIXTRL_BUFFER_DATASTORE_ALLOW_APPENDS |
                SIXTRL_BUFFER_DATASTORE_ALLOW_CLEAR   |
                SIXTRL_BUFFER_DATASTORE_ALLOW_REMAPPING;

            success = 0;
        }
        else if( ( NS(Buffer_allow_clear)( buffer ) ) &&
                 ( NS(Buffer_allow_modify_datastore_contents)( buffer ) ) )
        {
            bool const needs_resizing =
                ( in_length > NS(Buffer_get_capacity)( buffer ) );

            success = ( !needs_resizing ) ? 0 : -1;

            if( ( needs_resizing ) &&
                ( NS(Buffer_owns_datastore)( buffer ) ) &&
                ( NS(Buffer_allow_resize)( buffer ) ) )
            {
                #if !defined( NDEBUG ) || \
                    ( !defined( _GPUCODE ) && !defined( __CUDACC__ ) )

                address_t const datastore_addr =
                    NS(Buffer_get_datastore_begin_addr)( buffer );

                SIXTRL_ASSERT( datastore_addr != ( address_t )0u );

                #endif /* !defined( NDEBUG ) ||
                        ( !defined( _GPUCODE ) && !defined( __CUDACC__ ) ) */

                if( NS(Buffer_uses_mempool_datastore)( buffer ) )
                {
                    #if !defined( _GPUCODE ) && !defined( __CUDACC__ )

                    typedef NS(MemPool)                   mem_pool_t;
                    typedef NS(AllocResult)               alloc_result_t;
                    typedef SIXTRL_ARGPTR_DEC mem_pool_t* ptr_mem_pool_t;

                    ptr_mem_pool_t mem_pool = ( ptr_mem_pool_t)(
                        uintptr_t )datastore_addr;

                    SIXTRL_ASSERT( mem_pool != SIXTRL_NULLPTR );
                    NS(MemPool_clear)( mem_pool );

                    success = NS(MemPool_reserve_aligned)(
                        mem_pool, in_length, slot_size );

                    if( success == 0 )
                    {
                        alloc_result_t result = NS(MemPool_append_aligned)(
                            mem_pool, in_length, slot_size );

                        if( NS(AllocResult_valid)( &result ) )
                        {
                            buffer->data_addr = ( address_t )( uintptr_t
                                )NS(AllocResult_get_pointer)( &result );

                            buffer->data_capacity =
                                NS(AllocResult_get_length)( &result );

                            buffer->data_size = ZERO_SIZE;

                            success = 0;
                        }
                    }

                    #endif /* !defined( _GPUCODE ) && !defined( __CUDACC__ ) */
                }
            }

            if( ( success == 0 ) &&
                ( in_length <= NS(Buffer_get_capacity)( buffer ) ) )
            {
                ptr_to_raw_t out_begin = ( ptr_to_raw_t )( uintptr_t
                    )NS(Buffer_get_data_begin_addr)( buffer );

                SIXTRACKLIB_COPY_VALUES( raw_t, out_begin, begin, in_length );

                if( NS(ManagedBuffer_needs_remapping)( out_begin, slot_size ) )
                {
                    success = NS(ManagedBuffer_remap)( out_begin, slot_size );
                }

                if( success == 0 )
                {
                    buffer->data_addr   = ( address_t )( uintptr_t )out_begin;
                    buffer->data_size   = in_length;
                    buffer->object_addr = ( address_t )( uintptr_t
                        )NS(ManagedBuffer_get_const_objects_index_begin)(
                            begin, slot_size );

                    buffer->num_objects =
                        NS(ManagedBuffer_get_section_num_entities)(
                            begin, OBJECTS_ID, slot_size );

                    buffer->datastore_flags |=
                        SIXTRL_BUFFER_DATASTORE_ALLOW_APPENDS |
                        SIXTRL_BUFFER_DATASTORE_ALLOW_REMAPPING;
                }

                success = 0;
            }
        }
    }

    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Buffer_init_on_flat_memory)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const buffer_capacity )
{
    return NS(Buffer_init_on_flat_memory_detailed)(
        buffer, data_buffer_begin, buffer_capacity, 0u, 0u, 0u, 0u );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Buffer_init_on_flat_memory_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const buffer_capacity,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_ranges )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) address_t;

    buf_size_t new_buffer_size = ( buf_size_t )0u;
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    int success = NS(ManagedBuffer_init)( data_buffer_begin, &new_buffer_size,
        max_num_objects,  max_num_slots, max_num_dataptrs,
            max_num_garbage_ranges, buffer_capacity, slot_size );

    if( success == 0 )
    {
        SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID = ( buf_size_t )4u;

        buffer->data_addr       = ( address_t )( uintptr_t )data_buffer_begin;
        buffer->data_size       = new_buffer_size;
        buffer->data_capacity   = buffer_capacity;

        buffer->num_objects     = NS(ManagedBuffer_get_section_num_entities)(
            data_buffer_begin, OBJECTS_ID, slot_size );

        buffer->object_addr     = ( address_t )( uintptr_t
            )NS(ManagedBuffer_get_ptr_to_section_data)(
                data_buffer_begin, OBJECTS_ID, slot_size );

        buffer->datastore_addr  = buffer->data_addr;
        buffer->datastore_flags =
            SIXTRL_BUFFER_DATASTORE_ALLOW_APPENDS |
            SIXTRL_BUFFER_USES_DATASTORE |
            SIXTRL_BUFFER_DATASTORE_ALLOW_CLEAR |
            SIXTRL_BUFFER_DATASTORE_ALLOW_DELETES |
            SIXTRL_BUFFER_DATASTORE_ALLOW_REMAPPING;
    }

    return success;
}

/* ========================================================================= */

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*
NS(Buffer_get_const_object)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const object_index )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_object_t;
    typedef NS(buffer_size_t) buf_size_t;

    ptr_object_t ptr_obj = NS(Buffer_get_const_objects_begin)( buffer );
    buf_size_t const num_objects = NS(Buffer_get_num_of_objects)( buffer );

    if( ( ptr_obj != SIXTRL_NULLPTR ) && ( object_index <= num_objects ) )
    {
        ptr_obj = ptr_obj + object_index;
    }
    else if( ptr_obj != SIXTRL_NULLPTR )
    {
        ptr_obj = SIXTRL_NULLPTR;
    }

    return ptr_obj;
}

SIXTRL_INLINE SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* NS(Buffer_get_object)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const object_index )
{
    return ( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
        )NS(Buffer_get_const_object)( buffer, object_index );
}

SIXTRL_INLINE SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*
NS(Buffer_get_const_objects_begin)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buf )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_to_obj_t;
    typedef uintptr_t uptr_t;
    return ( ptr_to_obj_t )( uptr_t )NS(Buffer_get_objects_begin_addr)( buf );
}

SIXTRL_INLINE SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*
NS(Buffer_get_const_objects_end)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buf )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_to_obj_t;
    return ( ptr_to_obj_t )( uintptr_t )NS(Buffer_get_objects_end_addr)( buf );
}

SIXTRL_INLINE SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
NS(Buffer_get_objects_begin)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* ptr_to_obj_t;
    return ( ptr_to_obj_t )NS(Buffer_get_const_objects_begin)( buffer);
}

SIXTRL_INLINE SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
NS(Buffer_get_objects_end)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* ptr_to_obj_t;
    return ( ptr_to_obj_t )NS(Buffer_get_const_objects_end)( buffer);
}

#endif /* !defined( _GPUCODE ) */

/* ========================================================================= */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRKL_COMMON_BUFFER_BUFFER_GENERIC_H__ */
/*end: sixtracklib/sixtracklib/common/buffer/buffer_generic.h */
