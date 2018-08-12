#ifndef SIXTRKL_COMMON_IMPL_BUFFER_GENERIC_H__
#define SIXTRKL_COMMON_IMPL_BUFFER_GENERIC_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <limits>
        #include <vector>
    #else /* defined( __cplusplus ) */
        #include <stdbool.h>
        #include <stddef.h>
        #include <stdint.h>
        #include <stdlib.h>
        #include <limits.h>
    #endif /* defined( __cplusplus ) */
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

SIXTRL_FN SIXTRL_STATIC unsigned char const*
NS(Object_get_const_begin_ptr)(
    const struct NS(Object) *const SIXTRL_RESTRICT object );

SIXTRL_FN SIXTRL_STATIC unsigned char* NS(Object_get_begin_ptr)(
    struct NS(Object)* SIXTRL_RESTRICT object );

SIXTRL_FN SIXTRL_STATIC  void NS(Object_set_begin_ptr)(
    struct NS(Object)* SIXTRL_RESTRICT object,
    unsigned char* SIXTRL_RESTRICT begin_ptr );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC unsigned char const* NS(Buffer_get_const_data_begin)(
    const struct NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC unsigned char const* NS(Buffer_get_const_data_end)(
    const struct NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC  unsigned char* NS(Buffer_get_data_begin)(
    struct NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC  unsigned char* NS(Buffer_get_data_end)(
    struct NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(Object) const* NS(Buffer_get_const_objects_begin)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(Object) const* NS(Buffer_get_const_objects_end)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(Object)*
NS(Buffer_get_objects_begin)( NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(Object)*
NS(Buffer_get_objects_end)( NS(Buffer)* SIXTRL_RESTRICT buffer );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC unsigned char const*
NS(Buffer_get_const_ptr_to_section)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_section_id );

SIXTRL_FN SIXTRL_STATIC unsigned char const*
NS(Buffer_get_const_ptr_to_section_data)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_section_id );

SIXTRL_FN SIXTRL_STATIC unsigned char* NS(Buffer_get_ptr_to_section)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_section_id );

SIXTRL_FN SIXTRL_STATIC unsigned char* NS(Buffer_get_ptr_to_section_data)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_section_id );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(Buffer_get_section_extent_generic)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_section_id );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(Buffer_get_section_data_extent_generic)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_section_id );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Buffer_get_section_size)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_section_id );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Buffer_get_section_data_size)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_section_id );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(Buffer_get_section_num_entities_generic)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_section_id );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Buffer_get_section_entity_size)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_section_id );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(Buffer_get_section_max_num_entities)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_section_id );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(Buffer_get_size_from_header_generic)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC void NS(Buffer_set_section_extent_generic)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const section_extent );

SIXTRL_FN SIXTRL_STATIC void NS(Buffer_set_section_num_entities_generic)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const section_num_elements );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_clear_and_resize_datastore_generic)(
    struct NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const new_buffer_capacity );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_clear_generic)(
    struct NS(Buffer)* SIXTRL_RESTRICT buffer );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_reset_generic)(
    struct NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_reset_detailed_generic)(
    struct NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const initial_max_num_objects,
    NS(buffer_size_t) const initial_max_num_slots,
    NS(buffer_size_t) const initial_max_num_dataptrs,
    NS(buffer_size_t) const initial_max_num_garbage_elems );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_needs_remapping_generic)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC int
NS(Buffer_remap_get_addr_offset_generic)(
    NS(buffer_addr_diff_t)* SIXTRL_RESTRICT ptr_to_addr_offset,
    unsigned char const* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(Buffer_remap_get_buffer_size_generic)(
    unsigned char const* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(Buffer_remap_get_header_size_generic)(
    unsigned char const* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_remap_header_generic)(
    unsigned char* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT addr_offset,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_remap_section_slots_generic)(
    unsigned char*  SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_addr_t)** SIXTRL_RESTRICT ptr_to_slots_begin_itself,
    NS(buffer_size_t)*  SIXTRL_RESTRICT ptr_to_slots_capacity,
    NS(buffer_size_t)*  SIXTRL_RESTRICT ptr_to_num_slots,
    NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT addr_offsets,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_remap_section_objects_generic)(
    unsigned char*      SIXTRL_RESTRICT data_buffer_begin,
    struct NS(Object)** SIXTRL_RESTRICT ptr_to_objects_begin_itself,
    NS(buffer_size_t)*  SIXTRL_RESTRICT ptr_to_objects_capacity,
    NS(buffer_size_t)*  SIXTRL_RESTRICT ptr_to_num_of_objects,
    NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT addr_offsets,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_remap_section_dataptrs_generic)(
    unsigned char*      SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_addr_t)** SIXTRL_RESTRICT ptr_to_dataptrs_begin_itself,
    NS(buffer_size_t)*  SIXTRL_RESTRICT ptr_to_dataptrs_capacity,
    NS(buffer_size_t)*  SIXTRL_RESTRICT ptr_to_num_of_dataptrs,
    NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT addr_offsets,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_remap_section_garbage_generic)(
    unsigned char*      SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_addr_t)** SIXTRL_RESTRICT ptr_to_garbage_begin_itself,
    NS(buffer_size_t)*  SIXTRL_RESTRICT ptr_to_garbage_capacity,
    NS(buffer_size_t)*  SIXTRL_RESTRICT ptr_to_num_garbage_elements,
    NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT addr_offsets,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_remap_flat_buffer_generic)(
    unsigned char* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_remap_generic)(
    struct NS(Buffer)* SIXTRL_RESTRICT buffer );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_reserve_generic)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_elems );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC void NS(Buffer_free_generic)(
    NS(Buffer)* SIXTRL_RESTRICT buffer );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(Object)* NS(Buffer_add_object)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    const void *const SIXTRL_RESTRICT object,
    NS(buffer_size_t)        const object_size,
    NS(object_type_id_t)     const type_id,
    NS(buffer_size_t)        const num_obj_dataptr,
    const NS(buffer_size_t) *const SIXTRL_RESTRICT obj_dataptr_offsets,
    const NS(buffer_size_t) *const SIXTRL_RESTRICT obj_dataptr_sizes,
    const NS(buffer_size_t) *const SIXTRL_RESTRICT obj_dataptr_counts );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_init_from_data)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    unsigned char* SIXTRL_RESTRICT data_buffer_begin );

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_init_on_flat_memory)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    unsigned char* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const buffer_capacity );

SIXTRL_FN SIXTRL_STATIC  int NS(Buffer_init_on_flat_memory_detailed)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    unsigned char* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const buffer_capacity,
    NS(buffer_size_t) const initial_max_num_objects,
    NS(buffer_size_t) const initial_max_num_slots,
    NS(buffer_size_t) const initial_max_num_dataptrs,
    NS(buffer_size_t) const initial_max_num_garbage_elems );

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* *
 * *****         Implementation of inline functions and methods        ***** *
 * ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/mem_pool.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

SIXTRL_INLINE unsigned char const* NS(Object_get_const_begin_ptr)(
    const NS(Object) *const SIXTRL_RESTRICT object )
{
    typedef unsigned char const* ptr_to_raw_t;
    return ( ptr_to_raw_t )( uintptr_t )NS(Object_get_begin_addr)( object );
}

SIXTRL_INLINE unsigned char* NS(Object_get_begin_ptr)(
    NS(Object)* SIXTRL_RESTRICT object )
{
    return ( unsigned char* )NS(Object_get_const_begin_ptr)( object );
}

SIXTRL_INLINE void NS(Object_set_begin_ptr)(
    NS(Object)* SIXTRL_RESTRICT object,
    unsigned char* SIXTRL_RESTRICT begin_ptr )
{
    typedef NS(buffer_addr_t) address_t;
    NS(Object_set_begin_addr)( object, ( address_t )( uintptr_t )begin_ptr );
    return;
}

/* ========================================================================= */

SIXTRL_INLINE unsigned char const* NS(Buffer_get_const_data_begin)(
    const NS(Buffer) *const SIXTRL_RESTRICT buf )
{
    typedef unsigned char const* ptr_to_raw_t;
    return ( ptr_to_raw_t )( uintptr_t )NS(Buffer_get_data_begin_addr)( buf );
}

SIXTRL_INLINE unsigned char const* NS(Buffer_get_const_data_end)(
    const NS(Buffer) *const SIXTRL_RESTRICT buf )
{
    typedef unsigned char const* ptr_to_raw_t;
    return ( ptr_to_raw_t )( uintptr_t )NS(Buffer_get_data_end_addr)( buf );
}

SIXTRL_INLINE unsigned char* NS(Buffer_get_data_begin)(
    NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef unsigned char* ptr_to_raw_t;
    return ( ptr_to_raw_t )NS(Buffer_get_const_data_begin)( buffer );
}

SIXTRL_INLINE unsigned char* NS(Buffer_get_data_end)(
    NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef unsigned char* ptr_to_raw_t;
    return ( ptr_to_raw_t )NS(Buffer_get_const_data_end)( buffer );
}

/* ========================================================================= */

SIXTRL_INLINE NS(Object) const* NS(Buffer_get_const_objects_begin)(
    const NS(Buffer) *const SIXTRL_RESTRICT buf )
{
    typedef NS(Object) const* ptr_to_obj_t;
    typedef uintptr_t         uptr_t;
    return ( ptr_to_obj_t )( uptr_t )NS(Buffer_get_objects_begin_addr)( buf );
}

SIXTRL_INLINE NS(Object) const* NS(Buffer_get_const_objects_end)(
    const NS(Buffer) *const SIXTRL_RESTRICT buf )
{
    typedef NS(Object) const* ptr_to_obj_t;
    return ( ptr_to_obj_t )( uintptr_t )NS(Buffer_get_objects_end_addr)( buf );
}

SIXTRL_INLINE NS(Object)* NS(Buffer_get_objects_begin)(
    NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(Object)* ptr_to_obj_t;
    return ( ptr_to_obj_t )NS(Buffer_get_const_objects_begin)( buffer);
}

SIXTRL_INLINE NS(Object)* NS(Buffer_get_objects_end)(
    NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(Object)* ptr_to_obj_t;
    return ( ptr_to_obj_t )NS(Buffer_get_const_objects_end)( buffer);
}

/* ========================================================================= */

SIXTRL_INLINE unsigned char const* NS(Buffer_get_const_ptr_to_section)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const section_id )
{
    typedef unsigned char const* ptr_to_raw_t;
    typedef NS(buffer_size_t)    buf_size_t;
    typedef NS(buffer_addr_t)    address_t;
    typedef address_t const*     ptr_to_addr_t;

    ptr_to_raw_t  ptr_to_section = SIXTRL_NULLPTR;
    buf_size_t const   slot_size = NS(Buffer_get_slot_size)( buffer );
    buf_size_t const header_size = NS(Buffer_get_header_size)( buffer );

    ptr_to_raw_t ptr_header = ( ptr_to_raw_t )( uintptr_t
        )NS(Buffer_get_data_begin_addr)( buffer );

    buf_size_t const addr_size  =
        NS(Buffer_get_slot_based_length)( sizeof( address_t ), slot_size );

    buf_size_t const header_offset = addr_size * section_id;

    SIXTRL_ASSERT( slot_size     > ( buf_size_t )0u );
    SIXTRL_ASSERT( addr_size     > ( buf_size_t )0u );
    SIXTRL_ASSERT( header_offset < header_size );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping_generic)( buffer ) );

    if( ( ptr_header != SIXTRL_NULLPTR ) && ( header_offset <  header_size ) &&
        ( ( ( ( uintptr_t )ptr_header ) % slot_size ) == 0u ) )
    {
        ptr_to_section = ( ptr_to_raw_t )( uintptr_t )(
            *( ptr_to_addr_t )( ptr_header + header_offset ) );
    }

    return ptr_to_section;
}

SIXTRL_INLINE unsigned char const*
NS(Buffer_get_const_ptr_to_section_data)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const section_id )
{
    typedef unsigned char const* ptr_to_raw_t;
    typedef NS(buffer_size_t)    buf_size_t;
    typedef NS(buffer_addr_t)    address_t;
    typedef address_t const*     ptr_to_addr_t;

    ptr_to_raw_t ptr_to_section_data = SIXTRL_NULLPTR;
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    buf_size_t const addr_size  =
        NS(Buffer_get_slot_based_length)( sizeof( address_t ), slot_size );

    ptr_to_raw_t ptr_section_begin =
        NS(Buffer_get_const_ptr_to_section)( buffer, section_id );

    SIXTRL_ASSERT( slot_size > 0u );
    SIXTRL_ASSERT( ( ( ( uintptr_t )ptr_section_begin ) % slot_size ) == 0u );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping_generic)( buffer ) );

    if( ptr_section_begin != SIXTRL_NULLPTR )
    {
        buf_size_t const extent = *( ( ptr_to_addr_t )ptr_section_begin );
        buf_size_t const section_header_size = 2u * addr_size;

        ptr_to_section_data = ( extent >= section_header_size )
            ? ( ptr_section_begin + section_header_size )
            : SIXTRL_NULLPTR;
    }

    return ptr_to_section_data;
}

SIXTRL_INLINE unsigned char* NS(Buffer_get_ptr_to_section)(
    NS(Buffer)* SIXTRL_RESTRICT buffer, NS(buffer_size_t) const section_id )
{
    return ( unsigned char* )NS(Buffer_get_const_ptr_to_section)(
        buffer, section_id );
}

SIXTRL_INLINE unsigned char* NS(Buffer_get_ptr_to_section_data)(
    NS(Buffer)* SIXTRL_RESTRICT buffer, NS(buffer_size_t) const section_id )
{
    return ( unsigned char* )NS(Buffer_get_const_ptr_to_section_data)(
        buffer, section_id );
}

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_section_extent_generic)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const section_id )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) address_t;
    typedef address_t const*  ptr_to_addr_t;

    ptr_to_addr_t ptr_to_section = ( ptr_to_addr_t
        )NS(Buffer_get_const_ptr_to_section)( buffer, section_id );

    return ( ptr_to_section != SIXTRL_NULLPTR )
        ? ( buf_size_t )( *ptr_to_section ) : ( buf_size_t )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_section_data_extent_generic)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const section_id )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) address_t;

    buf_size_t const slot_size =
        NS(Buffer_get_slot_size)( buffer );

    buf_size_t const addr_size =
        NS(Buffer_get_slot_based_length)( sizeof( address_t ), slot_size );

    buf_size_t const section_header_size = 2u * addr_size;

    buf_size_t const extent =
        NS(Buffer_get_section_extent_generic)( buffer, section_id );

    SIXTRL_ASSERT( !NS(Buffer_needs_remapping_generic)( buffer ) );

    return ( extent >= section_header_size )
        ? ( extent - section_header_size ) : ( buf_size_t )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_section_size)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const section_id )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const section_hd_size =
        NS(Buffer_get_section_header_size)( buffer );

    buf_size_t const section_data_size = NS(Buffer_get_section_data_size)(
        buffer, section_id );

    buf_size_t const section_size = section_hd_size + section_data_size;

    SIXTRL_ASSERT( section_size <= NS(Buffer_get_section_extent_generic)(
        buffer, section_id ) );

    return section_size;
}

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_section_data_size)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const section_id )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_entries =
        NS(Buffer_get_section_num_entities_generic)( buffer, section_id );

    buf_size_t const entry_size = NS(Buffer_get_section_entity_size)(
        buffer, section_id );

    buf_size_t const section_data_size = num_entries * entry_size;

    SIXTRL_ASSERT( ( section_data_size +
        NS(Buffer_get_section_header_size)( buffer ) ) <=
        NS(Buffer_get_section_extent_generic)( buffer, section_id ) );

    return section_data_size;
}

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_section_num_entities_generic)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const section_id )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) address_t;
    typedef address_t const*  ptr_to_addr_t;

    ptr_to_addr_t ptr_to_section = ( ptr_to_addr_t
        )NS(Buffer_get_const_ptr_to_section)( buffer, section_id );

    SIXTRL_ASSERT( !NS(Buffer_needs_remapping_generic)( buffer ) );

    return ( ptr_to_section != SIXTRL_NULLPTR )
        ? ( buf_size_t )ptr_to_section[ 1 ] : ( buf_size_t )0u;
}


SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_section_entity_size)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_section_id )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t element_size = ( buf_size_t )0u;
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    switch( header_section_id )
    {
        case 3u:
        {
            element_size = slot_size;
            break;
        }

        case 4u:
        {
            typedef NS(Object) object_t;
            element_size = NS(Buffer_get_slot_based_length)(
                sizeof( object_t ), slot_size );

            break;
        }

        case 5u:
        {
            typedef NS(buffer_addr_t) const* ptr_to_addr_t;
            element_size = NS(Buffer_get_slot_based_length)(
                sizeof( ptr_to_addr_t ), slot_size );
            break;
        }

        case 6u:
        {
            typedef NS(BufferGarbage) garbage_range_t;
            element_size = NS(Buffer_get_slot_based_length)(
                sizeof( garbage_range_t ), slot_size );

            break;
        }

        default:
        {
            element_size = ( buf_size_t )0u;
        }
    };

    return element_size;
}

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_section_max_num_entities)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_section_id )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const section_data_extent =
        NS(Buffer_get_section_data_extent_generic)( buffer, header_section_id );

    buf_size_t const section_elem_size =
        NS(Buffer_get_section_entity_size)( buffer, header_section_id );

    SIXTRL_ASSERT( section_elem_size != 0u );
    SIXTRL_ASSERT( ( section_data_extent % section_elem_size ) == 0u );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping_generic)( buffer ) );

    return ( section_data_extent / section_elem_size );
}

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_get_size_from_header_generic)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(buffer_addr_t)       address_t;
    typedef unsigned char const*    ptr_to_raw_t;
    typedef address_t const*        ptr_to_addr_t;

    buf_size_t const header_len = NS(Buffer_get_header_size)( buffer );
    buf_size_t const slot_size  = NS(Buffer_get_slot_size)( buffer );
    buf_size_t const addr_size  =
        NS(Buffer_get_slot_based_length)( sizeof( address_t ), slot_size );

    ptr_to_raw_t begin = ( ptr_to_raw_t )( uintptr_t
        )NS(Buffer_get_data_begin_addr)( buffer );

    SIXTRL_ASSERT( slot_size != 0u );
    SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size ) == 0u );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping_generic)( buffer ) );

    return ( ( begin != SIXTRL_NULLPTR ) && ( header_len > addr_size ) )
        ? ( *( ( ptr_to_addr_t )( begin + addr_size ) ) ) : ( buf_size_t )0u;
}

SIXTRL_INLINE void NS(Buffer_set_section_extent_generic)(
    NS(Buffer)* SIXTRL_RESTRICT buffer, NS(buffer_size_t) const section_id,
    NS(buffer_size_t) const section_extent )
{
    typedef NS(buffer_addr_t) address_t;
    typedef address_t*        ptr_to_addr_t;

    ptr_to_addr_t ptr_to_section = ( ptr_to_addr_t
        )NS(Buffer_get_ptr_to_section)( buffer, section_id );

    SIXTRL_ASSERT( !NS(Buffer_needs_remapping_generic)( buffer ) );

    if( ptr_to_section != SIXTRL_NULLPTR )
    {
        ptr_to_section[ 0 ] = section_extent;
    }

    return;
}

SIXTRL_INLINE void NS(Buffer_set_section_num_entities_generic)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const section_id,
    NS(buffer_size_t) const section_num_elements )
{
    typedef NS(buffer_addr_t) address_t;
    typedef address_t*        ptr_to_addr_t;

    ptr_to_addr_t ptr_to_section = ( ptr_to_addr_t
        )NS(Buffer_get_ptr_to_section)( buffer, section_id );

    SIXTRL_ASSERT( !NS(Buffer_needs_remapping_generic)( buffer ) );

    if( ptr_to_section != SIXTRL_NULLPTR )
    {
        ptr_to_section[ 1 ] = section_num_elements;
    }

    return;
}

/* ========================================================================= */

SIXTRL_INLINE int NS(Buffer_clear_and_resize_datastore_generic)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const new_buffer_capacity )
{
    int success = -1;

    typedef NS(buffer_size_t) buf_size_t;
    typedef unsigned char     raw_t;
    typedef raw_t*            ptr_to_raw_t;

    if( NS(Buffer_uses_datastore)( buffer ) )
    {
        typedef NS(buffer_flags_t)  buffer_flags_t;
        typedef NS(buffer_addr_t)   address_t;

        buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

        ptr_to_raw_t data_begin = ( ptr_to_raw_t )( uintptr_t
            )NS(Buffer_get_data_begin_addr)( buffer );

        buf_size_t data_capacity = ( buf_size_t )0u;

        address_t const datastore_addr =
            NS(Buffer_get_datastore_begin_addr)( buffer );

        buffer_flags_t const special_buffer_flags =
            NS(Buffer_get_datastore_special_flags)( buffer );

        SIXTRL_ASSERT( slot_size > 0u );

        if( ( NS(Buffer_uses_mempool_datastore)( buffer ) ) &&
            ( special_buffer_flags == ( buffer_flags_t )0u ) )
        {
            #if !defined( _GPUCODE )
            typedef NS(MemPool) mem_pool_t;
            typedef mem_pool_t* ptr_to_mem_pool_t;

            ptr_to_mem_pool_t ptr_mem_pool =
                ( ptr_to_mem_pool_t )( uintptr_t )datastore_addr;

            data_capacity = NS(MemPool_get_size)( ptr_mem_pool );

            if( ( data_begin    == SIXTRL_NULLPTR ) ||
                ( data_capacity <= new_buffer_capacity ) )
            {
                NS(MemPool_clear)( ptr_mem_pool );
                success = ( NS(MemPool_reserve_aligned)( ptr_mem_pool,
                    new_buffer_capacity, slot_size ) ) ? 0 : -1;

                if( success == 0 )
                {
                    NS(AllocResult) result = NS(MemPool_append_aligned)(
                        ptr_mem_pool, new_buffer_capacity, slot_size );

                    if( NS(AllocResult_valid)( &result ) )
                    {
                        data_begin    = NS(AllocResult_get_pointer)( &result );
                        data_capacity = NS(AllocResult_get_length)( &result );
                    }
                    else
                    {
                        success = -1;
                    }
                }
            }
            else
            {
                success = 0;
            }

            #else /* !defined( _GPUCODE ) */
            success = -1;
            #endif /* !defined( _GPUCODE ) */
        }

        if( ( success == 0 ) && ( data_begin != SIXTRL_NULLPTR ) &&
            ( ( ( ( uintptr_t )data_begin ) % slot_size ) == 0u ) &&
            ( data_capacity >= new_buffer_capacity ) &&
            ( ( data_capacity % slot_size ) == 0u ) )
        {
            buf_size_t const hdr_len = NS(Buffer_get_header_size)( buffer );

            if( ( hdr_len > 0u ) && ( hdr_len < data_capacity ) )
            {
                typedef NS(buffer_addr_t)   address_t;
                typedef address_t*          ptr_to_addr_t;

                ptr_to_addr_t ptr_header = ( ptr_to_addr_t )data_begin;

                SIXTRACKLIB_SET_VALUES( raw_t, data_begin, hdr_len, 0 );
                ptr_header[ 0 ] = ( address_t )( uintptr_t )data_begin;

                buffer->data_addr     = ( address_t )( uintptr_t )data_begin;
                buffer->data_capacity = ( buf_size_t )data_capacity;
                buffer->data_size     = ( buf_size_t )0u;
            }
            else
            {
                success = -1;
            }
        }
    }

    return success;
}

SIXTRL_INLINE int NS(Buffer_clear_generic)(
    NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    int success = 0;

    typedef NS(buffer_size_t)   buf_size_t;
    typedef NS(buffer_addr_t)   address_t;
    typedef unsigned char       raw_t;
    typedef raw_t*              ptr_to_raw_t;
    typedef address_t*          ptr_to_addr_t;

    ptr_to_addr_t ptr_header = ( ptr_to_addr_t )( uintptr_t
        )NS(Buffer_get_data_begin_addr)( buffer );

    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    SIXTRL_ASSERT( slot_size > 0u );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping_generic)( buffer ) );

    if( ( ptr_header != SIXTRL_NULLPTR ) &&
        ( ( ( ( uintptr_t )ptr_header ) % slot_size ) == 0u ) )
    {
        address_t const base_addr = ( address_t )( uintptr_t )ptr_header;

        if( base_addr == ptr_header[ 0 ] )
        {
            buf_size_t const SLOTS_ID    = 3u;
            buf_size_t const OBJS_ID     = 4u;
            buf_size_t const DATAPTRS_ID = 5u;
            buf_size_t const GARBAGE_ID  = 6u;

            buf_size_t const num_slots  =
                NS(Buffer_get_section_num_entities_generic)( buffer, SLOTS_ID);

            buf_size_t const num_objects =
                NS(Buffer_get_section_num_entities_generic)( buffer, OBJS_ID);

            buf_size_t const num_dataptrs =
                NS(Buffer_get_section_num_entities_generic)(
                    buffer, DATAPTRS_ID);

            buf_size_t const num_garbage_elems =
                NS(Buffer_get_section_num_entities_generic)(
                    buffer, GARBAGE_ID );

            if( num_slots > 0u )
            {
                raw_t const Z = ( raw_t )0u;

                ptr_to_raw_t section_begin =
                    NS(Buffer_get_ptr_to_section_data)(
                        buffer, SLOTS_ID );

                buf_size_t const data_extent =
                    NS(Buffer_get_section_data_extent_generic)(
                        buffer, SLOTS_ID );

                SIXTRL_ASSERT( section_begin != SIXTRL_NULLPTR );
                SIXTRL_ASSERT( data_extent   >  0u );
                SIXTRACKLIB_SET_VALUES( raw_t, section_begin, data_extent, Z );

                NS(Buffer_set_section_num_entities_generic)(
                    buffer, SLOTS_ID, 0u );
            }

            if( num_objects > 0u )
            {
                typedef NS(Object) object_t;

                object_t* obj_it  =
                    ( object_t* )NS(Buffer_get_ptr_to_section_data)(
                        buffer, OBJS_ID );

                object_t* obj_end = obj_it + num_objects;

                SIXTRL_ASSERT( obj_it  != SIXTRL_NULLPTR );
                SIXTRL_ASSERT( ( sizeof( object_t ) * num_objects ) <=
                    NS(Buffer_get_section_data_extent_generic)(
                        buffer, OBJS_ID ) );

                for( ; obj_it != obj_end ; ++obj_it )
                {
                    NS(Object_preset)( obj_it );
                }

                NS(Buffer_set_section_num_entities_generic)(
                    buffer, OBJS_ID, 0u );
            }

            if( num_dataptrs > 0u )
            {
                ptr_to_addr_t dataptrs_it = ( ptr_to_addr_t
                    )NS(Buffer_get_ptr_to_section_data)(
                        buffer, DATAPTRS_ID );

                ptr_to_addr_t dataptrs_end = dataptrs_it + num_dataptrs;

                SIXTRL_ASSERT( dataptrs_it != SIXTRL_NULLPTR );
                SIXTRL_ASSERT( ( sizeof( ptr_to_addr_t ) * num_dataptrs ) <=
                    NS(Buffer_get_section_data_extent_generic)(
                        buffer, DATAPTRS_ID ) );

                for( ; dataptrs_it != dataptrs_end ; ++dataptrs_it )
                {
                    *dataptrs_it = ( address_t )0u;
                }

                NS(Buffer_set_section_num_entities_generic)(
                    buffer, DATAPTRS_ID, 0u );
            }

            if( num_garbage_elems > 0u )
            {
                raw_t const Z = ( raw_t )0u;

                ptr_to_raw_t section_begin =
                    NS(Buffer_get_ptr_to_section_data)(
                        buffer, GARBAGE_ID );

                buf_size_t const data_extent =
                    NS(Buffer_get_section_data_extent_generic)(
                        buffer, GARBAGE_ID );

                SIXTRL_ASSERT( section_begin != SIXTRL_NULLPTR );
                SIXTRL_ASSERT( data_extent   >  0u );
                SIXTRACKLIB_SET_VALUES( raw_t, section_begin, data_extent, Z );

                NS(Buffer_set_section_num_entities_generic)(
                    buffer, GARBAGE_ID, 0u );
            }

            success = 0;
        }
    }

    return success;
}

/* ========================================================================= */

SIXTRL_INLINE int NS(Buffer_reset_generic)(
    NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_reset_detailed_generic)( buffer, 0u, 0u, 0u, 0u );
}

SIXTRL_INLINE int NS(Buffer_reset_detailed_generic)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const initial_max_num_objects,
    NS(buffer_size_t) const initial_max_num_slots,
    NS(buffer_size_t) const initial_max_num_dataptrs,
    NS(buffer_size_t) const initial_max_num_garbage_elems )
{
    int success = 0;

    typedef NS(buffer_size_t)   buf_size_t;
    typedef NS(buffer_addr_t)   address_t;
    typedef unsigned char*      ptr_to_raw_t;
    typedef address_t*          ptr_to_addr_t;

    ptr_to_raw_t begin = ( ptr_to_raw_t )( uintptr_t
        )NS(Buffer_get_data_begin_addr)( buffer );

    buf_size_t const slot_size   = NS(Buffer_get_slot_size)( buffer );
    buf_size_t buffer_capacity   = NS(Buffer_get_capacity)( buffer );
    buf_size_t const header_size = NS(Buffer_get_header_size)( buffer );

    SIXTRL_ASSERT( slot_size > 0u );
    SIXTRL_ASSERT( header_size >= NS(BUFFER_DEFAULT_HEADER_SIZE) );
    SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size ) == 0u );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping_generic)( buffer ) );

    if( ( NS(Buffer_uses_datastore)( buffer ) ) &&
        ( NS(Buffer_allow_modify_datastore_contents)( buffer ) ) &&
        ( NS(Buffer_allow_clear)( buffer ) ) )
    {
        buf_size_t const addr_size = NS(Buffer_get_slot_based_length)(
            sizeof( address_t ), slot_size );

        buf_size_t const obj_info_size = NS(Buffer_get_slot_based_length)(
            sizeof( NS(Object) ), slot_size );

        buf_size_t const dataptrs_size = NS(Buffer_get_slot_based_length)(
            sizeof( ptr_to_addr_t ), slot_size );

        buf_size_t const garbage_range_size = NS(Buffer_get_slot_based_length)(
            sizeof( NS(BufferGarbage) ), slot_size );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        buf_size_t const slots_extent =
            2u * addr_size + initial_max_num_slots * slot_size;

        buf_size_t const objects_extent =
            2u * addr_size + initial_max_num_objects * obj_info_size;

        buf_size_t const dataptrs_extent =
            2u * addr_size + initial_max_num_dataptrs * dataptrs_size;

        buf_size_t const garbage_extent = 2u * addr_size +
            initial_max_num_garbage_elems * garbage_range_size;

        buf_size_t const required_buffer_capacity =
            header_size + slots_extent + objects_extent +
                          dataptrs_extent + garbage_extent;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        ptr_to_addr_t ptr_header = ( ptr_to_addr_t )begin;
        SIXTRACKLIB_SET_VALUES( address_t, ptr_header, 1u, ( address_t )0u );

        success = 0;

        if( buffer_capacity < required_buffer_capacity )
        {
            buf_size_t const twice_current_buffer_capacity =
                2u * buffer_capacity;

            buf_size_t const new_buffer_capacity =
                ( twice_current_buffer_capacity > required_buffer_capacity )
                    ? twice_current_buffer_capacity
                    : required_buffer_capacity;

            success = NS(Buffer_clear_and_resize_datastore_generic)(
                buffer, new_buffer_capacity );

            if( success == 0 )
            {
                begin = ( ptr_to_raw_t )( uintptr_t
                    )NS(Buffer_get_data_begin_addr)( buffer );

                ptr_header = ( ptr_to_addr_t )begin;
                buffer_capacity = NS(Buffer_get_capacity)( buffer );
                SIXTRL_ASSERT( begin != SIXTRL_NULLPTR );
                SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size ) == 0u );
            }
        }

        if( ( success == 0 ) &&
            ( buffer_capacity >= required_buffer_capacity ) )
        {
            typedef unsigned char raw_t;
            buf_size_t const SECTION_HDR_SIZE = 2u * addr_size;

            address_t begin_addr  = ( uintptr_t )begin;
            address_t buffer_size = required_buffer_capacity;
            address_t header_len  = header_size;

            ptr_to_raw_t slot_section_begin = begin + header_size;
            ptr_to_raw_t obj_section_begin = slot_section_begin + slots_extent;

            ptr_to_raw_t dataptr_section_begin =
                obj_section_begin + objects_extent;

            ptr_to_raw_t garbage_section_begin =
                dataptr_section_begin + dataptrs_extent;

            ptr_to_raw_t dest   = begin;
            uintptr_t dest_addr = ( uintptr_t )dest;
            uintptr_t end_addr  = dest_addr + buffer_capacity;

            success = ( ( ( ( uintptr_t )garbage_section_begin ) +
                garbage_extent ) <= end_addr ) ? 0 : -1;

            SIXTRL_ASSERT( slots_extent    >= SECTION_HDR_SIZE );
            SIXTRL_ASSERT( objects_extent  >= SECTION_HDR_SIZE );
            SIXTRL_ASSERT( dataptrs_extent >= SECTION_HDR_SIZE );
            SIXTRL_ASSERT( garbage_extent  >= SECTION_HDR_SIZE );

            /* ------------------------------------------------------------- */
            /* header: */

            if( success == 0 )
            {
                SIXTRACKLIB_COPY_VALUES( raw_t, dest, &begin_addr, addr_size );
                dest = dest + addr_size;
                if( ( ( uintptr_t )dest ) >= end_addr ) success = -1;
            }

            if( success == 0 )
            {
                SIXTRACKLIB_COPY_VALUES( raw_t, dest, &buffer_size, addr_size );
                dest = dest + addr_size;
                if( ( ( uintptr_t )dest ) >= end_addr ) success = -1;
            }

            if( success == 0 )
            {
                SIXTRACKLIB_COPY_VALUES( raw_t, dest, &header_len, addr_size );
                dest = dest + addr_size;
                if( ( ( uintptr_t )dest ) >= end_addr ) success = -1;
            }

            if( success == 0 )
            {
                uintptr_t section_addr = ( uintptr_t )slot_section_begin;
                SIXTRACKLIB_COPY_VALUES( raw_t, dest, &section_addr, addr_size );
                dest = dest + addr_size;
                if( ( ( uintptr_t )dest ) >= end_addr ) success = -1;
            }

            if( success == 0 )
            {
                uintptr_t section_addr = ( uintptr_t )obj_section_begin;
                SIXTRACKLIB_COPY_VALUES( raw_t, dest, &section_addr, addr_size );
                dest = dest + addr_size;
                if( ( ( uintptr_t )dest ) >= end_addr ) success = -1;
            }

            if( success == 0 )
            {
                uintptr_t section_addr = ( uintptr_t )dataptr_section_begin;
                SIXTRACKLIB_COPY_VALUES( raw_t, dest, &section_addr, addr_size );
                dest = dest + addr_size;
                if( ( ( uintptr_t )dest ) >= end_addr ) success = -1;
            }

            if( success == 0 )
            {
                uintptr_t section_addr = ( uintptr_t )garbage_section_begin;
                SIXTRACKLIB_COPY_VALUES( raw_t, dest, &section_addr, addr_size );
                dest = dest + addr_size;
                if( ( ( uintptr_t )dest ) >= end_addr ) success = -1;
            }

            /* ------------------------------------------------------------- */
            /* slots section: */

            if( success == 0 )
            {
                address_t temp = slots_extent;
                dest = slot_section_begin;
                SIXTRACKLIB_COPY_VALUES( raw_t, dest, &temp, addr_size );

                temp = ( address_t )0u;
                SIXTRACKLIB_COPY_VALUES( raw_t, dest + addr_size,
                                         &temp, addr_size );

                dest = dest + slots_extent;
                if( ( ( uintptr_t )dest ) > ( ( uintptr_t )obj_section_begin ) )
                {
                    success = -1;
                }
            }

            /* ------------------------------------------------------------- */
            /* objects section: */

            if( success == 0 )
            {
                address_t temp = objects_extent;

                dest = obj_section_begin;
                SIXTRACKLIB_COPY_VALUES( raw_t, dest, &temp, addr_size );

                temp = ( address_t )0u;
                SIXTRACKLIB_COPY_VALUES( raw_t, dest + addr_size,
                                         &temp, addr_size );

                dest = dest + objects_extent;
                if( ( ( uintptr_t )dest ) > ( ( uintptr_t )dataptr_section_begin ) )
                {
                    success = -1;
                }
            }

            /* ------------------------------------------------------------- */
            /* dataptrs section: */

            if( success == 0 )
            {
                address_t temp = dataptrs_extent;

                dest = dataptr_section_begin;
                SIXTRACKLIB_COPY_VALUES( raw_t, dest, &temp, addr_size );

                temp = ( address_t )0u;
                SIXTRACKLIB_COPY_VALUES( raw_t, dest + addr_size,
                                         &temp, addr_size );

                dest = dest + dataptrs_extent;
                if( ( ( uintptr_t )dest ) > ( ( uintptr_t )garbage_section_begin ) )
                {
                    success = -1;
                }
            }

            /* ------------------------------------------------------------- */
            /* garbage section: */

            if( success == 0 )
            {
                address_t temp = garbage_extent;

                dest = garbage_section_begin;
                SIXTRACKLIB_COPY_VALUES( raw_t, dest, &temp, addr_size );

                temp = ( address_t )0u;
                SIXTRACKLIB_COPY_VALUES( raw_t, dest + addr_size,
                                         &temp, addr_size );

                dest = dest + garbage_extent;
                if( ( ( uintptr_t )dest ) >= end_addr )
                {
                    success = -1;
                }
            }

            /* ------------------------------------------------------------- */

            if( success == 0 )
            {
                SIXTRL_ASSERT( buffer_size <= buffer_capacity );
                SIXTRL_ASSERT( obj_section_begin != SIXTRL_NULLPTR );

                buffer->data_size     = buffer_size;
                buffer->data_capacity = buffer_capacity;

                buffer->num_objects   = ( buf_size_t )0u;
                buffer->object_addr   = ( address_t )( uintptr_t )(
                    obj_section_begin + SECTION_HDR_SIZE );

                NS(Buffer_clear_generic)( buffer );
            }
        }
    }

    return success;
}

/* ========================================================================= */

SIXTRL_INLINE bool NS(Buffer_needs_remapping_generic)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    typedef NS(buffer_addr_t)   address_t;
    typedef address_t const*    ptr_to_addr_t;

    ptr_to_addr_t ptr_header = ( ptr_to_addr_t )( uintptr_t
        )NS(Buffer_get_data_begin_addr)( buffer );

    address_t const begin_addr = ( address_t )ptr_header;

    return ( ( ptr_header      != SIXTRL_NULLPTR ) &&
             ( ptr_header[ 0 ] != begin_addr ) );
}


SIXTRL_INLINE int NS(Buffer_remap_get_addr_offset_generic)(
    NS(buffer_addr_diff_t)* SIXTRL_RESTRICT ptr_to_addr_offset,
    unsigned char const* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(buffer_addr_t)       address_t;
    typedef NS(buffer_addr_diff_t)  addr_diff_t;
    typedef address_t*              ptr_to_addr_t;

    int success = -1;

    if( ( data_buffer_begin  != SIXTRL_NULLPTR ) &&
        ( slot_size > ( buf_size_t )0u ) &&
        ( ( ( ( uintptr_t )data_buffer_begin ) % slot_size ) == 0u ) )
    {
        addr_diff_t addr_offset = ( addr_diff_t )0;

        address_t const stored_base_addr =
            ( data_buffer_begin != SIXTRL_NULLPTR )
                ? *( ( ptr_to_addr_t )data_buffer_begin ) : ( address_t )0u;

        address_t const base_address =
            ( address_t )( uintptr_t )data_buffer_begin;

        addr_offset = ( base_address >= stored_base_addr )
            ?  ( addr_diff_t )( base_address - stored_base_addr )
            : -( addr_diff_t )( stored_base_addr - base_address );

        if(  ptr_to_addr_offset != SIXTRL_NULLPTR )
        {
            *ptr_to_addr_offset = addr_offset;
            success = 0;
        }
    }

    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_remap_get_buffer_size_generic)(
    unsigned char const* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(buffer_addr_t)       address_t;
    typedef address_t*              ptr_to_addr_t;

    buf_size_t buffer_size = ( buf_size_t )0u;

    if( ( data_buffer_begin != SIXTRL_NULLPTR ) &&
        ( slot_size > ( buf_size_t )0u ) &&
        ( ( ( ( uintptr_t )data_buffer_begin ) % slot_size ) == 0u ) )
    {
        buf_size_t const addr_size = NS(Buffer_get_slot_based_length)(
            sizeof( address_t ), slot_size );

        SIXTRL_ASSERT( ( buf_size_t )( ( ( ptr_to_addr_t )(
            data_buffer_begin ) )[ 2 ] ) > addr_size );

        buffer_size = *( ( ptr_to_addr_t )( data_buffer_begin + addr_size ) );
    }

    return buffer_size;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(buffer_size_t) NS(Buffer_remap_get_header_size_generic)(
    unsigned char const* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) address_t;
    typedef address_t const*  ptr_to_addr_t;

    buf_size_t header_size = ( buf_size_t )0u;

    if( ( data_buffer_begin != SIXTRL_NULLPTR ) &&
        ( slot_size > ( buf_size_t )0u ) &&
        ( ( ( ( uintptr_t )data_buffer_begin ) % slot_size ) == 0u ) )
    {
        buf_size_t const addr_size = NS(Buffer_get_slot_based_length)(
            sizeof( address_t ), slot_size );

        buf_size_t const offset = ( address_t )2u * addr_size;

        SIXTRL_ASSERT( ( buf_size_t )( ( ( ptr_to_addr_t )(
            data_buffer_begin ) )[ 2 ] ) > offset );

        header_size = *( ( ptr_to_addr_t )( data_buffer_begin + offset ) );
    }

    return header_size;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Buffer_remap_header_generic)(
    unsigned char* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT addr_offsets,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(buffer_addr_t)       address_t;
    typedef NS(buffer_addr_diff_t)  addr_diff_t;
    typedef address_t*              ptr_to_addr_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR buf_size_t const ZERO_ADDR = ( address_t  )0u;

    if( ( data_buffer_begin != SIXTRL_NULLPTR ) &&
        ( addr_offsets      != SIXTRL_NULLPTR ) &&
        ( slot_size > ZERO_SIZE ) &&
        ( ( ( ( uintptr_t )data_buffer_begin ) % slot_size ) == 0u ) )
    {
        ptr_to_addr_t begin_addr = ( ptr_to_addr_t )data_buffer_begin;

        address_t const stored_base_addr =
            ( data_buffer_begin != SIXTRL_NULLPTR )
                ? *( ( ptr_to_addr_t )data_buffer_begin ) : ZERO_ADDR;

        uintptr_t base_address = ( uintptr_t )data_buffer_begin;
        addr_diff_t const base_addr_offset = base_address - stored_base_addr;

        if( base_addr_offset != addr_offsets[ 0 ] )
        {
            return success;
        }

        if( base_addr_offset != ( addr_diff_t )0  )
        {
            addr_diff_t const slots_addr_offset    = addr_offsets[ 1 ];
            addr_diff_t const objects_addr_offset  = addr_offsets[ 2 ];
            addr_diff_t const dataptrs_addr_offset = addr_offsets[ 3 ];
            addr_diff_t const garbage_addr_offset  = addr_offsets[ 4 ];

            buf_size_t  const addr_size = NS(Buffer_get_slot_based_length)(
                sizeof( address_t ), slot_size );

            ptr_to_addr_t ptr_slots_begin = ( ptr_to_addr_t )(
                data_buffer_begin + 3u * addr_size );

            ptr_to_addr_t ptr_objects_begin = ( ptr_to_addr_t )(
                data_buffer_begin + 4u * addr_size );

            ptr_to_addr_t ptr_dataptrs_begin = ( ptr_to_addr_t )(
                data_buffer_begin + 5u * addr_size );

            ptr_to_addr_t ptr_garbage_begin = ( ptr_to_addr_t )(
                data_buffer_begin + 6u * addr_size );

            address_t const remapped_base_addr = NS(Buffer_perform_addr_shift)(
                stored_base_addr, base_addr_offset, slot_size );

            SIXTRL_ASSERT( ptr_slots_begin    != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( ptr_objects_begin  != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( ptr_dataptrs_begin != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( ptr_garbage_begin  != SIXTRL_NULLPTR );

            SIXTRL_ASSERT( stored_base_addr    <= *ptr_slots_begin    );
            SIXTRL_ASSERT( *ptr_slots_begin    <= *ptr_objects_begin  );
            SIXTRL_ASSERT( *ptr_objects_begin  <= *ptr_dataptrs_begin );
            SIXTRL_ASSERT( *ptr_dataptrs_begin <= *ptr_garbage_begin  );

            if( remapped_base_addr == ( address_t )(
                    uintptr_t )data_buffer_begin )
            {
                address_t const remapped_slots_begin_addr =
                    NS(Buffer_perform_addr_shift)( *ptr_slots_begin,
                        slots_addr_offset, slot_size );

                address_t const remapped_obj_begin_addr =
                    NS(Buffer_perform_addr_shift)( *ptr_objects_begin,
                        objects_addr_offset, slot_size );

                address_t const remapped_dataptrs_begin_addr =
                    NS(Buffer_perform_addr_shift)( *ptr_dataptrs_begin,
                        dataptrs_addr_offset, slot_size );

                address_t const remapped_garbage_begin_addr =
                    NS(Buffer_perform_addr_shift)( *ptr_garbage_begin,
                        garbage_addr_offset, slot_size );

                if( ( remapped_slots_begin_addr    != ZERO_ADDR ) &&
                    ( remapped_obj_begin_addr      != ZERO_ADDR ) &&
                    ( remapped_dataptrs_begin_addr != ZERO_ADDR ) &&
                    ( remapped_garbage_begin_addr  != ZERO_ADDR ) &&
                    ( remapped_base_addr           != ZERO_ADDR ) )
                {
                    SIXTRL_ASSERT( remapped_base_addr <
                                   remapped_slots_begin_addr );

                    SIXTRL_ASSERT( remapped_slots_begin_addr <=
                                   remapped_obj_begin_addr );

                    SIXTRL_ASSERT( remapped_obj_begin_addr <=
                                   remapped_dataptrs_begin_addr );

                    SIXTRL_ASSERT( remapped_dataptrs_begin_addr <=
                                   remapped_garbage_begin_addr );

                    *begin_addr         = remapped_base_addr;
                    *ptr_slots_begin    = remapped_slots_begin_addr;
                    *ptr_objects_begin  = remapped_obj_begin_addr;
                    *ptr_dataptrs_begin = remapped_dataptrs_begin_addr;
                    *ptr_garbage_begin  = remapped_garbage_begin_addr;

                    success = 0;
                }
            }
        }
        else
        {
            success = 0;
        }
    }

    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Buffer_remap_section_slots_generic)(
    unsigned char*      SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_addr_t)** SIXTRL_RESTRICT ptr_to_slots_begin_itself,
    NS(buffer_size_t)*  SIXTRL_RESTRICT ptr_to_slots_capacity,
    NS(buffer_size_t)*  SIXTRL_RESTRICT ptr_to_num_slots,
    NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT addr_offsets,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(buffer_addr_t)*      ptr_to_addr_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    ptr_to_addr_t ptr_slots_begin = SIXTRL_NULLPTR;
    buf_size_t    slots_capacity  = ZERO_SIZE;
    buf_size_t    num_slots       = ZERO_SIZE;

    if( ( data_buffer_begin != SIXTRL_NULLPTR ) &&
        ( addr_offsets      != SIXTRL_NULLPTR ) &&
        ( slot_size         != ZERO_SIZE ) &&
        ( ( ( ( uintptr_t )data_buffer_begin ) % slot_size ) == 0u ) )
    {
        typedef NS(buffer_addr_t) address_t;

        SIXTRL_STATIC_VAR buf_size_t const SECTION_ID = 3u;

        buf_size_t const addr_size = NS(Buffer_get_slot_based_length)(
            sizeof( address_t ), slot_size );

        ptr_to_addr_t ptr_section_begin = ( ptr_to_addr_t )( uintptr_t )(
            *( ptr_to_addr_t )( data_buffer_begin + SECTION_ID * addr_size ) );

        if( ptr_section_begin != SIXTRL_NULLPTR )
        {
            slots_capacity  =  ptr_section_begin[ 0 ];
            num_slots       =  ptr_section_begin[ 1 ];
            ptr_slots_begin = &ptr_section_begin[ 2 ];

            success = ( slots_capacity >= ( num_slots * slot_size ) ) ? 0 : -1;
        }
    }

    if(  ptr_to_slots_begin_itself != SIXTRL_NULLPTR )
    {
        *ptr_to_slots_begin_itself = ptr_slots_begin;
    }

    if(  ptr_to_slots_capacity != SIXTRL_NULLPTR )
    {
        *ptr_to_slots_capacity  = slots_capacity;
    }

    if(  ptr_to_num_slots != SIXTRL_NULLPTR )
    {
        *ptr_to_num_slots  = num_slots;
    }

    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Buffer_remap_section_objects_generic)(
    unsigned char*      SIXTRL_RESTRICT data_buffer_begin,
    struct NS(Object)** SIXTRL_RESTRICT ptr_to_objects_begin_itself,
    NS(buffer_size_t)*  SIXTRL_RESTRICT ptr_to_objects_capacity,
    NS(buffer_size_t)*  SIXTRL_RESTRICT ptr_to_num_of_objects,
    NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT addr_offsets,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)      buf_size_t;
    typedef NS(buffer_addr_t)      address_t;
    typedef NS(buffer_addr_diff_t) addr_diff_t;
    typedef address_t*             ptr_to_addr_t;
    typedef struct NS(Object)      object_t;
    typedef object_t*              ptr_to_object_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR buf_size_t const ZERO_ADDR = ( address_t  )0u;

    ptr_to_object_t ptr_objects_begin = SIXTRL_NULLPTR;
    buf_size_t objects_capacity       = ZERO_SIZE;
    buf_size_t num_objects            = ZERO_SIZE;

    if( ( data_buffer_begin != SIXTRL_NULLPTR ) &&
        ( addr_offsets      != SIXTRL_NULLPTR ) &&
        ( slot_size         != ZERO_SIZE ) &&
        ( ( ( ( uintptr_t )data_buffer_begin ) % slot_size ) == 0u ) )
    {
        SIXTRL_STATIC_VAR buf_size_t const OBJECTS_SECTION_ID = 4u;

        buf_size_t const addr_size = NS(Buffer_get_slot_based_length)(
            sizeof( address_t ), slot_size );

        ptr_to_addr_t ptr_section_begin =
            ( ptr_to_addr_t )( uintptr_t )( *( ptr_to_addr_t )(
                data_buffer_begin + OBJECTS_SECTION_ID * addr_size ) );

        if( ptr_section_begin != SIXTRL_NULLPTR )
        {
            buf_size_t const obj_size = NS(Buffer_get_slot_based_length)(
                sizeof( object_t ), slot_size );

            objects_capacity  =  ptr_section_begin[ 0 ];
            num_objects       =  ptr_section_begin[ 1 ];
            ptr_objects_begin = ( ptr_to_object_t )&ptr_section_begin[ 2 ];

            success = ( ( obj_size != ZERO_SIZE ) &&
                ( objects_capacity >= ( obj_size * num_objects ) ) )
                    ? 0 : -1;
        }

        if( ( success == 0 ) && ( num_objects > ZERO_SIZE ) )
        {
            SIXTRL_STATIC_VAR buf_size_t const SLOTS_SECTION_ID = 3u;

            addr_diff_t const slots_addr_offset = addr_offsets[ 1 ];

            ptr_to_addr_t ptr_slots_section = ( ptr_to_addr_t )( uintptr_t )(
                *( ptr_to_addr_t )( data_buffer_begin +
                    SLOTS_SECTION_ID * addr_size ) );

            buf_size_t const num_slots =
                ( ptr_slots_section != SIXTRL_NULLPTR )
                    ? ptr_slots_section[ 1 ] : ZERO_SIZE;

            buf_size_t const slots_extent = num_slots * slot_size;

            address_t const slots_section_begin_addr =
                ( ptr_slots_section != SIXTRL_NULLPTR )
                    ? ( ( ( address_t )( uintptr_t )ptr_slots_section )
                        + 2u * addr_size ) : ZERO_ADDR;

            address_t const slots_section_end_addr =
                slots_section_begin_addr + slots_extent;

            address_t min_valid_obj_addr = slots_section_begin_addr;

            ptr_to_object_t obj_it   = ptr_objects_begin;
            ptr_to_object_t obj_end  = ptr_objects_begin + num_objects;

            SIXTRL_ASSERT( ptr_slots_section != SIXTRL_NULLPTR );

            for( ; obj_it != obj_end ; ++obj_it )
            {
                buf_size_t const obj_size = NS(Object_get_size)( obj_it );

                address_t  const obj_begin_addr =
                    NS(Object_get_begin_addr)( obj_it );

                address_t const remapped_obj_begin_addr =
                    NS(Buffer_perform_addr_shift)(
                        obj_begin_addr, slots_addr_offset, slot_size );

                SIXTRL_ASSERT( ( min_valid_obj_addr % slot_size ) == 0u );
                SIXTRL_ASSERT(  min_valid_obj_addr <= slots_section_end_addr );

                SIXTRL_ASSERT(   remapped_obj_begin_addr != ZERO_ADDR );
                SIXTRL_ASSERT( ( remapped_obj_begin_addr % slot_size ) == 0u );

                if( ( remapped_obj_begin_addr < slots_section_begin_addr ) ||
                    ( remapped_obj_begin_addr > slots_section_end_addr   ) ||
                    ( remapped_obj_begin_addr < min_valid_obj_addr       ) )
                {
                    success = -1;
                    break;
                }

                min_valid_obj_addr = obj_begin_addr + obj_size;
                NS(Object_set_begin_addr)( obj_it, remapped_obj_begin_addr );
            }
        }
    }

    if(  ptr_to_objects_begin_itself != SIXTRL_NULLPTR )
    {
        *ptr_to_objects_begin_itself =  ptr_objects_begin;
    }

    if(  ptr_to_objects_capacity    != SIXTRL_NULLPTR )
    {
        *ptr_to_objects_capacity     = objects_capacity;
    }

    if(  ptr_to_num_of_objects      != SIXTRL_NULLPTR )
    {
        *ptr_to_num_of_objects       = num_objects;
    }

    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Buffer_remap_section_dataptrs_generic)(
    unsigned char*      SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_addr_t)** SIXTRL_RESTRICT ptr_to_dataptrs_begin_itself,
    NS(buffer_size_t)*  SIXTRL_RESTRICT ptr_to_dataptrs_capacity,
    NS(buffer_size_t)*  SIXTRL_RESTRICT ptr_to_num_dataptrs,
    NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT addr_offsets,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(buffer_addr_t)       address_t;
    typedef NS(buffer_addr_diff_t)  addr_diff_t;
    typedef address_t*              ptr_to_addr_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR address_t  const ZERO_ADDR = ( address_t  )0u;

    ptr_to_addr_t ptr_dataptrs_begin = SIXTRL_NULLPTR;
    buf_size_t dataptrs_capacity     = ZERO_SIZE;
    buf_size_t num_dataptrs          = ZERO_SIZE;

    if( ( data_buffer_begin != SIXTRL_NULLPTR ) &&
        ( addr_offsets      != SIXTRL_NULLPTR ) &&
        ( slot_size         != ZERO_SIZE ) &&
        ( ( ( ( uintptr_t )data_buffer_begin ) % slot_size ) == 0u ) )
    {
        SIXTRL_STATIC_VAR buf_size_t const DATAPTRS_SECTION_ID = 5u;

        buf_size_t const addr_size = NS(Buffer_get_slot_based_length)(
            sizeof( address_t ), slot_size );

        ptr_to_addr_t ptr_section_begin =
            ( ptr_to_addr_t )( uintptr_t )( *( ptr_to_addr_t )(
                data_buffer_begin + ( DATAPTRS_SECTION_ID * addr_size ) ) );

        buf_size_t const dataptr_size = NS(Buffer_get_slot_based_length)(
                sizeof( ptr_to_addr_t ), slot_size );

        buf_size_t dataptrs_extent = ZERO_SIZE;

        if( ptr_section_begin != SIXTRL_NULLPTR )
        {
            dataptrs_capacity  =  ptr_section_begin[ 0 ];
            num_dataptrs       =  ptr_section_begin[ 1 ];
            ptr_dataptrs_begin = &ptr_section_begin[ 2 ];

            dataptrs_extent    = dataptr_size * num_dataptrs;

            success = ( ( dataptr_size != ZERO_SIZE ) &&
                ( dataptrs_capacity >= dataptrs_extent ) ) ? 0 : -1;
        }

        if( ( success == 0 ) && ( num_dataptrs > ZERO_SIZE ) )
        {
            SIXTRL_STATIC_VAR buf_size_t const SLOTS_SECTION_ID = 3u;

            addr_diff_t const slots_addr_offset = addr_offsets[ 1 ];

            ptr_to_addr_t ptr_slots_section = ( ptr_to_addr_t )( uintptr_t )(
                *( ptr_to_addr_t )( data_buffer_begin +
                    SLOTS_SECTION_ID * addr_size ) );

            buf_size_t const num_slots =
                ( ptr_slots_section != SIXTRL_NULLPTR )
                    ? ptr_slots_section[ 1 ] : ZERO_SIZE;

            buf_size_t const slots_extent = num_slots * slot_size;

            address_t const slots_section_begin_addr =
                ( ptr_slots_section != SIXTRL_NULLPTR )
                    ? ( ( ( address_t )( uintptr_t )ptr_slots_section )
                        + 2u * addr_size ) : ZERO_ADDR;

            address_t const slots_section_end_addr =
                slots_section_begin_addr + slots_extent;

            ptr_to_addr_t dataptr_it  = ptr_dataptrs_begin;
            ptr_to_addr_t dataptr_end = ptr_dataptrs_begin + num_dataptrs;

            if( ( slots_extent >= dataptrs_extent ) &&
                ( slots_section_begin_addr != ZERO_ADDR ) )
            {
                for( ; dataptr_it != dataptr_end ; ++dataptr_it )
                {
                    address_t const slot_ptr_addr = *dataptr_it;

                    address_t const remap_slot_ptr_addr =
                        NS(Buffer_perform_addr_shift)( slot_ptr_addr,
                            slots_addr_offset, slot_size );

                    ptr_to_addr_t slot_ptr =
                        ( ptr_to_addr_t )( uintptr_t )remap_slot_ptr_addr;

                    address_t const remap_slot_addr =
                        ( slot_ptr != SIXTRL_NULLPTR ) ?
                            NS(Buffer_perform_addr_shift)( *slot_ptr,
                                slots_addr_offset, slot_size ) : 0;

                    if( ( remap_slot_ptr_addr != ZERO_ADDR ) &&
                        ( remap_slot_addr     != ZERO_ADDR ) &&
                        ( remap_slot_ptr_addr >= slots_section_begin_addr ) &&
                        ( remap_slot_ptr_addr <  slots_section_end_addr   ) &&
                        ( remap_slot_addr     >= slots_section_begin_addr ) &&
                        ( remap_slot_addr     <  slots_section_end_addr   ) )
                    {
                        *slot_ptr   = remap_slot_addr;
                        *dataptr_it = remap_slot_ptr_addr;
                    }
                    else
                    {
                        success = -1;
                        break;
                    }
                }
            }
        }
    }

    if(  ptr_to_dataptrs_begin_itself != SIXTRL_NULLPTR )
    {
        *ptr_to_dataptrs_begin_itself  = ptr_dataptrs_begin;
    }

    if(  ptr_to_dataptrs_capacity     != SIXTRL_NULLPTR )
    {
        *ptr_to_dataptrs_capacity      = dataptrs_capacity;
    }

    if(  ptr_to_num_dataptrs          != SIXTRL_NULLPTR )
    {
        *ptr_to_num_dataptrs           = num_dataptrs;
    }

    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Buffer_remap_section_garbage_generic)(
    unsigned char*      SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_addr_t)** SIXTRL_RESTRICT ptr_to_garbage_begin_itself,
    NS(buffer_size_t)*  SIXTRL_RESTRICT ptr_to_garbage_capacity,
    NS(buffer_size_t)*  SIXTRL_RESTRICT ptr_to_num_garbage_elements,
    NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT addr_offsets,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(buffer_addr_t)       address_t;
    typedef NS(buffer_addr_diff_t)  addr_diff_t;
    typedef address_t*              ptr_to_addr_t;
    typedef NS(BufferGarbage)       garbage_range_t;
    typedef garbage_range_t*        ptr_to_garbage_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    ptr_to_addr_t ptr_garbage_begin   = SIXTRL_NULLPTR;
    buf_size_t garbage_range_capacity = ZERO_SIZE;
    buf_size_t num_garbage_ranges     = ZERO_SIZE;

    if( ( data_buffer_begin != SIXTRL_NULLPTR ) &&
        ( addr_offsets      != SIXTRL_NULLPTR ) &&
        ( slot_size         != ZERO_SIZE ) &&
        ( ( ( ( uintptr_t )data_buffer_begin ) % slot_size ) == 0u ) )
    {
        SIXTRL_STATIC_VAR buf_size_t const GARBAGE_SECTION_ID = 6u;

        buf_size_t const addr_size = NS(Buffer_get_slot_based_length)(
            sizeof( address_t ), slot_size );

        ptr_to_addr_t ptr_section_begin =
            ( ptr_to_addr_t )( uintptr_t )( *( ptr_to_addr_t )(
                data_buffer_begin + ( GARBAGE_SECTION_ID * addr_size ) ) );

        buf_size_t const garbage_range_size = NS(Buffer_get_slot_based_length)(
                sizeof( ptr_to_garbage_t ), slot_size );

        buf_size_t garbage_extent = ZERO_SIZE;

        if( ptr_section_begin != SIXTRL_NULLPTR )
        {
            garbage_range_capacity =  ptr_section_begin[ 0 ];
            num_garbage_ranges     =  ptr_section_begin[ 1 ];
            ptr_garbage_begin      = &ptr_section_begin[ 2 ];

            garbage_extent = garbage_range_size * num_garbage_ranges;

            success = ( ( garbage_range_size != ZERO_SIZE ) &&
                ( garbage_range_capacity >= garbage_extent ) ) ? 0 : -1;
        }

        if( ( success == 0 ) && ( num_garbage_ranges > ZERO_SIZE ) )
        {
            SIXTRL_STATIC_VAR address_t  const ZERO_ADDR = ( address_t  )0u;
            SIXTRL_STATIC_VAR buf_size_t const SLOTS_SECTION_ID = 3u;

            addr_diff_t const slots_addr_offset = addr_offsets[ 1 ];

            ptr_to_addr_t ptr_slots_section = ( ptr_to_addr_t )( uintptr_t )(
                *( ptr_to_addr_t )( data_buffer_begin +
                    SLOTS_SECTION_ID * addr_size ) );

            buf_size_t const num_slots =
                ( ptr_slots_section != SIXTRL_NULLPTR )
                    ? ( ptr_slots_section[ 1 ] ) : ZERO_SIZE;

            buf_size_t const slots_extent = num_slots * slot_size;

            address_t const slots_section_begin_addr =
                ( ptr_slots_section != SIXTRL_NULLPTR )
                    ? ( ( ( address_t )( uintptr_t )ptr_slots_section )
                        + 2u * addr_size ) : ZERO_ADDR;

            address_t const slots_section_end_addr =
                slots_section_begin_addr + slots_extent;

            ptr_to_garbage_t it = ( slots_section_begin_addr != ZERO_ADDR )
                ? ( ptr_to_garbage_t )ptr_garbage_begin : SIXTRL_NULLPTR;

            ptr_to_garbage_t end = ( it != SIXTRL_NULLPTR )
                ? ( it + num_garbage_ranges ) : SIXTRL_NULLPTR;

            for( ; it != end ; ++it )
            {
                address_t const garbage_begin_addr =
                    NS(BufferGarbage_get_begin_addr)( it );

                address_t const remap_slot_addr =
                    NS(Buffer_perform_addr_shift)( garbage_begin_addr,
                        slots_addr_offset, slot_size );

                if( ( remap_slot_addr >= slots_section_begin_addr ) &&
                    ( remap_slot_addr <  slots_section_end_addr   ) )
                {
                    NS(BufferGarbage_set_begin_addr)( it, remap_slot_addr );
                }
                else
                {
                    success = -1;
                    break;
                }
            }
        }
    }

    if(  ptr_to_garbage_begin_itself != SIXTRL_NULLPTR )
    {
        *ptr_to_garbage_begin_itself  = ptr_garbage_begin;
    }

    if(  ptr_to_garbage_capacity     != SIXTRL_NULLPTR )
    {
        *ptr_to_garbage_capacity      = garbage_range_capacity;
    }

    if(  ptr_to_num_garbage_elements != SIXTRL_NULLPTR )
    {
        *ptr_to_num_garbage_elements  = num_garbage_ranges;
    }

    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Buffer_remap_flat_buffer_generic)(
    unsigned char* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(buffer_addr_t)       address_t;
    typedef NS(buffer_addr_diff_t)  addr_diff_t;
    typedef address_t*              ptr_to_addr_t;
    typedef struct NS(Object)       object_t;
    typedef object_t*               ptr_to_object_t;

    SIXTRL_STATIC_VAR buf_size_t  const ZERO_SIZE = ( buf_size_t  )0u;
    SIXTRL_STATIC_VAR addr_diff_t const ZERO_ADDR = ( addr_diff_t )0;

    ptr_to_addr_t   ptr_slots_begin    = SIXTRL_NULLPTR;
    ptr_to_object_t ptr_objects_begin  = SIXTRL_NULLPTR;
    ptr_to_addr_t   ptr_dataptrs_begin = SIXTRL_NULLPTR;
    ptr_to_addr_t   ptr_garbage_begin  = SIXTRL_NULLPTR;

    buf_size_t      num_slots          = ZERO_SIZE;
    buf_size_t      num_objects        = ZERO_SIZE;
    buf_size_t      num_dataptrs       = ZERO_SIZE;
    buf_size_t      num_garbage        = ZERO_SIZE;

    buf_size_t      slots_capacity     = ZERO_SIZE;
    buf_size_t      objects_capacity   = ZERO_SIZE;
    buf_size_t      dataptrs_capacity  = ZERO_SIZE;
    buf_size_t      garbage_capacity   = ZERO_SIZE;

    addr_diff_t     addr_offset        = ZERO_ADDR;

    int success = NS(Buffer_remap_get_addr_offset_generic)(
        &addr_offset, data_buffer_begin, slot_size );

    if( ( success == 0 ) && ( addr_offset != ZERO_ADDR ) )
    {
        addr_diff_t const addr_offsets[] =
        {
            addr_offset, addr_offset, addr_offset, addr_offset, addr_offset
        };

        success  = NS(Buffer_remap_header_generic)(
            data_buffer_begin, &addr_offsets[ 0 ], slot_size );

        success |= NS(Buffer_remap_section_slots_generic)(
            data_buffer_begin, &ptr_slots_begin, &slots_capacity,
                &num_slots, &addr_offsets[ 0 ], slot_size );

        success |= NS(Buffer_remap_section_objects_generic)(
            data_buffer_begin, &ptr_objects_begin, &objects_capacity,
                &num_objects, &addr_offsets[ 0 ], slot_size );

        success |= NS(Buffer_remap_section_dataptrs_generic)(
            data_buffer_begin, &ptr_dataptrs_begin, &dataptrs_capacity,
                &num_dataptrs, &addr_offsets[ 0 ], slot_size );

        success |= NS(Buffer_remap_section_garbage_generic)(
            data_buffer_begin, &ptr_garbage_begin, &garbage_capacity,
                &num_garbage, &addr_offsets[ 0 ], slot_size );

        if( ( slots_capacity     <  num_slots      ) ||
            ( objects_capacity   <  num_objects    ) ||
            ( dataptrs_capacity  <  num_dataptrs   ) ||
            ( garbage_capacity   <  num_garbage    ) ||
            ( ptr_slots_begin    == SIXTRL_NULLPTR ) ||
            ( ptr_objects_begin  == SIXTRL_NULLPTR ) ||
            ( ptr_dataptrs_begin == SIXTRL_NULLPTR ) ||
            ( ptr_garbage_begin  == SIXTRL_NULLPTR ) )
        {
            success |= -1;
        }
    }

    return success;
}

SIXTRL_INLINE int NS(Buffer_remap_generic)( NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    int success = -1;

    typedef NS(buffer_size_t )  buf_size_t;
    typedef NS(buffer_addr_t )  address_t;
    typedef unsigned char*      ptr_to_raw_t;
    typedef address_t*          ptr_to_addr_t;

    if( NS(Buffer_has_datastore)( buffer ) )
    {
        ptr_to_raw_t data_buffer_begin = ( ptr_to_raw_t )( uintptr_t
            )NS(Buffer_get_data_begin_addr)( buffer );

        ptr_to_addr_t ptr_header   = ( ptr_to_addr_t )data_buffer_begin;
        buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

        SIXTRL_ASSERT( ptr_header != SIXTRL_NULLPTR );

        if( ( ptr_header[ 0 ] != ( address_t )0 ) &&
            ( ptr_header[ 0 ] == ( address_t )( uintptr_t )ptr_header ) )
        {
            success = 0;
        }
        else if( ( ptr_header[ 0 ] != ( address_t )0 ) &&
                 ( slot_size > 0u ) &&
                 ( data_buffer_begin != SIXTRL_NULLPTR ) &&
                 ( ( ( ( uintptr_t )data_buffer_begin ) % slot_size ) == 0u ) )
        {
            success = NS(Buffer_remap_flat_buffer_generic)(
                data_buffer_begin, slot_size );

            if( success == 0 )
            {
                SIXTRL_ASSERT( ptr_header[ 0 ] ==
                    ( address_t )( uintptr_t )ptr_header );

                buffer->data_addr   = ptr_header[ 0 ];
                buffer->data_size   = ptr_header[ 1 ];
                buffer->header_size = ptr_header[ 2 ];

                buffer->object_addr = ( address_t )( uintptr_t
                    )NS(Buffer_get_ptr_to_section_data)( buffer, 4u );

                buffer->num_objects =
                    NS(Buffer_get_section_num_entities_generic)( buffer, 4u );
            }
        }
    }

    return success;
}

/* ========================================================================= */

SIXTRL_INLINE int NS(Buffer_reserve_generic)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_elems )
{
    typedef NS(buffer_size_t)   buf_size_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const SLOTS_ID     = 3u;
    SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID   = 4u;
    SIXTRL_STATIC_VAR buf_size_t const DATAPTRS_ID  = 5u;
    SIXTRL_STATIC_VAR buf_size_t const GARBAGE_ID   = 6u;

    buf_size_t const current_max_num_slots =
        NS(Buffer_get_section_max_num_entities)( buffer, SLOTS_ID );

    buf_size_t const current_max_num_objects =
        NS(Buffer_get_section_max_num_entities)( buffer, OBJECTS_ID );

    buf_size_t const current_max_num_dataptrs =
        NS(Buffer_get_section_max_num_entities)( buffer, DATAPTRS_ID );

    buf_size_t const current_max_num_garbage_elems =
        NS(Buffer_get_section_max_num_entities)( buffer, GARBAGE_ID );

    SIXTRL_ASSERT( !NS(Buffer_needs_remapping_generic)( buffer ) );

    if( ( NS(Buffer_uses_datastore)( buffer ) ) &&
        ( ( max_num_objects       > current_max_num_objects   ) ||
          ( max_num_slots         > current_max_num_slots     ) ||
          ( max_num_dataptrs      > current_max_num_dataptrs  ) ||
          ( max_num_garbage_elems > current_max_num_garbage_elems ) ) )
    {
        typedef NS(buffer_addr_t)       address_t;
        typedef address_t*              ptr_to_addr_t;
        typedef unsigned char*          ptr_to_raw_t;
        typedef address_t const*        ptr_to_const_addr_t;
        typedef NS(buffer_addr_diff_t)  addr_diff_t;

        buf_size_t const section_hd_size =
            NS(Buffer_get_section_header_size)( buffer );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        buf_size_t const requ_max_num_objs =
            ( max_num_objects > current_max_num_objects )
                ? max_num_objects : current_max_num_objects;

        buf_size_t const obj_elem_size =
            NS(Buffer_get_section_entity_size)( buffer, OBJECTS_ID );

        buf_size_t const current_objs_extent =
            section_hd_size + current_max_num_objects * obj_elem_size;

        buf_size_t const requ_objs_extent =
            section_hd_size + requ_max_num_objs * obj_elem_size;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        buf_size_t const requ_max_num_slots =
            ( max_num_slots > current_max_num_slots )
                ? max_num_slots : current_max_num_slots;

        buf_size_t const slots_elem_size =
            NS(Buffer_get_section_entity_size)( buffer, SLOTS_ID );

        buf_size_t const current_slots_extent =
            section_hd_size + current_max_num_slots * slots_elem_size;

        buf_size_t const requ_slots_extent =
            section_hd_size + requ_max_num_slots * slots_elem_size;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        buf_size_t const requ_max_num_dataptrs =
            ( max_num_dataptrs > current_max_num_dataptrs )
                ? max_num_dataptrs : current_max_num_dataptrs;

        buf_size_t const dataptrs_elem_size =
            NS(Buffer_get_section_entity_size)( buffer, DATAPTRS_ID );

        buf_size_t const current_dataptrs_extent =
            section_hd_size + current_max_num_dataptrs * dataptrs_elem_size;

        buf_size_t const requ_dataptrs_extent =
            section_hd_size + requ_max_num_dataptrs * dataptrs_elem_size;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        buf_size_t const requ_max_num_garbage_elems =
            ( max_num_garbage_elems > current_max_num_garbage_elems )
                ? max_num_garbage_elems : current_max_num_garbage_elems;

        buf_size_t const garbage_elem_size =
            NS(Buffer_get_section_entity_size)( buffer, GARBAGE_ID );

        buf_size_t const current_garbage_extent = section_hd_size +
            current_max_num_garbage_elems * garbage_elem_size;

        buf_size_t const requ_garbage_extent =
            section_hd_size + requ_max_num_garbage_elems * garbage_elem_size;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        buf_size_t const header_size = NS(Buffer_get_header_size)( buffer );

        buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

        buf_size_t const requ_buffer_capacity =
            header_size + requ_slots_extent + requ_objs_extent +
                requ_dataptrs_extent + requ_garbage_extent;

        buf_size_t current_buffer_capacity =
            NS(Buffer_get_size_from_header_generic)( buffer );

        addr_diff_t addr_offsets[ 5 ] =
        {
            ( addr_diff_t )0u, ( addr_diff_t )0u, ( addr_diff_t )0u,
            ( addr_diff_t )0u, ( addr_diff_t )0u
        };

        ptr_to_addr_t ptr_header = SIXTRL_NULLPTR;

        SIXTRL_ASSERT( current_buffer_capacity >=
            ( header_size + current_slots_extent + current_objs_extent +
              current_dataptrs_extent + current_garbage_extent ) );

        if( ( requ_buffer_capacity > current_buffer_capacity ) &&
            ( NS(Buffer_owns_datastore)( buffer ) ) &&
            ( NS(Buffer_allow_resize)( buffer ) ) )
        {
            buf_size_t const twice_current_buffer_capacity =
                ( buf_size_t )2u * current_buffer_capacity;

            buf_size_t const new_buffer_capacity =
                ( requ_buffer_capacity > twice_current_buffer_capacity )
                    ? requ_buffer_capacity : twice_current_buffer_capacity;

            if( NS(Buffer_uses_mempool_datastore)( buffer ) )
            {
                #if !defined( _GPUCODE )
                typedef NS(MemPool)  mem_pool_t;
                typedef mem_pool_t*  ptr_to_mem_pool_t;

                address_t const datastore_addr =
                    NS(Buffer_get_datastore_begin_addr)( buffer );

                ptr_to_mem_pool_t ptr_mem_pool =
                    ( ptr_to_mem_pool_t )( uintptr_t )datastore_addr;

                success = NS(MemPool_reserve_aligned)(
                    ptr_mem_pool, new_buffer_capacity, slot_size );

                if( success == 0 )
                {
                    address_t const new_begin_addr = ( address_t )(
                        uintptr_t )NS(MemPool_get_begin_pos)( ptr_mem_pool );

                    buf_size_t const new_buffer_size =
                        NS(MemPool_get_size)( ptr_mem_pool );

                    if( (   new_begin_addr != ( address_t )0u     ) &&
                        ( ( new_begin_addr  %  slot_size ) == 0u  ) &&
                        ( ( new_buffer_size %  slot_size ) == 0u  ) &&
                        ( new_buffer_size  >= new_buffer_capacity ) )
                    {
                        addr_diff_t base_addr_offset = ( addr_diff_t )0u;

                        ptr_header = ( ptr_to_addr_t )( uintptr_t )new_begin_addr;

                        success = NS(Buffer_remap_get_addr_offset_generic)(
                            &base_addr_offset, ( ptr_to_raw_t )ptr_header,
                                slot_size );

                        if( success == 0 )
                        {
                            buffer->data_addr       = new_begin_addr;
                            buffer->data_capacity   = new_buffer_size;
                            current_buffer_capacity = new_buffer_size;

                            addr_offsets[ 0 ] = base_addr_offset;
                            addr_offsets[ 1 ] = base_addr_offset;
                            addr_offsets[ 2 ] = base_addr_offset;
                            addr_offsets[ 3 ] = base_addr_offset;
                            addr_offsets[ 4 ] = base_addr_offset;
                        }
                    }
                    else
                    {
                        success = -1;
                    }
                }

                #endif /* !defined( _GPUCODE ) */
            }
        }

        if( ( success == 0 ) &&
            ( requ_buffer_capacity <= current_buffer_capacity ) )
        {
            buf_size_t const addr_size = NS(Buffer_get_slot_based_length)(
                sizeof( address_t ), slot_size );

            buf_size_t const current_slots_offset   = header_size;
            buf_size_t const current_slots_size =
                NS(Buffer_get_section_size)( buffer, SLOTS_ID );

            buf_size_t const new_slots_offset       = header_size;

            buf_size_t const current_objects_offset =
                current_slots_offset + current_slots_extent;

            buf_size_t const current_objects_size =
                NS(Buffer_get_section_size)( buffer, OBJECTS_ID );

            buf_size_t const new_objects_offset =
                new_slots_offset + requ_slots_extent;

            buf_size_t const current_dataptrs_offset =
                current_objects_offset + current_objs_extent;

            buf_size_t const current_dataptrs_size =
                NS(Buffer_get_section_size)( buffer, DATAPTRS_ID );

            buf_size_t const new_dataptrs_offset =
                new_objects_offset + requ_objs_extent;

            buf_size_t const current_garbage_offset =
                current_dataptrs_offset + current_dataptrs_extent;

            buf_size_t const current_garbage_size =
                NS(Buffer_get_section_size)( buffer, GARBAGE_ID );

            buf_size_t const new_garbage_offset =
                new_dataptrs_offset + requ_dataptrs_extent;

            ptr_to_raw_t begin = ( ptr_to_raw_t )( uintptr_t
                    )NS(Buffer_get_data_begin_addr)( buffer );

            addr_offsets[ 1 ] +=  ( new_slots_offset >= current_slots_offset )
                ?  ( addr_diff_t )( new_slots_offset  - current_slots_offset )
                : -( addr_diff_t )( current_slots_extent -  new_slots_offset );

            addr_offsets[ 2 ] +=  ( new_objects_offset >= current_objs_extent )
                ?  ( addr_diff_t )( new_objects_offset  - current_objs_extent )
                : -( addr_diff_t )( current_objs_extent -  new_objects_offset );

            addr_offsets[ 3 ] +=  ( new_dataptrs_offset >= current_dataptrs_offset )
                ?  ( addr_diff_t )( new_dataptrs_offset  - current_dataptrs_offset )
                : -( addr_diff_t )( current_dataptrs_offset -  new_dataptrs_offset );

            addr_offsets[ 4 ] +=  ( new_garbage_offset >=  current_garbage_offset  )
                ?  ( addr_diff_t )( new_garbage_offset -   current_garbage_offset  )
                : -( addr_diff_t )( current_garbage_offset -   new_garbage_offset  );

            SIXTRL_ASSERT( ptr_header != SIXTRL_NULLPTR );

            ptr_header[ 0 ] += addr_offsets[ 0 ];
            ptr_header[ 1 ]  = new_garbage_offset + requ_garbage_extent;
            ptr_header[ 2 ]  = header_size;
            ptr_header[ 3 ] += addr_offsets[ 1 ];
            ptr_header[ 4 ] += addr_offsets[ 2 ];
            ptr_header[ 5 ] += addr_offsets[ 3 ];
            ptr_header[ 6 ] += addr_offsets[ 4 ];
            ptr_header[ 7 ]  = ( address_t )0u;

            if( current_garbage_size >= addr_size )
            {
                buf_size_t const offset = current_garbage_size - addr_size;

                ptr_to_addr_t dest_it = ( ptr_to_addr_t )(
                    begin + ( new_garbage_offset + offset ) );

                ptr_to_const_addr_t src_it  = ( ptr_to_const_addr_t )(
                    begin + ( current_garbage_offset + offset ) );

                ptr_to_const_addr_t src_end = ( ptr_to_const_addr_t )(
                    begin + ( current_garbage_offset - addr_size ) );

                for( ; src_it != src_end ; --src_it, --dest_it )
                {
                    *dest_it = *src_it;
                }
            }

            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            if( current_dataptrs_size >= addr_size )
            {
                buf_size_t const offset = current_dataptrs_size - addr_size;

                ptr_to_addr_t dest_it = ( ptr_to_addr_t )(
                    begin + ( new_dataptrs_offset + offset ) );

                ptr_to_const_addr_t src_it  = ( ptr_to_const_addr_t )(
                    begin + ( current_dataptrs_offset + offset ) );

                ptr_to_const_addr_t src_end = ( ptr_to_const_addr_t )(
                    begin + ( current_dataptrs_offset - addr_size ) );

                for( ; src_it != src_end ; --src_it, --dest_it )
                {
                    *dest_it = *src_it;
                }
            }

            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            if( current_objects_size >= addr_size )
            {
                buf_size_t const offset = current_objects_size - addr_size;

                ptr_to_addr_t dest_it = ( ptr_to_addr_t )(
                    begin + ( new_objects_offset + offset ) );

                ptr_to_const_addr_t src_it  = ( ptr_to_const_addr_t )(
                    begin + ( current_objects_offset + offset ) );

                ptr_to_const_addr_t src_end = ( ptr_to_const_addr_t )(
                    begin + ( current_objects_offset - addr_size ) );

                for( ; src_it != src_end ; --src_it, --dest_it )
                {
                    *dest_it = *src_it;
                }
            }

            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            if( current_slots_size >= addr_size )
            {
                buf_size_t const offset = current_slots_size - addr_size;

                ptr_to_addr_t dest_it = ( ptr_to_addr_t )(
                    begin + ( new_slots_offset + offset ) );

                ptr_to_const_addr_t src_it  = ( ptr_to_const_addr_t )(
                    begin + ( current_slots_offset + offset ) );

                ptr_to_const_addr_t src_end = ( ptr_to_const_addr_t )(
                    begin + ( current_slots_offset - addr_size ) );

                for( ; src_it != src_end ; --src_it, --dest_it )
                {
                    *dest_it = *src_it;
                }
            }

            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            if( addr_offsets[ 1 ] != ( addr_diff_t )0u )
            {
                success  = NS(Buffer_remap_section_slots_generic)(
                    begin, 0, 0, 0, &addr_offsets[ 0 ], slot_size );

                success |= NS(Buffer_remap_section_objects_generic)(
                    begin, 0, 0, 0, &addr_offsets[ 0 ], slot_size );

                success |= NS(Buffer_remap_section_dataptrs_generic)(
                    begin, 0, 0, 0, &addr_offsets[ 0 ], slot_size );

                success |= NS(Buffer_remap_section_garbage_generic)(
                    begin, 0, 0, 0, &addr_offsets[ 0 ], slot_size );
            }
            else
            {
                success = 0;
            }

            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            if( success == 0 )
            {
                buffer->data_size = ptr_header[ 1 ];

                if( requ_slots_extent != current_slots_extent )
                {
                    NS(Buffer_set_section_extent_generic)(
                        buffer, 3u, requ_slots_extent );
                }

                if( requ_objs_extent != current_objs_extent )
                {
                    NS(Buffer_set_section_extent_generic)(
                        buffer, 4u, requ_objs_extent );
                }

                if( requ_dataptrs_extent != current_dataptrs_extent )
                {
                    NS(Buffer_set_section_extent_generic)(
                        buffer, 5u, requ_dataptrs_extent );
                }

                if( requ_garbage_extent != current_garbage_extent )
                {
                    NS(Buffer_set_section_extent_generic)(
                        buffer, 6u, requ_garbage_extent );
                }
            }
        }
    }
    else
    {
        success = 0;
    }

    return success;
}

/* ========================================================================= */

SIXTRL_INLINE void NS(Buffer_free_generic)(
    NS(Buffer)* SIXTRL_RESTRICT buffer )
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
            typedef NS(MemPool)         mem_pool_t;
            typedef mem_pool_t*         ptr_to_mem_pool_t;

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

SIXTRL_INLINE NS(Object)* NS(Buffer_add_object)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    const void *const SIXTRL_RESTRICT ptr_to_object,
    NS(buffer_size_t)        const object_size,
    NS(object_type_id_t)     const type_id,
    NS(buffer_size_t)        const num_obj_dataptr,
    const NS(buffer_size_t) *const SIXTRL_RESTRICT obj_dataptr_offsets,
    const NS(buffer_size_t) *const SIXTRL_RESTRICT obj_dataptr_sizes,
    const NS(buffer_size_t) *const SIXTRL_RESTRICT obj_dataptr_counts )
{
    typedef NS(Object)              object_t;
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(buffer_addr_t)       address_t;
    typedef unsigned char*          ptr_to_raw_t;
    typedef address_t*              ptr_to_addr_t;
    typedef object_t*               ptr_to_object_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR address_t  const ZERO_ADDR = ( address_t  )0u;

    object_t* result_object = SIXTRL_NULLPTR;
    int success = -1;

    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    if( ( slot_size     != ZERO_SIZE      ) &&
        ( object_size   >  ZERO_SIZE      ) &&
        ( ptr_to_object != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_uses_datastore)( buffer ) ) &&
        ( NS(Buffer_allow_append_objects)( buffer ) ) )
    {
        SIXTRL_STATIC_VAR buf_size_t const SLOTS_ID     = ( buf_size_t )3u;
        SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID   = ( buf_size_t )4u;
        SIXTRL_STATIC_VAR buf_size_t const DATAPTRS_ID  = ( buf_size_t )5u;
        SIXTRL_STATIC_VAR buf_size_t const GARBAGE_ID   = ( buf_size_t )6u;

        buf_size_t const current_num_slots =
            NS(Buffer_get_section_num_entities_generic)( buffer, SLOTS_ID );

        buf_size_t max_num_slots =
            NS(Buffer_get_section_max_num_entities)(
                buffer, SLOTS_ID );

        buf_size_t const current_num_objects =
            NS(Buffer_get_section_num_entities_generic)( buffer, OBJECTS_ID );

        buf_size_t max_num_objects =
            NS(Buffer_get_section_max_num_entities)(
                buffer, OBJECTS_ID );

        buf_size_t const current_num_dataptrs =
            NS(Buffer_get_section_num_entities_generic)( buffer, DATAPTRS_ID );

        buf_size_t max_num_dataptrs =
            NS(Buffer_get_section_max_num_entities)(
                buffer, DATAPTRS_ID );

        buf_size_t const current_num_garbage_elems =
            NS(Buffer_get_section_num_entities_generic)( buffer, GARBAGE_ID );

        buf_size_t const requ_num_objects  =
            current_num_objects  + 1u;

        buf_size_t const requ_num_dataptrs =
            current_num_dataptrs + num_obj_dataptr;

        buf_size_t const obj_handle_size =
            NS(Buffer_get_slot_based_length)( object_size, slot_size );

        buf_size_t requ_num_slots            = current_num_slots;
        buf_size_t additional_num_slots      = ZERO_SIZE;
        buf_size_t additional_num_slots_size = obj_handle_size;

        if( num_obj_dataptr > ZERO_SIZE )
        {
            buf_size_t ii = ZERO_SIZE;

            SIXTRL_ASSERT( obj_dataptr_offsets != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( obj_dataptr_sizes   != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( obj_dataptr_counts  != SIXTRL_NULLPTR );

            for( ; ii < num_obj_dataptr ; ++ii )
            {
                buf_size_t const elem_size = obj_dataptr_sizes[ ii ];
                buf_size_t const attr_cnt  = obj_dataptr_counts[ ii ];
                buf_size_t const attr_size = NS(Buffer_get_slot_based_length)(
                    elem_size * attr_cnt, slot_size );

                SIXTRL_ASSERT( ( obj_dataptr_offsets[ ii ] % slot_size ) == 0u );
                SIXTRL_ASSERT(   obj_dataptr_offsets[ ii ] < object_size );

                SIXTRL_ASSERT( elem_size > ZERO_SIZE );
                SIXTRL_ASSERT( attr_cnt  > ZERO_SIZE );
                SIXTRL_ASSERT( ( attr_size % slot_size ) == 0u );

                additional_num_slots_size += attr_size;
            }

            if( ( additional_num_slots_size % slot_size ) != 0u )
            {
                additional_num_slots_size += slot_size - (
                    additional_num_slots_size % slot_size );
            }

            additional_num_slots += additional_num_slots_size / slot_size;
            requ_num_slots       += additional_num_slots;
        }

        if( ( requ_num_objects  > max_num_objects  ) ||
            ( requ_num_slots    > max_num_slots    ) ||
            ( requ_num_dataptrs > max_num_dataptrs ) )
        {
            success = NS(Buffer_reserve_generic)( buffer,
                requ_num_objects, requ_num_slots,
                    requ_num_dataptrs, current_num_garbage_elems );
        }
        else
        {
            success = 0;
        }

        SIXTRL_ASSERT(
            ( success != 0 ) ||
            ( ( requ_num_objects <= NS(Buffer_get_section_max_num_entities)(
                    buffer, OBJECTS_ID ) ) &&
              ( requ_num_slots <= NS(Buffer_get_section_max_num_entities)(
                    buffer, SLOTS_ID ) ) &&
              ( requ_num_dataptrs <= NS(Buffer_get_section_max_num_entities)(
                    buffer, DATAPTRS_ID ) ) &&
              ( current_num_garbage_elems <=
                NS(Buffer_get_section_max_num_entities)(
                    buffer, GARBAGE_ID ) ) ) );

        /* ----------------------------------------------------------------- */

        if( success == 0 )
        {

            buf_size_t const current_slots_size =
                NS(Buffer_get_section_size)( buffer, SLOTS_ID );

            ptr_to_raw_t dest_slots = ( ( ptr_to_raw_t
                )NS(Buffer_get_ptr_to_section)( buffer, SLOTS_ID ) ) +
                    current_slots_size;

            ptr_to_raw_t stored_obj_begin = dest_slots;

            if( object_size > ZERO_SIZE )
            {
                ptr_to_raw_t dest_obj_info = ( ( ptr_to_raw_t
                    )NS(Buffer_get_ptr_to_section)(
                        buffer, OBJECTS_ID ) );

                object_t obj_info;
                NS(Object_preset)( &obj_info );

                if( stored_obj_begin != SIXTRL_NULLPTR )
                {
                    SIXTRL_ASSERT( ( ( ( uintptr_t )stored_obj_begin ) %
                        slot_size ) == 0 );

                    SIXTRACKLIB_COPY_VALUES( unsigned char, stored_obj_begin,
                        ptr_to_object, object_size );
                }
                else
                {
                    success = -1;
                }

                NS(Object_set_type_id)( &obj_info, type_id );
                NS(Object_set_begin_ptr)( &obj_info, stored_obj_begin );
                NS(Object_set_size)( &obj_info, additional_num_slots_size );

                if( ( success == 0 ) && ( dest_obj_info != SIXTRL_NULLPTR ) )
                {
                    dest_obj_info = dest_obj_info +
                        NS(Buffer_get_section_size)(
                            buffer, OBJECTS_ID );

                    SIXTRL_ASSERT( ( ( ( uintptr_t )dest_obj_info )
                        % slot_size ) == ZERO_SIZE );

                    SIXTRACKLIB_COPY_VALUES( unsigned char,
                        dest_obj_info, &obj_info, sizeof( object_t ) );

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
                    NS(Buffer_get_section_size)( buffer, DATAPTRS_ID );

                buf_size_t ii = ZERO_SIZE;

                ptr_to_raw_t dest_dataptrs = ( ( ptr_to_raw_t
                    )NS(Buffer_get_ptr_to_section)(
                        buffer, DATAPTRS_ID ) ) + current_dataptrs_size;

                ptr_to_addr_t out_it = ( ptr_to_addr_t )dest_dataptrs;

                SIXTRL_ASSERT( ( ( ( uintptr_t )dest_dataptrs )
                    % slot_size ) == ZERO_SIZE );

                for( ; ii < num_obj_dataptr ; ++ii, ++out_it )
                {
                    buf_size_t const offset    = obj_dataptr_offsets[ ii ];
                    buf_size_t const attr_cnt  = obj_dataptr_counts[ ii ];
                    buf_size_t const elem_size = obj_dataptr_sizes[ ii ];
                    buf_size_t const attr_size = attr_cnt * elem_size;

                    buf_size_t const attr_extent =
                        NS(Buffer_get_slot_based_length)( attr_size, slot_size );

                    ptr_to_raw_t ptr_attr_slot = stored_obj_begin + offset;

                    address_t const attr_slot_addr =
                        ( address_t )( uintptr_t )ptr_attr_slot;

                    address_t const source_addr = ( attr_slot_addr != ZERO_ADDR )
                        ? *( ( ptr_to_addr_t )ptr_attr_slot ) : ZERO_ADDR;

                    ptr_to_raw_t ptr_attr_data_begin_dest = dest_slots;

                    address_t const attr_data_begin_dest_addr =
                        ( address_t )( uintptr_t )ptr_attr_data_begin_dest;

                    if( source_addr != ZERO_ADDR )
                    {
                        SIXTRACKLIB_COPY_VALUES( unsigned char,
                            ptr_attr_data_begin_dest, ( ptr_to_raw_t )(
                                uintptr_t )source_addr,  attr_size );
                    }
                    else
                    {
                        SIXTRACKLIB_SET_VALUES( unsigned char,
                            ptr_attr_data_begin_dest, attr_extent, ( int )0u );
                    }

                    *( ( ptr_to_addr_t )ptr_attr_slot ) =
                        attr_data_begin_dest_addr;

                    *out_it = attr_slot_addr;

                    dest_slots = dest_slots + attr_extent;
                }

                SIXTRL_ASSERT( requ_num_dataptrs ==
                    ( current_num_dataptrs + num_obj_dataptr ) );

                NS(Buffer_set_section_num_entities_generic)(
                    buffer, DATAPTRS_ID, requ_num_dataptrs );
            }

            if( success == 0 )
            {
                NS(Buffer_set_section_num_entities_generic)(
                    buffer, SLOTS_ID, requ_num_slots   );

                NS(Buffer_set_section_num_entities_generic)(
                    buffer, OBJECTS_ID, requ_num_objects );

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
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    unsigned char* SIXTRL_RESTRICT data_buffer_begin )
{
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(buffer_addr_t)       address_t;
    typedef NS(buffer_addr_diff_t)  addr_diff_t;
    typedef unsigned char*          ptr_to_raw_t;
    typedef address_t*              ptr_to_addr_t;

    int success = -1;

    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    if( ( buffer != SIXTRL_NULLPTR ) && ( slot_size > 0u ) &&
        ( data_buffer_begin != SIXTRL_NULLPTR ) &&
        ( ( ( ( uintptr_t )data_buffer_begin ) % slot_size ) == 0u ) )
    {
        SIXTRL_STATIC_VAR address_t  const ZERO_ADDR = ( address_t  )0u;
        SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

        ptr_to_addr_t ptr_header = ( ptr_to_addr_t )data_buffer_begin;
        address_t const stored_base_addr = ptr_header[ 0 ];
        buf_size_t const in_size = ( buf_size_t )ptr_header[ 1 ];

        ptr_to_raw_t begin  = SIXTRL_NULLPTR;
        address_t base_addr = ZERO_ADDR;

        if( ( NS(Buffer_uses_datastore)( buffer ) ) &&
            ( NS(Buffer_allow_modify_datastore_contents)( buffer ) ) )
        {
            buf_size_t const datastore_capacity =
                NS(Buffer_get_capacity)( buffer );

            if( in_size <= datastore_capacity )
            {
                begin = ( ptr_to_raw_t )(
                    uintptr_t )NS(Buffer_get_data_begin_addr)( buffer );

                success = 0;
            }
            else if(
                ( NS(Buffer_owns_datastore)( buffer ) ) &&
                ( NS(Buffer_allow_resize)( buffer ) ) )
            {
                success = NS(Buffer_clear_and_resize_datastore_generic)(
                    buffer, in_size );

                if( success == 0 )
                {
                    begin = ( ptr_to_raw_t )(
                        uintptr_t )NS(Buffer_get_data_begin_addr)( buffer );
                }
            }

            if( success == 0 )
            {
                SIXTRL_ASSERT( begin != SIXTRL_NULLPTR );
                SIXTRACKLIB_COPY_VALUES(
                    unsigned char, begin, data_buffer_begin, in_size );
            }
        }
        else
        {
            begin   = data_buffer_begin;
            success = 0;
        }

        base_addr = ( address_t )( uintptr_t )begin;

        if( success != 0 )
        {
            return success;
        }

        SIXTRL_ASSERT( success == 0 );
        SIXTRL_ASSERT( base_addr != ZERO_ADDR );

        if( stored_base_addr == base_addr )
        {
            buffer->data_addr   = base_addr;
            buffer->data_size   = in_size;
            buffer->header_size = ptr_header[ 2 ];

            buffer->object_addr = ( address_t )( uintptr_t
                )NS(Buffer_get_ptr_to_section_data)( buffer, 4u );

            buffer->num_objects = ( buf_size_t
                )NS(Buffer_get_section_num_entities_generic)( buffer, 4u );

            buffer->datastore_flags |=
                    NS(BUFFER_USES_DATASTORE) |
                    NS(BUFFER_DATASTORE_ALLOW_REMAPPING);

            buffer->datastore_addr = base_addr;
        }
        else if( ( stored_base_addr != ZERO_ADDR ) &&
                 ( stored_base_addr != base_addr ) )
        {
            addr_diff_t addr_offsets[ 5 ] = { 0, 0, 0, 0, 0 };

            NS(Object)* ptr_objects_begin = SIXTRL_NULLPTR;
            buf_size_t num_objects        = ZERO_SIZE;
            buf_size_t objects_capacity   = ZERO_SIZE;

            success  = NS(Buffer_remap_get_addr_offset_generic)(
                &addr_offsets[ 0 ], begin, slot_size );

            if( success == 0 )
            {
                addr_offsets[ 1 ] = addr_offsets[ 0 ];
                addr_offsets[ 2 ] = addr_offsets[ 0 ];
                addr_offsets[ 3 ] = addr_offsets[ 0 ];
                addr_offsets[ 4 ] = addr_offsets[ 0 ];
            }

            success  = NS(Buffer_remap_header_generic)(
                begin, &addr_offsets[ 0 ], slot_size );

            success |= NS(Buffer_remap_section_slots_generic)(
                begin, 0, 0, 0, &addr_offsets[ 0 ], slot_size );

            success |= NS(Buffer_remap_section_objects_generic)(
                begin, &ptr_objects_begin, &objects_capacity,
                    &num_objects, &addr_offsets[ 0 ], slot_size );

            success |= NS(Buffer_remap_section_dataptrs_generic)(
                begin, 0, 0, 0, &addr_offsets[ 0 ], slot_size );

            success |= NS(Buffer_remap_section_garbage_generic)(
                begin, 0, 0, 0, &addr_offsets[ 0 ], slot_size );

            if( success == 0 )
            {
                buffer->data_addr   = base_addr;
                buffer->data_size   = in_size;
                buffer->header_size = ( buf_size_t )ptr_header[ 2 ];
                buffer->num_objects = num_objects;

                buffer->object_addr =
                    ( address_t )( uintptr_t )ptr_objects_begin;

                buffer->datastore_flags |=
                    NS(BUFFER_USES_DATASTORE) |
                    NS(BUFFER_DATASTORE_ALLOW_REMAPPING);

                buffer->datastore_addr = base_addr;
            }
        }
        else if( stored_base_addr == ZERO_ADDR )
        {
            buffer->data_addr = base_addr;
            buffer->data_size = in_size;

            success = NS(Buffer_reset_generic)( buffer );
        }
    }

    return success;
}

SIXTRL_INLINE int NS(Buffer_init_on_flat_memory)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    unsigned char* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const buffer_capacity )
{
    return NS(Buffer_init_on_flat_memory_detailed)(
        buffer, data_buffer_begin, buffer_capacity, 0u, 0u, 0u, 0u );
}

SIXTRL_INLINE int NS(Buffer_init_on_flat_memory_detailed)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    unsigned char* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const buffer_capacity,
    NS(buffer_size_t) const initial_max_num_objects,
    NS(buffer_size_t) const initial_max_num_slots,
    NS(buffer_size_t) const initial_max_num_dataptrs,
    NS(buffer_size_t) const initial_max_num_garbage_elems )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) address_t;
    typedef address_t*        ptr_to_addr_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID   = ( buf_size_t )4u;
    SIXTRL_STATIC_VAR buf_size_t const DATAPTRS_ID  = ( buf_size_t )5u;
    SIXTRL_STATIC_VAR buf_size_t const GARBAGE_ID   = ( buf_size_t )6u;

    NS(Buffer_preset)( buffer );
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    buf_size_t const addr_size = NS(Buffer_get_slot_based_length)(
        sizeof( address_t ), slot_size );

    buf_size_t const header_size = NS(Buffer_get_header_size)( buffer );

    buf_size_t const obj_info_size =
        NS(Buffer_get_section_entity_size)( buffer, OBJECTS_ID );

    buf_size_t const dataptrs_size =
        NS(Buffer_get_section_entity_size)( buffer, DATAPTRS_ID );

    buf_size_t const garbage_elem_size =
        NS(Buffer_get_section_entity_size)( buffer, GARBAGE_ID );

    buf_size_t const section_hd_size = 2u * addr_size;

    buf_size_t const min_required_capacity = header_size +
        section_hd_size + initial_max_num_objects  * obj_info_size +
        section_hd_size + initial_max_num_slots    * slot_size +
        section_hd_size + initial_max_num_dataptrs * dataptrs_size +
        section_hd_size + initial_max_num_garbage_elems * garbage_elem_size;

    if( ( slot_size > 0u ) &&
        ( buffer != SIXTRL_NULLPTR ) &&
        ( data_buffer_begin != SIXTRL_NULLPTR ) &&
        ( ( ( ( uintptr_t )data_buffer_begin ) % slot_size ) == 0u ) &&
        ( buffer_capacity >= min_required_capacity ) )
    {
        ptr_to_addr_t ptr_header = ( ptr_to_addr_t )data_buffer_begin;
        ptr_header[ 0 ] = ( address_t )( uintptr_t )data_buffer_begin;
        ptr_header[ 1 ] = ( buf_size_t )0u;
        ptr_header[ 2 ] = header_size;

        buffer->data_addr = ptr_header[ 0 ];
        buffer->data_size = ( buf_size_t )0u;

        buffer->data_capacity   = buffer_capacity;
        buffer->datastore_addr  = buffer->data_addr;
        buffer->datastore_flags =
            NS(BUFFER_DATASTORE_ALLOW_APPENDS) |
            NS(BUFFER_USES_DATASTORE) |
            NS(BUFFER_DATASTORE_ALLOW_CLEAR) |
            NS(BUFFER_DATASTORE_ALLOW_DELETES) |
            NS(BUFFER_DATASTORE_ALLOW_REMAPPING);

        success = NS(Buffer_reset_detailed_generic)( buffer, initial_max_num_objects,
            initial_max_num_slots, initial_max_num_dataptrs,
                initial_max_num_garbage_elems );
    }

    return success;
}

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRKL_COMMON_IMPL_BUFFER_GENERIC_H__ */
/*end: sixtracklib/sixtracklib/common/impl/buffer_generic.h */
