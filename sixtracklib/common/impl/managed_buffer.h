#ifndef SIXTRL_COMMON_IMPL_BUFFER_MEM_H__
#define SIXTRL_COMMON_IMPL_BUFFER_MEM_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <limits.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(ManagedBuffer_calculate_section_size)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const num_entities,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(ManagedBuffer_get_section_offset)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC void NS(ManagedBuffer_set_section_num_entities)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id,
    NS(buffer_size_t) const num_entities,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC void NS(ManagedBuffer_set_section_max_size)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id,
    NS(buffer_size_t) const max_section_size,
    NS(buffer_size_t) const slot_size );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(ManagedBuffer_calculate_buffer_length)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_ranges,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC bool NS(ManagedBuffer_can_reserve)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_buffer_len,
    NS(buffer_size_t) const max_buffer_length,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_ranges,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(ManagedBuffer_reserve)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_new_buffer_length,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_ranges,
    NS(buffer_size_t) const max_data_buffer_length,
    NS(buffer_size_t) const slot_size );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC int NS(ManagedBuffer_init)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_buffer_length,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_ranges,
    NS(buffer_size_t) const max_data_buffer_length,
    NS(buffer_size_t) const slot_size );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC void NS(ManagedBuffer_clear)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    bool const set_data_to_zero, NS(buffer_size_t) const slot_size );

/* ========================================================================= */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#if defined( __cplusplus )

namespace SIXTRL_NAMESPACE
{

}

#endif /* defined( __cplusplus ) */

/* ========================================================================= *
 * ======== INLINE IMPLEMENTATION                                            *
 * ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/impl/managed_buffer_minimal.h"
    #include "sixtracklib/common/impl/managed_buffer_remap.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(buffer_size_t) NS(ManagedBuffer_calculate_section_size)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const num_entities,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    buf_size_t calculated_size = ZERO_SIZE;

    if( slot_size > ZERO_SIZE )
    {
        buf_size_t const entity_size =
            NS(ManagedBuffer_get_section_entity_size)(
                begin, header_section_id, slot_size );

        buf_size_t const section_header_length =
            NS(ManagedBuffer_get_section_header_length)( begin, slot_size );

        SIXTRL_ASSERT( entity_size           > ZERO_SIZE );
        SIXTRL_ASSERT( section_header_length > ZERO_SIZE );

        SIXTRL_ASSERT( ( begin == SIXTRL_NULLPTR ) ||
            ( ( ( ( uintptr_t )begin ) %  slot_size )  == ZERO_SIZE ) );

        calculated_size = num_entities * entity_size + section_header_length;
    }

    return calculated_size;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(buffer_size_t) NS(ManagedBuffer_get_section_offset)(
    SIXTRL_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id, NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef NS(buffer_addr_t)                       address_t;
    typedef SIXTRL_DATAPTR_DEC address_t const*     ptr_to_addr_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    buf_size_t section_offset = ZERO_SIZE;

    if( ( begin != SIXTRL_NULLPTR ) && ( slot_size > ZERO_SIZE ) )
    {
        ptr_to_addr_t ptr_header = ( ptr_to_addr_t )begin;

        #if !defined( NDEBUG )
        buf_size_t const addr_size = NS(ManagedBuffer_get_slot_based_length)(
            sizeof( address_t ), slot_size );

        buf_size_t const header_length  = ptr_header[ 2 ];

        SIXTRL_ASSERT( ptr_header[ 0 ] != ( address_t )0u );
        SIXTRL_ASSERT(   addr_size > ZERO_SIZE );
        SIXTRL_ASSERT( ( addr_size * section_id ) < header_length  );
        SIXTRL_ASSERT( ( section_id >= 3u ) && ( section_id <= 6u ) );
        SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size ) == 0u );

        #endif /* !defined( NDEBUG ) */

        section_offset = (    ptr_header[ section_id ] >= ptr_header[ 0 ] )
            ? ( buf_size_t )( ptr_header[ section_id ] -  ptr_header[ 0 ] )
            : ZERO_SIZE;
    }

    return section_offset;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(ManagedBuffer_set_section_num_entities)(
    SIXTRL_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id, NS(buffer_size_t) const num_entities,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)             buf_size_t;
    typedef NS(buffer_addr_t)             address_t;
    typedef SIXTRL_DATAPTR_DEC address_t* ptr_to_addr_t;

    ptr_to_addr_t ptr_to_section = ( ptr_to_addr_t
        )NS(ManagedBuffer_get_ptr_to_section)( begin, section_id, slot_size );

    SIXTRL_ASSERT( NS(ManagedBuffer_get_section_entity_size)(
        begin, section_id, slot_size ) > ( buf_size_t )0u );

    if( ptr_to_section != SIXTRL_NULLPTR )
    {
        ptr_to_section[ 1 ] = num_entities;
    }

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(ManagedBuffer_set_section_max_size)(
    SIXTRL_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id,
    NS(buffer_size_t) const max_section_size,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)             buf_size_t;
    typedef NS(buffer_addr_t)             address_t;
    typedef SIXTRL_DATAPTR_DEC address_t* ptr_to_addr_t;

    ptr_to_addr_t ptr_to_section = ( ptr_to_addr_t
        )NS(ManagedBuffer_get_ptr_to_section)( begin, section_id, slot_size );

    SIXTRL_ASSERT( NS(ManagedBuffer_get_section_entity_size)(
        begin, section_id, slot_size ) > ( buf_size_t )0u );

    if( ptr_to_section != SIXTRL_NULLPTR )
    {
        ptr_to_section[ 0 ] = max_section_size;
    }

    return;
}

/* ========================================================================= */

SIXTRL_INLINE NS(buffer_size_t) NS(ManagedBuffer_calculate_buffer_length)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_ranges,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t const SLOTS_ID     = ( buf_size_t )3u;
    SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID   = ( buf_size_t )4u;
    SIXTRL_STATIC_VAR buf_size_t const DATAPTRS_ID  = ( buf_size_t )5u;
    SIXTRL_STATIC_VAR buf_size_t const GARBAGE_ID   = ( buf_size_t )6u;

    buf_size_t const header_length =
        NS(ManagedBuffer_get_header_length)( begin, slot_size );

    buf_size_t const requ_slots_extent =
        NS(ManagedBuffer_calculate_section_size)(
            begin, SLOTS_ID, max_num_slots, slot_size );

    buf_size_t const requ_objs_extent =
        NS(ManagedBuffer_calculate_section_size)(
            begin, OBJECTS_ID, max_num_objects, slot_size );

    buf_size_t const requ_dataptrs_extent =
        NS(ManagedBuffer_calculate_section_size)(
            begin, DATAPTRS_ID, max_num_dataptrs, slot_size );

    buf_size_t const requ_garbage_extent =
        NS(ManagedBuffer_calculate_section_size)(
            begin, GARBAGE_ID, max_num_garbage_ranges, slot_size );

    return header_length + requ_slots_extent +
        requ_objs_extent + requ_dataptrs_extent + requ_garbage_extent;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE bool NS(ManagedBuffer_can_reserve)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_buffer_len,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_ranges,
    NS(buffer_size_t) const max_data_buffer_length,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t const SLOTS_ID     = ( buf_size_t )3u;
    SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID   = ( buf_size_t )4u;
    SIXTRL_STATIC_VAR buf_size_t const DATAPTRS_ID  = ( buf_size_t )5u;
    SIXTRL_STATIC_VAR buf_size_t const GARBAGE_ID   = ( buf_size_t )6u;

    buf_size_t const requ_buffer_length =
        NS(ManagedBuffer_calculate_buffer_length)(
            begin, max_num_objects, max_num_slots, max_num_dataptrs,
                max_num_garbage_ranges, slot_size );

    if(  ptr_requ_buffer_len != SIXTRL_NULLPTR )
    {
        *ptr_requ_buffer_len  = requ_buffer_length;
    }

    return (
        ( requ_buffer_length <= max_data_buffer_length ) &&
        ( NS(ManagedBuffer_get_section_num_entities)(
            begin, SLOTS_ID, slot_size )    <= max_num_slots ) &&
        ( NS(ManagedBuffer_get_section_num_entities)(
            begin, OBJECTS_ID, slot_size )  <= max_num_objects ) &&
        ( NS(ManagedBuffer_get_section_num_entities)(
            begin, DATAPTRS_ID, slot_size ) <= max_num_dataptrs ) &&
        ( NS(ManagedBuffer_get_section_num_entities)(
            begin, GARBAGE_ID, slot_size )  <= max_num_garbage_ranges ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(ManagedBuffer_reserve)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_new_buffer_length,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_ranges,
    NS(buffer_size_t) const max_data_buffer_length,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef unsigned char                           raw_t;
    typedef SIXTRL_ARGPTR_DEC unsigned char*        ptr_to_raw_t;
    typedef SIXTRL_ARGPTR_DEC unsigned char const*  ptr_to_const_raw_t;

    typedef NS(buffer_addr_t)                       address_t;
    typedef SIXTRL_ARGPTR_DEC address_t*            ptr_to_addr_t;

    SIXTRL_STATIC_VAR buf_size_t const SLOTS_ID     = ( buf_size_t )3u;
    SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID   = ( buf_size_t )4u;
    SIXTRL_STATIC_VAR buf_size_t const DATAPTRS_ID  = ( buf_size_t )5u;
    SIXTRL_STATIC_VAR buf_size_t const GARBAGE_ID   = ( buf_size_t )6u;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE    = ( buf_size_t )0u;

    buf_size_t const cur_max_num_slots =
        NS(ManagedBuffer_get_section_max_num_entities)( begin, SLOTS_ID, slot_size );

    buf_size_t const cur_num_slots =
        NS(ManagedBuffer_get_section_num_entities)( begin, SLOTS_ID, slot_size );


    buf_size_t const cur_max_num_objects =
        NS(ManagedBuffer_get_section_max_num_entities)(
            begin, OBJECTS_ID, slot_size );

    buf_size_t const cur_num_objects =
        NS(ManagedBuffer_get_section_num_entities)(
            begin, OBJECTS_ID, slot_size );


    buf_size_t const cur_max_num_dataptrs =
        NS(ManagedBuffer_get_section_max_num_entities)(
            begin, DATAPTRS_ID, slot_size );

    buf_size_t const cur_num_dataptrs =
        NS(ManagedBuffer_get_section_num_entities)(
            begin, DATAPTRS_ID, slot_size );


    buf_size_t const cur_max_num_garbage_ranges =
        NS(ManagedBuffer_get_section_max_num_entities)(
            begin, GARBAGE_ID, slot_size );

    buf_size_t const cur_num_garbage_ranges =
        NS(ManagedBuffer_get_section_num_entities)(
            begin, GARBAGE_ID, slot_size );


    buf_size_t const current_buffer_length =
        NS(ManagedBuffer_get_buffer_length)( begin, slot_size );

    buf_size_t new_buffer_length = ZERO_SIZE;

    int success = -1;

    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)( begin, slot_size ) );
    SIXTRL_ASSERT( cur_max_num_objects        >= cur_num_objects        );
    SIXTRL_ASSERT( cur_max_num_slots          >= cur_num_slots          );
    SIXTRL_ASSERT( cur_max_num_dataptrs       >= cur_num_dataptrs       );
    SIXTRL_ASSERT( cur_max_num_garbage_ranges >= cur_num_garbage_ranges );

    success = (
        ( begin                    != SIXTRL_NULLPTR         ) &&
        ( slot_size                >  ZERO_SIZE              ) &&
        ( current_buffer_length    >  ZERO_SIZE              ) &&
        ( current_buffer_length    <= max_data_buffer_length ) &&
        ( cur_num_objects          <= max_num_objects        ) &&
        ( cur_num_slots            <= max_num_slots          ) &&
        ( cur_num_dataptrs         <= max_num_dataptrs       ) &&
        ( cur_num_garbage_ranges   <= max_num_garbage_ranges ) ) ? 0 : -1;

    if( ( success == 0 ) &&
        ( ( max_num_objects        != cur_max_num_objects   ) ||
          ( max_num_slots          != cur_max_num_slots     ) ||
          ( max_num_dataptrs       != cur_max_num_dataptrs  ) ||
          ( max_num_garbage_ranges != cur_max_num_garbage_ranges ) ) )
    {
        buf_size_t const header_length =
            NS(ManagedBuffer_get_header_length)( begin, slot_size );

        buf_size_t const requ_objs_extent =
            NS(ManagedBuffer_calculate_section_size)(
                begin, OBJECTS_ID, max_num_objects, slot_size );

        buf_size_t const requ_slots_extent =
            NS(ManagedBuffer_calculate_section_size)(
                begin, SLOTS_ID, max_num_slots, slot_size );

        buf_size_t const requ_dataptrs_extent =
            NS(ManagedBuffer_calculate_section_size)(
                begin, DATAPTRS_ID, max_num_dataptrs, slot_size );

        buf_size_t const requ_garbage_extent =
            NS(ManagedBuffer_calculate_section_size)(
                begin, GARBAGE_ID, max_num_garbage_ranges, slot_size );

        new_buffer_length = header_length + requ_slots_extent +
            requ_objs_extent + requ_dataptrs_extent + requ_garbage_extent;

        success = -1;

        if( new_buffer_length <= max_data_buffer_length )
        {
            buf_size_t const cur_slots_offset =
                NS(ManagedBuffer_get_section_offset)(
                    begin, SLOTS_ID, slot_size );

            buf_size_t const cur_slots_size =
                NS(ManagedBuffer_get_section_size)(
                    begin, SLOTS_ID, slot_size );

            buf_size_t const new_slots_offset = header_length;


            buf_size_t const cur_objs_offset =
                NS(ManagedBuffer_get_section_offset)(
                    begin, OBJECTS_ID, slot_size );

            buf_size_t const cur_objs_size =
                NS(ManagedBuffer_get_section_size)(
                    begin, OBJECTS_ID, slot_size );

            buf_size_t const new_objs_offset =
                new_slots_offset + requ_slots_extent;


            buf_size_t const cur_dataptrs_offset =
                NS(ManagedBuffer_get_section_offset)(
                    begin, DATAPTRS_ID, slot_size );

            buf_size_t const cur_dataptrs_size =
                NS(ManagedBuffer_get_section_size)(
                    begin, DATAPTRS_ID, slot_size );

            buf_size_t const new_dataptrs_offset =
                new_objs_offset + requ_objs_extent;


            buf_size_t const cur_garbage_offset =
                NS(ManagedBuffer_get_section_offset)(
                    begin, GARBAGE_ID, slot_size );

            buf_size_t const cur_garbage_size =
                NS(ManagedBuffer_get_section_size)(
                    begin, GARBAGE_ID, slot_size );

            buf_size_t const new_garbage_offset =
                new_dataptrs_offset + requ_dataptrs_extent;

            /* ------------------------------------------------------------- */

            bool objs_finished  = ( new_objs_offset  == cur_objs_offset  );
            bool slots_finished = ( new_slots_offset == cur_slots_offset );

            bool dataptrs_finished =
                ( new_dataptrs_offset == cur_dataptrs_offset );

            bool garbage_finished =
                ( new_garbage_offset == cur_garbage_offset );

            SIXTRL_ASSERT( slots_finished );

            /* ------------------------------------------------------------- */

            if( !garbage_finished )
            {
                if( new_garbage_offset > cur_garbage_offset )
                {
                    ptr_to_const_raw_t source = begin + cur_garbage_offset;
                    ptr_to_raw_t  destination = begin + new_garbage_offset;

                    raw_t const z = ( raw_t )0u;

                    SIXTRACKLIB_MOVE_VALUES(
                        raw_t, destination, source, cur_garbage_size );

                    SIXTRACKLIB_SET_VALUES(
                        raw_t, ( ptr_to_raw_t )source, cur_garbage_size, z );

                    if( requ_garbage_extent > cur_garbage_size )
                    {
                        buf_size_t const bytes_to_fill =
                            requ_garbage_extent - cur_garbage_size;

                        SIXTRACKLIB_SET_VALUES( raw_t,
                            destination + cur_garbage_size, bytes_to_fill, z );
                    }

                    garbage_finished = true;
                }
                else if( new_garbage_offset >= (
                         cur_dataptrs_offset + cur_dataptrs_size ) )
                {
                    ptr_to_const_raw_t source  = begin + cur_garbage_offset;
                    ptr_to_raw_t  destination  = begin + new_garbage_offset;

                    SIXTRACKLIB_MOVE_VALUES(
                        raw_t, destination, source, cur_garbage_size );

                    garbage_finished = true;
                }
            }

            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            if( ( garbage_finished ) && ( !dataptrs_finished ) )
            {
                if( new_dataptrs_offset > cur_dataptrs_offset )
                {
                    ptr_to_const_raw_t source  = begin + cur_dataptrs_offset;
                    ptr_to_raw_t  destination  = begin + new_dataptrs_offset;

                    raw_t const z = ( raw_t )0u;

                    SIXTRACKLIB_MOVE_VALUES(
                        raw_t, destination, source, cur_dataptrs_size );

                    SIXTRACKLIB_SET_VALUES(
                        raw_t, ( ptr_to_raw_t )source, cur_dataptrs_size, z );

                    if( requ_dataptrs_extent > cur_dataptrs_size )
                    {
                        buf_size_t const bytes_to_fill =
                            requ_dataptrs_extent - cur_dataptrs_size;

                        SIXTRACKLIB_SET_VALUES( raw_t,
                            destination + cur_dataptrs_size, bytes_to_fill, z);
                    }

                    dataptrs_finished = true;
                }
                else if( new_dataptrs_offset >= (
                    ( cur_objs_offset + cur_objs_size ) ) )
                {
                    ptr_to_const_raw_t source  = begin + cur_dataptrs_offset;
                    ptr_to_raw_t  destination  = begin + new_dataptrs_offset;

                    SIXTRACKLIB_MOVE_VALUES(
                        raw_t, destination, source, cur_dataptrs_size );

                    dataptrs_finished = true;
                }
            }

            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            if( ( garbage_finished ) && ( dataptrs_finished ) &&
                ( !objs_finished   ) )
            {
                if( new_objs_offset > cur_objs_offset )
                {
                    ptr_to_const_raw_t source  = begin + cur_objs_offset;
                    ptr_to_raw_t  destination  = begin + new_objs_offset;

                    raw_t const z = ( raw_t )0u;

                    SIXTRACKLIB_MOVE_VALUES(
                        raw_t, destination, source, cur_objs_size );

                    SIXTRACKLIB_SET_VALUES(
                        raw_t, ( ptr_to_raw_t )source, cur_objs_size, z );

                    if( requ_objs_extent > cur_objs_size )
                    {
                        buf_size_t const bytes_to_fill =
                            requ_objs_extent - cur_objs_size;

                        SIXTRACKLIB_SET_VALUES( raw_t,
                            destination + cur_objs_size, bytes_to_fill, z );
                    }

                    objs_finished = true;
                }
                else if( new_objs_offset >= (
                        cur_slots_offset + cur_slots_size ) )
                {
                    ptr_to_const_raw_t source  = begin + cur_objs_offset;
                    ptr_to_raw_t  destination  = begin + new_objs_offset;

                    SIXTRACKLIB_MOVE_VALUES(
                        raw_t, destination, source, cur_objs_size );

                    objs_finished = true;
                }
            }

            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            if( ( slots_finished ) && ( dataptrs_finished ) &&
                ( objs_finished  ) && ( garbage_finished  ) )
            {
                if( requ_slots_extent > cur_slots_size )
                {
                    ptr_to_raw_t destination = begin + new_slots_offset;

                    raw_t const z = ( raw_t )0u;

                    buf_size_t const bytes_to_fill =
                        requ_slots_extent - cur_slots_size;

                    SIXTRACKLIB_SET_VALUES( raw_t,
                        destination + cur_slots_size, bytes_to_fill, z );
                }
            }

            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            if( ( slots_finished ) && ( dataptrs_finished ) &&
                ( objs_finished  ) && ( garbage_finished  ) )
            {
                address_t const new_slots_section_begin_addr =
                    ( address_t )( uintptr_t )( begin + new_slots_offset );

                address_t const new_objs_section_begin_addr =
                    ( address_t )( uintptr_t )( begin + new_objs_offset );

                address_t const new_dataptrs_section_begin_addr =
                    ( address_t )( uintptr_t )( begin  + new_dataptrs_offset );

                address_t const new_garbage_section_begin_addr =
                    ( address_t )( uintptr_t )( begin + new_garbage_offset );

                ptr_to_addr_t ptr_header = ( ptr_to_addr_t )begin;

                SIXTRL_ASSERT( ptr_header[ 0 ] ==
                    ( address_t )( uintptr_t )begin );

                SIXTRL_ASSERT( ptr_header[ SLOTS_ID ] ==
                    new_slots_section_begin_addr );

                ptr_header[ 1 ]           = new_buffer_length;
                ptr_header[ SLOTS_ID    ] = new_slots_section_begin_addr;
                ptr_header[ OBJECTS_ID  ] = new_objs_section_begin_addr;
                ptr_header[ DATAPTRS_ID ] = new_dataptrs_section_begin_addr;
                ptr_header[ GARBAGE_ID  ] = new_garbage_section_begin_addr;

                NS(ManagedBuffer_set_section_max_size)(
                    begin, SLOTS_ID, requ_slots_extent, slot_size );

                NS(ManagedBuffer_set_section_max_size)(
                    begin, OBJECTS_ID, requ_objs_extent, slot_size );

                NS(ManagedBuffer_set_section_max_size)(
                    begin, DATAPTRS_ID, requ_dataptrs_extent, slot_size );

                NS(ManagedBuffer_set_section_max_size)(
                    begin, GARBAGE_ID, requ_garbage_extent, slot_size );

                SIXTRL_ASSERT( cur_num_slots ==
                    NS(ManagedBuffer_get_section_num_entities)(
                        begin, SLOTS_ID, slot_size ) );

                SIXTRL_ASSERT( cur_num_objects ==
                    NS(ManagedBuffer_get_section_num_entities)(
                        begin, OBJECTS_ID, slot_size ) );

                SIXTRL_ASSERT( cur_num_dataptrs ==
                    NS(ManagedBuffer_get_section_num_entities)(
                        begin, DATAPTRS_ID, slot_size ) );

                SIXTRL_ASSERT( cur_num_garbage_ranges ==
                    NS(ManagedBuffer_get_section_num_entities)(
                        begin, GARBAGE_ID, slot_size ) );

                success = 0;
            }
        }

        if(  ptr_new_buffer_length != SIXTRL_NULLPTR )
        {
            *ptr_new_buffer_length  = new_buffer_length;
        }
    }
    else if( success == 0 )
    {
        if(  ptr_new_buffer_length != SIXTRL_NULLPTR )
        {
            *ptr_new_buffer_length =
                NS(ManagedBuffer_get_buffer_length)( begin, slot_size );
        }
    }

    return success;
}

/* ========================================================================= */

SIXTRL_INLINE int NS(ManagedBuffer_init)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_buffer_length,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_ranges,
    NS(buffer_size_t) const max_data_buffer_length,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)               buf_size_t;
    typedef NS(buffer_addr_t)               address_t;
    typedef SIXTRL_ARGPTR_DEC address_t*    ptr_to_addr_t;
    typedef unsigned char                   raw_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    if( ( begin != SIXTRL_NULLPTR ) && ( slot_size != ZERO_SIZE ) )
    {
        SIXTRL_STATIC_VAR buf_size_t const SLOTS_ID        = ( buf_size_t )3u;
        SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID      = ( buf_size_t )4u;
        SIXTRL_STATIC_VAR buf_size_t const DATAPTRS_ID     = ( buf_size_t )5u;
        SIXTRL_STATIC_VAR buf_size_t const GARBAGE_ID      = ( buf_size_t )6u;

        buf_size_t const header_length =
            NS(ManagedBuffer_get_header_length)( begin, slot_size );

        buf_size_t const slots_section_max_size =
            NS(ManagedBuffer_calculate_section_size)(
                begin, SLOTS_ID, max_num_slots, slot_size );

        buf_size_t const objs_section_max_size =
            NS(ManagedBuffer_calculate_section_size)(
                begin, OBJECTS_ID, max_num_objects, slot_size );

        buf_size_t const dataptrs_section_max_size =
            NS(ManagedBuffer_calculate_section_size)(
                begin, DATAPTRS_ID, max_num_dataptrs, slot_size );

        buf_size_t const garbage_section_max_size =
            NS(ManagedBuffer_calculate_section_size)(
                begin, GARBAGE_ID, max_num_garbage_ranges, slot_size );

        buf_size_t const requ_buffer_length = header_length   +
            slots_section_max_size    + objs_section_max_size +
            dataptrs_section_max_size + garbage_section_max_size;

        ptr_to_addr_t ptr_header = ( ptr_to_addr_t )begin;

        SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size ) == 0u );

        if( requ_buffer_length <= max_data_buffer_length )
        {
            buf_size_t const slots_offset = header_length;
            buf_size_t const objs_offset  =
                slots_offset + slots_section_max_size;

            buf_size_t const dataptrs_offset =
                objs_offset + objs_section_max_size;

            buf_size_t const garbage_offset  =
                dataptrs_offset + dataptrs_section_max_size;

            ptr_to_addr_t slots_section_begin =
                ( ptr_to_addr_t )( begin + slots_offset );

            ptr_to_addr_t objs_section_begin =
                ( ptr_to_addr_t )( begin + objs_offset );

            ptr_to_addr_t dataptrs_section_begin =
                ( ptr_to_addr_t )( begin + dataptrs_offset );

            ptr_to_addr_t garbage_section_begin =
                ( ptr_to_addr_t )( begin + garbage_offset );

            unsigned char const zero = ( unsigned char )0u;

            SIXTRACKLIB_SET_VALUES( raw_t, begin, requ_buffer_length, zero );

            ptr_header[ 0 ] = ( address_t )( uintptr_t )begin;
            ptr_header[ 1 ] = requ_buffer_length;
            ptr_header[ 2 ] = header_length;
            ptr_header[ 3 ] = ( address_t )( uintptr_t )slots_section_begin;
            ptr_header[ 4 ] = ( address_t )( uintptr_t )objs_section_begin;
            ptr_header[ 5 ] = ( address_t )( uintptr_t )dataptrs_section_begin;
            ptr_header[ 6 ] = ( address_t )( uintptr_t )garbage_section_begin;

            NS(ManagedBuffer_set_section_max_size)(
                begin, SLOTS_ID, slots_section_max_size, slot_size );

            NS(ManagedBuffer_set_section_num_entities)(
                begin, SLOTS_ID, 0u, slot_size );

            NS(ManagedBuffer_set_section_max_size)(
                begin, OBJECTS_ID, objs_section_max_size, slot_size );

            NS(ManagedBuffer_set_section_num_entities)(
                begin, OBJECTS_ID, 0u, slot_size );

            NS(ManagedBuffer_set_section_max_size)(
                begin, DATAPTRS_ID, dataptrs_section_max_size, slot_size );

            NS(ManagedBuffer_set_section_num_entities)(
                begin, DATAPTRS_ID, 0u, slot_size );

            NS(ManagedBuffer_set_section_max_size)(
                begin, GARBAGE_ID, garbage_section_max_size, slot_size );

            NS(ManagedBuffer_set_section_num_entities)(
                begin, GARBAGE_ID, 0u, slot_size );

            success = 0;
        }

        if(  ptr_buffer_length != SIXTRL_NULLPTR )
        {
            *ptr_buffer_length  = requ_buffer_length;
        }
    }

    return success;
}

/* ========================================================================= */

SIXTRL_INLINE void NS(ManagedBuffer_clear)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    bool const set_data_to_zero, NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)            buf_size_t;
    typedef NS(buffer_addr_t)            address_t;
    typedef SIXTRL_ARGPTR_DEC address_t* ptr_to_addr_t;

    SIXTRL_STATIC_VAR buf_size_t const SLOTS_ID    = ( buf_size_t )3u;
    SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID  = ( buf_size_t )4u;
    SIXTRL_STATIC_VAR buf_size_t const DATAPTRS_ID = ( buf_size_t )5u;
    SIXTRL_STATIC_VAR buf_size_t const GARBAGE_ID  = ( buf_size_t )6u;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE   = ( buf_size_t )0u;

    if( ( begin != SIXTRL_NULLPTR ) && ( slot_size > ZERO_SIZE ) &&
        ( ( ( ( uintptr_t )begin ) % slot_size )  == ZERO_SIZE ) )
    {
        buf_size_t const num_objects =
            NS(ManagedBuffer_get_section_num_entities)(
                begin, OBJECTS_ID, slot_size );

        buf_size_t const num_dataptrs =
            NS(ManagedBuffer_get_section_num_entities)(
                begin, DATAPTRS_ID, slot_size );

        buf_size_t const num_slots =
            NS(ManagedBuffer_get_section_num_entities)(
                begin, SLOTS_ID, slot_size );

        buf_size_t const num_garbage_ranges =
            NS(ManagedBuffer_get_section_num_entities)(
                begin, GARBAGE_ID, slot_size );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_section_max_num_entities)(
            begin, SLOTS_ID, slot_size ) >= num_slots );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_section_max_num_entities)(
            begin, OBJECTS_ID, slot_size ) >= num_objects );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_section_max_num_entities)(
            begin, DATAPTRS_ID, slot_size ) >= num_dataptrs );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_section_max_num_entities)(
            begin, GARBAGE_ID, slot_size ) >= num_garbage_ranges );

        if( set_data_to_zero )
        {
            if( num_objects > ZERO_SIZE )
            {
                typedef NS(Object)                      object_t;
                typedef SIXTRL_ARGPTR_DEC object_t*     ptr_to_object_t;

                ptr_to_object_t obj_it = ( ptr_to_object_t
                    )NS(ManagedBuffer_get_ptr_to_section_data)(
                        begin, OBJECTS_ID, slot_size );

                ptr_to_object_t obj_end = obj_it + num_objects;

                SIXTRL_ASSERT( obj_it != SIXTRL_NULLPTR );

                for( ; obj_it != obj_end ; ++obj_it )
                {
                    NS(Object_preset)( obj_it );
                }
            }

            if( num_garbage_ranges > ZERO_SIZE )
            {
                typedef NS(BufferGarbage)               garbage_t;
                typedef SIXTRL_ARGPTR_DEC garbage_t*    ptr_to_garbage_t;

                ptr_to_garbage_t it  = ( ptr_to_garbage_t
                    )NS(ManagedBuffer_get_ptr_to_section_data)(
                        begin, GARBAGE_ID, slot_size );

                ptr_to_garbage_t end = it + num_garbage_ranges;

                SIXTRL_ASSERT( it != SIXTRL_NULLPTR );

                for( ; it != end ; ++it )
                {
                    NS(BufferGarbage_preset)( it );
                }
            }

            if( num_dataptrs > ZERO_SIZE )
            {
                address_t const ZERO_ADDR = ( address_t )0;

                ptr_to_addr_t it  = ( ptr_to_addr_t
                    )NS(ManagedBuffer_get_ptr_to_section_data)(
                        begin, DATAPTRS_ID, slot_size );

                ptr_to_addr_t end = it + num_dataptrs;

                SIXTRL_ASSERT( it != SIXTRL_NULLPTR );

                for( ; it != end ; ++it ) { *it = ZERO_ADDR; }
            }

            if( num_slots > ZERO_SIZE )
            {
                address_t const ZERO_ADDR = ( address_t )0;

                ptr_to_addr_t it  = ( ptr_to_addr_t
                    )NS(ManagedBuffer_get_ptr_to_section_data)(
                        begin, SLOTS_ID, slot_size );

                ptr_to_addr_t end = it + num_slots;

                SIXTRL_ASSERT( it != SIXTRL_NULLPTR );

                for( ; it != end ; ++it ) { *it = ZERO_ADDR; }
            }
        }

        NS(ManagedBuffer_set_section_num_entities)(
            begin, SLOTS_ID, ZERO_SIZE, slot_size );

        NS(ManagedBuffer_set_section_num_entities)(
            begin, OBJECTS_ID, ZERO_SIZE, slot_size );

        NS(ManagedBuffer_set_section_num_entities)(
            begin, DATAPTRS_ID, ZERO_SIZE, slot_size );

        NS(ManagedBuffer_set_section_num_entities)(
            begin, GARBAGE_ID, ZERO_SIZE, slot_size );
    }

    return;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRL_COMMON_IMPL_BUFFER_MEM_H__ */

/*end: sixtracklib/common/impl/managed_buffer.h */
