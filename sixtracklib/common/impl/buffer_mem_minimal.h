#ifndef SIXTRL_COMMON_IMPL_BUFFER_MINIMAL_H__
#define SIXTRL_COMMON_IMPL_BUFFER_MINIMAL_H__

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

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(BufferMem_get_slot_based_length)(
    NS(buffer_size_t) const in_length, NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BufferMem_get_section_entity_size)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BufferMem_get_section_header_length)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(BufferMem_get_header_length)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(BufferMem_get_buffer_length)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC SIXTRL_DATAPTR_DEC unsigned char const*
NS(BufferMem_get_const_ptr_to_section)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_DATAPTR_DEC unsigned char const*
NS(BufferMem_get_const_ptr_to_section_data)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_DATAPTR_DEC unsigned char const*
NS(BufferMem_get_const_ptr_to_section_end)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_DATAPTR_DEC unsigned char*
NS(BufferMem_get_ptr_to_section)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_DATAPTR_DEC unsigned char*
NS(BufferMem_get_ptr_to_section_data)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_DATAPTR_DEC unsigned char*
NS(BufferMem_get_ptr_to_section_end)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BufferMem_get_section_max_size)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BufferMem_get_section_size)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BufferMem_get_section_max_num_entities)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BufferMem_get_section_num_entities)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(Object) const*
NS(BufferMem_get_const_objects_index_begin)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(Object) const*
NS(BufferMem_get_const_objects_index_end)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT end,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(BufferMem_get_num_objects)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) slot_size );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC bool NS(BufferMem_needs_remapping)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size );

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

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(buffer_size_t) NS(BufferMem_get_slot_based_length)(
    NS(buffer_size_t) const in_length, NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)  buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t const remainder = ( slot_size > ZERO )
        ? ( in_length % slot_size ) : ZERO;

    return ( remainder == ZERO )
        ? ( in_length ) : ( in_length + ( slot_size - remainder ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(BufferMem_get_section_entity_size)(
    SIXTRL_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    buf_size_t entity_size = ZERO_SIZE;

    if( ( begin != SIXTRL_NULLPTR ) && ( slot_size > ZERO_SIZE ) )
    {
        SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size )  == ZERO_SIZE );

        switch( header_section_id )
        {
            case 3u:
            {
                entity_size = slot_size;
                break;
            }

            case 4u:
            {
                typedef NS(Object) object_t;

                entity_size = NS(BufferMem_get_slot_based_length)(
                    sizeof( object_t ), slot_size );

                break;
            }

            case 5u:
            {
                typedef NS(buffer_addr_t) const* ptr_to_addr_t;

                entity_size = NS(BufferMem_get_slot_based_length)(
                    sizeof( ptr_to_addr_t ), slot_size );

                break;
            }

            case 6u:
            {
                typedef NS(BufferGarbage) garbage_range_t;

                entity_size = NS(BufferMem_get_slot_based_length)(
                    sizeof( garbage_range_t ), slot_size );

                break;
            }

            default:
            {
                entity_size = ( buf_size_t )0u;
            }
        };
    }

    return entity_size;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(BufferMem_get_section_header_length)(
    SIXTRL_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) address_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    buf_size_t section_header_length = ZERO_SIZE;

    if( ( begin != SIXTRL_NULLPTR ) && ( slot_size > ZERO_SIZE ) )
    {
        buf_size_t const addr_size = NS(BufferMem_get_slot_based_length)(
            sizeof( address_t ), slot_size );

        SIXTRL_ASSERT( addr_size > ZERO_SIZE );
        SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size )  == ZERO_SIZE );

        section_header_length = ( buf_size_t )2u * addr_size;
    }

    return section_header_length;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(BufferMem_get_header_length)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)            buf_size_t;
    typedef NS(buffer_addr_t)            address_t;
    typedef SIXTRL_ARGPTR_DEC address_t* ptr_to_addr_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE       = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR buf_size_t const NUM_HDR_ENTRIES = ( buf_size_t )8u;

    buf_size_t header_length = ZERO_SIZE;

    if( ( begin != SIXTRL_NULLPTR ) && ( slot_size > ZERO_SIZE ) )
    {
        ptr_to_addr_t ptr_header = ( ptr_to_addr_t )begin;

        header_length = ptr_header[ 2 ];

        SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size )  == ZERO_SIZE );
        SIXTRL_ASSERT( ptr_header != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( ( ( ( uintptr_t )ptr_header ) %
            sizeof( address_t ) ) == 0u );

        if( header_length == ZERO_SIZE )
        {
            header_length = NS(BufferMem_get_slot_based_length)(
                sizeof( address_t ), slot_size ) * NUM_HDR_ENTRIES;
        }

        SIXTRL_ASSERT( ( ptr_header[ 1 ] >= header_length ) ||
                           ( ( ptr_header[ 1 ] == ZERO_SIZE ) &&
                             ( ptr_header[ 2 ] == ZERO_SIZE ) ) );
    }

    return header_length;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(BufferMem_get_buffer_length)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)            buf_size_t;
    typedef NS(buffer_addr_t)            address_t;
    typedef SIXTRL_ARGPTR_DEC address_t* ptr_to_addr_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    buf_size_t buffer_length = ZERO_SIZE;

    if( ( begin != SIXTRL_NULLPTR ) && ( slot_size > ZERO_SIZE ) )
    {
        ptr_to_addr_t  ptr_header  = ( ptr_to_addr_t )begin;

        #if !defined( NDEBUG )
        buf_size_t const header_length =
            NS(BufferMem_get_header_length)( begin, slot_size );

        #endif /* !defined( NDEBUG ) */

        SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size ) == ZERO_SIZE );
        SIXTRL_ASSERT( ptr_header != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( ( ( ( uintptr_t )ptr_header ) %
            sizeof( address_t ) ) == 0u );

        buffer_length = ptr_header[ 1 ];

        SIXTRL_ASSERT( ( header_length <= buffer_length ) ||
                       ( buffer_length == ZERO_SIZE ) );
    }

    return buffer_length;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_DATAPTR_DEC unsigned char const*
NS(BufferMem_get_const_ptr_to_section)(
    SIXTRL_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id, NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_DATAPTR_DEC unsigned char const* ptr_to_raw_t;
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef NS(buffer_addr_t)                       address_t;
    typedef SIXTRL_DATAPTR_DEC address_t const*     ptr_to_addr_t;

    ptr_to_raw_t data_begin_ptr = SIXTRL_NULLPTR;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( begin != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) &&
        ( section_id >= 3u ) && ( section_id <= 6u ) )
    {
        ptr_to_addr_t ptr_header = ( ptr_to_addr_t )begin;

        #if !defined( NDEBUG )
        buf_size_t const addr_size = NS(BufferMem_get_slot_based_length)(
            sizeof( address_t ), slot_size );

        buf_size_t const header_length =
            NS(BufferMem_get_header_length)( begin, slot_size );

        SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size ) == 0u );
        SIXTRL_ASSERT( header_length >= ( section_id * addr_size ) );

        #endif /* !defined( NDEBUG ) */

        data_begin_ptr = ( ptr_to_raw_t )( uintptr_t )ptr_header[ section_id ];
    }

    return data_begin_ptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_DATAPTR_DEC unsigned char const*
NS(BufferMem_get_const_ptr_to_section_data)(
    SIXTRL_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id,
    NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_DATAPTR_DEC unsigned char const* ptr_to_raw_t;
    typedef NS(buffer_size_t)                       buf_size_t;

    ptr_to_raw_t ptr_data_begin = NS(BufferMem_get_const_ptr_to_section)(
        begin, section_id, slot_size );

    if( ptr_data_begin != SIXTRL_NULLPTR )
    {
        buf_size_t const section_hdr_len =
            NS(BufferMem_get_section_header_length)( begin, slot_size );

        ptr_data_begin = ptr_data_begin + section_hdr_len;
    }

    return ptr_data_begin;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_DATAPTR_DEC unsigned char const*
NS(BufferMem_get_const_ptr_to_section_end)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id, NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_DATAPTR_DEC unsigned char const* ptr_to_raw_t;
    typedef NS(buffer_size_t)                       buf_size_t;

    ptr_to_raw_t end_ptr =
        NS(BufferMem_get_const_ptr_to_section)( begin, section_id, slot_size );

    if( end_ptr != SIXTRL_NULLPTR )
    {
        buf_size_t const section_size =
            NS(BufferMem_get_section_size)( begin, section_id, slot_size );

        end_ptr = end_ptr + section_size;
    }

    return end_ptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_DATAPTR_DEC unsigned char*
NS(BufferMem_get_ptr_to_section)(
    SIXTRL_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id, NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_DATAPTR_DEC unsigned char* ptr_t;
    return ( ptr_t )NS(BufferMem_get_const_ptr_to_section)(
        begin, section_id, slot_size );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_DATAPTR_DEC unsigned char*
NS(BufferMem_get_ptr_to_section_data)(
    SIXTRL_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id, NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_DATAPTR_DEC unsigned char* ptr_t;
    return ( ptr_t )NS(BufferMem_get_const_ptr_to_section_data)(
            begin, section_id, slot_size );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_DATAPTR_DEC unsigned char*
NS(BufferMem_get_ptr_to_section_end)(
    SIXTRL_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id, NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_DATAPTR_DEC unsigned char* ptr_t;
    return ( ptr_t )NS(BufferMem_get_const_ptr_to_section_end)(
            begin, section_id, slot_size );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(BufferMem_get_section_max_size)(
    SIXTRL_DATAPTR_DEC unsigned char const * SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef NS(buffer_addr_t)                       address_t;
    typedef SIXTRL_DATAPTR_DEC address_t const*     ptr_to_addr_t;

    ptr_to_addr_t ptr_to_section = ( ptr_to_addr_t
        )NS(BufferMem_get_const_ptr_to_section)( begin, section_id, slot_size );

    buf_size_t const max_section_size = ( ptr_to_section != SIXTRL_NULLPTR )
        ? ( buf_size_t )( *ptr_to_section ) : ( buf_size_t )0u;

    SIXTRL_ASSERT( NS(BufferMem_get_section_header_length)( begin, slot_size )
        <= max_section_size );

    return max_section_size;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(BufferMem_get_section_size)(
    SIXTRL_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef NS(buffer_addr_t)                       address_t;
    typedef SIXTRL_DATAPTR_DEC address_t const*     ptr_to_addr_t;

    ptr_to_addr_t ptr_to_section = ( ptr_to_addr_t
        )NS(BufferMem_get_const_ptr_to_section)( begin, section_id, slot_size );

    buf_size_t const section_header_length =
        NS(BufferMem_get_section_header_length)( begin, slot_size );

    buf_size_t const num_elements = ( ptr_to_section != SIXTRL_NULLPTR )
        ? ( buf_size_t )( ( ( ptr_to_addr_t )ptr_to_section )[ 1 ] )
        : ( buf_size_t )0u;

    buf_size_t const entity_size = NS(BufferMem_get_section_entity_size)(
        begin, section_id, slot_size );

    buf_size_t section_size =
        section_header_length + num_elements * entity_size;

    SIXTRL_ASSERT( entity_size > ( buf_size_t )0u );

    #if !defined( NDEBUG )
    buf_size_t const max_section_size = ( ptr_to_section != SIXTRL_NULLPTR )
        ? ( buf_size_t )( ( ( ptr_to_addr_t )ptr_to_section )[ 0 ] )
        : ( buf_size_t )0u;

    SIXTRL_ASSERT( section_header_length <= max_section_size );
    SIXTRL_ASSERT( max_section_size >= section_size );
    #endif /* !defined( NDEBUG ) */

    return section_size;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(BufferMem_get_section_max_num_entities)(
    SIXTRL_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id, NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef NS(buffer_addr_t)                       address_t;
    typedef SIXTRL_DATAPTR_DEC address_t const*     ptr_to_addr_t;

    ptr_to_addr_t ptr_to_section = ( ptr_to_addr_t
        )NS(BufferMem_get_const_ptr_to_section)( begin, section_id, slot_size );

    buf_size_t const section_header_length =
        NS(BufferMem_get_section_header_length)( begin, slot_size );

    buf_size_t const max_section_size = ( ptr_to_section != SIXTRL_NULLPTR )
        ? ( buf_size_t )( ( ( ptr_to_addr_t )ptr_to_section )[ 0 ] )
        : ( buf_size_t )0u;

    buf_size_t const entity_size = NS(BufferMem_get_section_entity_size)(
        begin, section_id, slot_size );

    buf_size_t max_num_elements = ( max_section_size >= section_header_length )
        ? ( max_section_size - section_header_length )
        : ( buf_size_t )0u;

    SIXTRL_ASSERT( entity_size > ( buf_size_t )0u );
    max_num_elements /= entity_size;

    SIXTRL_ASSERT( max_num_elements >= NS(BufferMem_get_section_num_entities)(
        begin, section_id, slot_size ) );

    return max_num_elements;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(BufferMem_get_section_num_entities)(
    SIXTRL_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id, NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef NS(buffer_addr_t)                       address_t;
    typedef SIXTRL_DATAPTR_DEC address_t const*     ptr_to_addr_t;

    ptr_to_addr_t ptr_to_section = ( ptr_to_addr_t
        )NS(BufferMem_get_const_ptr_to_section)(
            begin, section_id, slot_size );

    buf_size_t const num_elements = ( ptr_to_section != SIXTRL_NULLPTR )
        ? ( buf_size_t )( ( ( ptr_to_addr_t )ptr_to_section )[ 1 ] )
        : ( buf_size_t )0u;

    #if !defined( NDEBUG )

    buf_size_t const max_section_size = ( ptr_to_section != SIXTRL_NULLPTR )
        ? ( buf_size_t )( ( ( ptr_to_addr_t )ptr_to_section )[ 0 ] )
        : ( buf_size_t )0u;

    buf_size_t const entity_size = NS(BufferMem_get_section_entity_size)(
        begin, section_id, slot_size );

    SIXTRL_ASSERT( entity_size > ( buf_size_t )0u );
    SIXTRL_ASSERT( ( ( entity_size * num_elements ) +
        NS(BufferMem_get_section_header_length)( begin, slot_size ) )
            <= max_section_size );

    #endif /* !defined( NDEBUG ) */

    return num_elements;
}

/* ========================================================================= */

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(Object) const*
NS(BufferMem_get_const_objects_index_begin)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_ARGPTR_DEC NS(Object) const* ptr_to_object_t;

    SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID = ( buf_size_t )4u;

    return ( ptr_to_object_t )NS(BufferMem_get_const_ptr_to_section_data)(
        begin, OBJECTS_ID, slot_size );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(Object) const*
NS(BufferMem_get_const_objects_index_end)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)                   buf_size_t;
    typedef SIXTRL_ARGPTR_DEC NS(Object) const* ptr_to_object_t;

    SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID = ( buf_size_t )4u;

    ptr_to_object_t end_ptr = ( ptr_to_object_t
        )NS(BufferMem_get_const_ptr_to_section_data)(
            begin, OBJECTS_ID, slot_size );

    SIXTRL_ASSERT( end_ptr != SIXTRL_NULLPTR );

    return end_ptr + NS(BufferMem_get_section_num_entities)(
        begin, OBJECTS_ID, slot_size );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(BufferMem_get_num_objects)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t const OBJS_ID = ( buf_size_t )4u;
    return NS(BufferMem_get_section_num_entities)( begin, OBJS_ID, slot_size );
}

/* ========================================================================= */

SIXTRL_INLINE bool NS(BufferMem_needs_remapping)(
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_addr_t)   address_t;
    typedef NS(buffer_size_t)   buf_size_t;
    typedef address_t const*    ptr_to_addr_t;

    ptr_to_addr_t ptr_header = ( ptr_to_addr_t )begin;

    SIXTRL_ASSERT( ( slot_size > ( buf_size_t )0u ) &&
                   ( ( ( ( uintptr_t )begin ) % slot_size ) == 0u ) &&
                   ( ptr_header != SIXTRL_NULLPTR ) );

    return ( ( slot_size > ( buf_size_t )0u ) &&
             ( ptr_header[ 0 ] != ( address_t )ptr_header ) &&
             ( ptr_header[ 0 ] != ( address_t )0u ) );
}

/* ========================================================================= */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRL_COMMON_IMPL_BUFFER_MINIMAL_H__ */

/* end: sixtracklib/common/impl/buffer_minimal.h */
