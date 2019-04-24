#ifndef SIXTRL_COMMON_BUFFER_BUFFER_MINIMAL_H__
#define SIXTRL_COMMON_BUFFER_BUFFER_MINIMAL_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if !defined( __cplusplus )
        #include <stdbool.h>
        #include <stddef.h>
        #include <stdlib.h>
        #include <limits.h>
    #else
        #include <cstddef>
        #include <cstdlib>
        #include <limits>
    #endif /* !defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_diff_t)
    NS(ManagedBuffer_get_limit_offset_max)( void );

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_diff_t)
    NS(ManagedBuffer_get_limit_offset_min)( void );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
    NS(ManagedBuffer_get_slot_based_length)(
        NS(buffer_size_t) const in_length, NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(ManagedBuffer_get_section_entity_size)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(ManagedBuffer_get_section_header_length)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(ManagedBuffer_get_header_length)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(ManagedBuffer_get_buffer_length)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(ManagedBuffer_get_const_ptr_to_section)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(ManagedBuffer_get_const_ptr_to_section_data)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(ManagedBuffer_get_const_ptr_to_section_end)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(ManagedBuffer_get_ptr_to_section)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(ManagedBuffer_get_ptr_to_section_data)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(ManagedBuffer_get_ptr_to_section_end)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(ManagedBuffer_get_section_max_size)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(ManagedBuffer_get_section_size)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(ManagedBuffer_get_section_max_num_entities)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(ManagedBuffer_get_section_num_entities)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size );

/* ------------------------------------------------------------------------- */

struct NS(Object);

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC struct NS(Object) const*
NS(ManagedBuffer_get_const_objects_index_begin)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC struct NS(Object) const*
NS(ManagedBuffer_get_const_objects_index_end)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT end,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC struct NS(Object)*
NS(ManagedBuffer_get_objects_index_begin)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC struct NS(Object)*
NS(ManagedBuffer_get_objects_index_end)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(ManagedBuffer_get_num_objects)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) slot_size );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC bool NS(ManagedBuffer_needs_remapping)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size );

/* ========================================================================= */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer/buffer_object.h"
    #include "sixtracklib/common/buffer/buffer_garbage.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

/* ========================================================================= *
 * ======== INLINE IMPLEMENTATION                                            *
 * ========================================================================= */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(buffer_addr_diff_t)
    NS(ManagedBuffer_get_limit_offset_max)()
{
    #if defined( _GPUCODE )
        #if defined( __OPENCL_VERSION__ )

            SIXTRL_ASSERT( sizeof( NS(buffer_addr_diff_t) >=
                           sizeof( ptr_to_raw_t ) ) );

            return ( NS(buffer_addr_diff_t) )LONG_MAX;

        #elif defined( __CUDACC__ )

            SIXTRL_ASSERT( sizeof( NS(buffer_addr_diff_t) ) >=
                           sizeof( long long int ) );

            return ( NS(buffer_addr_diff_t) )9223372036854775807L;
        #else
            return ( NS(buffer_addr_diff_t) )9223372036854775807L;

        #endif /* defined( __OPENCL_VERSION__ ) */
    #elif defined( __cplusplus )
        typedef NS(buffer_addr_diff_t) addr_diff_t;

        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::digits >= 63u );
        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::is_signed  );
        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::is_integer );

        #if defined( __CUDA_ARCH__ )

        return ( NS(buffer_addr_diff_t) )9223372036854775807L;

        #else  /* defined( __CUDA_ARCH__ ) */

        return std::numeric_limits< addr_diff_t >::max();

        #endif /* defined( __CUDA_ARCH__ ) */

    #else
        SIXTRL_ASSERT( sizeof( NS(buffer_addr_diff_t) ) >=
                       sizeof( long long int ) );

        return ( NS(buffer_addr_diff_t) )LLONG_MAX;

    #endif /* defined( _GPUCODE ) */
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_addr_diff_t)
    NS(ManagedBuffer_get_limit_offset_min)()
{
    #if defined( _GPUCODE )
         #if defined( __OPENCL_VERSION__ )  && \
             defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
             ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )

            SIXTRL_ASSERT( sizeof( NS(buffer_addr_diff_t) >=
                           sizeof( ptr_to_raw_t ) ) );

            return ( NS(buffer_addr_diff_t ) )LONG_MIN;

        #elif defined( __CUDACC__ )  && \
              defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
              ( SIXTRACKLIB_ENABLE_MODULE_CUDA == 1 )

            SIXTRL_ASSERT( sizeof( NS(buffer_addr_diff_t) ) >=
                           sizeof( long long int ) );

            return ( NS(buffer_addr_diff_t) )-9223372036854775807L;
        #else

            return ( NS(buffer_addr_diff_t) )-9223372036854775807L;

        #endif /* defined( __OPENCL_VERSION__ ) */
    #elif defined( __cplusplus )
        typedef NS(buffer_addr_diff_t) addr_diff_t;

        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::digits >= 63u );
        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::is_signed  );
        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::is_integer );

        #if defined( __CUDA_ARCH__ )

        return ( NS(buffer_addr_diff_t) )-9223372036854775807L;

        #else /* defined( __CUDA_ARCH__ ) */

        return std::numeric_limits< addr_diff_t >::min();

        #endif /* defined( __CUDA_ARCH__ ) */

    #else
        SIXTRL_ASSERT( sizeof( NS(buffer_addr_diff_t) ) >=
                       sizeof( long long int ) );

        return ( NS(buffer_addr_diff_t) )LLONG_MIN;

    #endif /* defined( _GPUCODE ) */
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(ManagedBuffer_get_slot_based_length)(
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

SIXTRL_INLINE NS(buffer_size_t) NS(ManagedBuffer_get_section_entity_size)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const header_section_id,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    buf_size_t entity_size = ZERO_SIZE;

    ( void )begin;

    if( slot_size > ZERO_SIZE )
    {
        SIXTRL_ASSERT(
            ( begin == SIXTRL_NULLPTR ) ||
            ( ( ( ( uintptr_t )begin ) % slot_size )  == ZERO_SIZE ) );

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

                entity_size = NS(ManagedBuffer_get_slot_based_length)(
                    sizeof( object_t ), slot_size );

                break;
            }

            case 5u:
            {
                typedef NS(buffer_addr_t) const* ptr_to_addr_t;

                entity_size = NS(ManagedBuffer_get_slot_based_length)(
                    sizeof( ptr_to_addr_t ), slot_size );

                break;
            }

            case 6u:
            {
                typedef NS(BufferGarbage) garbage_range_t;

                entity_size = NS(ManagedBuffer_get_slot_based_length)(
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

SIXTRL_INLINE NS(buffer_size_t) NS(ManagedBuffer_get_section_header_length)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) address_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    buf_size_t section_header_length = ZERO_SIZE;

    ( void )begin;

    if( slot_size > ZERO_SIZE )
    {
        buf_size_t const addr_size = NS(ManagedBuffer_get_slot_based_length)(
            sizeof( address_t ), slot_size );

        SIXTRL_ASSERT( addr_size > ZERO_SIZE );
        SIXTRL_ASSERT( ( begin == SIXTRL_NULLPTR ) ||
            ( ( ( ( uintptr_t )begin ) % slot_size )  == ZERO_SIZE ) );

        section_header_length = ( buf_size_t )2u * addr_size;
    }

    return section_header_length;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(ManagedBuffer_get_header_length)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)             buf_size_t;
    typedef NS(buffer_addr_t)             address_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC address_t* ptr_to_addr_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE       = ( buf_size_t )0u;

    buf_size_t header_length = ZERO_SIZE;

    if( slot_size > ZERO_SIZE )
    {
        ptr_to_addr_t ptr_header = ( ptr_to_addr_t )begin;

        SIXTRL_ASSERT( ( begin == SIXTRL_NULLPTR ) ||
            ( ( ( ( uintptr_t )begin ) % slot_size )  == ZERO_SIZE ) );

        header_length = ( begin != SIXTRL_NULLPTR )
            ? ptr_header[ 2 ] : ZERO_SIZE;

        if( header_length == ZERO_SIZE )
        {
            SIXTRL_STATIC_VAR buf_size_t const
                NUM_HDR_ENTRIES = ( buf_size_t )8u;

            header_length = NS(ManagedBuffer_get_slot_based_length)(
                sizeof( address_t ), slot_size ) * NUM_HDR_ENTRIES;
        }

        SIXTRL_ASSERT( ( ptr_header == SIXTRL_NULLPTR ) ||
                       ( ( ptr_header[ 1 ] >= header_length ) ||
                           ( ( ptr_header[ 1 ] == ZERO_SIZE ) &&
                             ( ptr_header[ 2 ] == ZERO_SIZE ) ) ) );
    }

    return header_length;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(ManagedBuffer_get_buffer_length)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)            buf_size_t;
    typedef NS(buffer_addr_t)            address_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC address_t* ptr_to_addr_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    buf_size_t buffer_length = ZERO_SIZE;

    if( ( begin != SIXTRL_NULLPTR ) && ( slot_size > ZERO_SIZE ) )
    {
        ptr_to_addr_t  ptr_header  = ( ptr_to_addr_t )begin;

        #if !defined( NDEBUG )
        buf_size_t const header_length =
            NS(ManagedBuffer_get_header_length)( begin, slot_size );

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

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(ManagedBuffer_get_const_ptr_to_section)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id, NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* ptr_to_raw_t;
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef NS(buffer_addr_t)                       address_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC address_t const*     ptr_to_addr_t;

    ptr_to_raw_t data_begin_ptr = SIXTRL_NULLPTR;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( begin != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) &&
        ( section_id >= 3u ) && ( section_id <= 6u ) )
    {
        ptr_to_addr_t ptr_header = ( ptr_to_addr_t )begin;

        #if !defined( NDEBUG )
        buf_size_t const addr_size = NS(ManagedBuffer_get_slot_based_length)(
            sizeof( address_t ), slot_size );

        buf_size_t const header_length =
            NS(ManagedBuffer_get_header_length)( begin, slot_size );

        SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size ) == 0u );
        SIXTRL_ASSERT( header_length >= ( section_id * addr_size ) );

        #endif /* !defined( NDEBUG ) */

        data_begin_ptr = ( ptr_to_raw_t )( uintptr_t )ptr_header[ section_id ];
    }

    return data_begin_ptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(ManagedBuffer_get_const_ptr_to_section_data)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id,
    NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* ptr_to_raw_t;
    typedef NS(buffer_size_t)                       buf_size_t;

    ptr_to_raw_t ptr_data_begin = NS(ManagedBuffer_get_const_ptr_to_section)(
        begin, section_id, slot_size );

    if( ptr_data_begin != SIXTRL_NULLPTR )
    {
        buf_size_t const section_hdr_len =
            NS(ManagedBuffer_get_section_header_length)( begin, slot_size );

        ptr_data_begin = ptr_data_begin + section_hdr_len;
    }

    return ptr_data_begin;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(ManagedBuffer_get_const_ptr_to_section_end)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id, NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* ptr_to_raw_t;
    typedef NS(buffer_size_t)                       buf_size_t;

    ptr_to_raw_t end_ptr = NS(ManagedBuffer_get_const_ptr_to_section)(
            begin, section_id, slot_size );

    if( end_ptr != SIXTRL_NULLPTR )
    {
        buf_size_t const section_size = NS(ManagedBuffer_get_section_size)(
            begin, section_id, slot_size );

        end_ptr = end_ptr + section_size;
    }

    return end_ptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(ManagedBuffer_get_ptr_to_section)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id, NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_t;
    return ( ptr_t )NS(ManagedBuffer_get_const_ptr_to_section)(
        begin, section_id, slot_size );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(ManagedBuffer_get_ptr_to_section_data)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id, NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_t;
    return ( ptr_t )NS(ManagedBuffer_get_const_ptr_to_section_data)(
            begin, section_id, slot_size );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(ManagedBuffer_get_ptr_to_section_end)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id, NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_t;
    return ( ptr_t )NS(ManagedBuffer_get_const_ptr_to_section_end)(
            begin, section_id, slot_size );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(ManagedBuffer_get_section_max_size)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const * SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef NS(buffer_addr_t)                       address_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC address_t const*     ptr_to_addr_t;

    ptr_to_addr_t ptr_to_section = ( ptr_to_addr_t
        )NS(ManagedBuffer_get_const_ptr_to_section)(
            begin, section_id, slot_size );

    buf_size_t const max_section_size = ( ptr_to_section != SIXTRL_NULLPTR )
        ? ( buf_size_t )( *ptr_to_section ) : ( buf_size_t )0u;

    SIXTRL_ASSERT( NS(ManagedBuffer_get_section_header_length)(
        begin, slot_size ) <= max_section_size );

    return max_section_size;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(ManagedBuffer_get_section_size)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef NS(buffer_addr_t)                       address_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC address_t const*     ptr_to_addr_t;

    ptr_to_addr_t ptr_to_section = ( ptr_to_addr_t
        )NS(ManagedBuffer_get_const_ptr_to_section)(
            begin, section_id, slot_size );

    buf_size_t const section_header_length =
        NS(ManagedBuffer_get_section_header_length)( begin, slot_size );

    buf_size_t const num_elements = ( ptr_to_section != SIXTRL_NULLPTR )
        ? ( buf_size_t )( ( ( ptr_to_addr_t )ptr_to_section )[ 1 ] )
        : ( buf_size_t )0u;

    buf_size_t const entity_size = NS(ManagedBuffer_get_section_entity_size)(
        begin, section_id, slot_size );

    buf_size_t section_size =
        section_header_length + num_elements * entity_size;

    SIXTRL_ASSERT( entity_size > ( buf_size_t )0u );

    #if !defined( NDEBUG ) && !defined( _GPUCODE )
    buf_size_t const max_section_size = ( ptr_to_section != SIXTRL_NULLPTR )
        ? ( buf_size_t )( ( ( ptr_to_addr_t )ptr_to_section )[ 0 ] )
        : ( buf_size_t )0u;

    SIXTRL_ASSERT( section_header_length <= max_section_size );
    SIXTRL_ASSERT( max_section_size >= section_size );
    #endif /* !defined( NDEBUG ) && !defined( _GPUCODE ) */

    return section_size;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(ManagedBuffer_get_section_max_num_entities)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id, NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef NS(buffer_addr_t)                       address_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC address_t const*     ptr_to_addr_t;

    ptr_to_addr_t ptr_to_section = ( ptr_to_addr_t
        )NS(ManagedBuffer_get_const_ptr_to_section)(
            begin, section_id, slot_size );

    buf_size_t const section_header_length =
        NS(ManagedBuffer_get_section_header_length)( begin, slot_size );

    buf_size_t const max_section_size = ( ptr_to_section != SIXTRL_NULLPTR )
        ? ( buf_size_t )( ( ( ptr_to_addr_t )ptr_to_section )[ 0 ] )
        : ( buf_size_t )0u;

    buf_size_t const entity_size = NS(ManagedBuffer_get_section_entity_size)(
        begin, section_id, slot_size );

    buf_size_t max_num_elements = ( max_section_size >= section_header_length )
        ? ( max_section_size - section_header_length )
        : ( buf_size_t )0u;

    SIXTRL_ASSERT( entity_size > ( buf_size_t )0u );
    max_num_elements /= entity_size;

    SIXTRL_ASSERT( max_num_elements >=
        NS(ManagedBuffer_get_section_num_entities)(
            begin, section_id, slot_size ) );

    return max_num_elements;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(ManagedBuffer_get_section_num_entities)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const section_id, NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef NS(buffer_addr_t)                       address_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC address_t const*     ptr_to_addr_t;

    ptr_to_addr_t ptr_to_section = ( ptr_to_addr_t
        )NS(ManagedBuffer_get_const_ptr_to_section)(
            begin, section_id, slot_size );

    buf_size_t const num_elements = ( ptr_to_section != SIXTRL_NULLPTR )
        ? ( buf_size_t )( ( ( ptr_to_addr_t )ptr_to_section )[ 1 ] )
        : ( buf_size_t )0u;

    #if !defined( NDEBUG ) && !defined( _GPUCODE )

    buf_size_t const max_section_size = ( ptr_to_section != SIXTRL_NULLPTR )
        ? ( buf_size_t )( ( ( ptr_to_addr_t )ptr_to_section )[ 0 ] )
        : ( buf_size_t )0u;

    buf_size_t const entity_size = NS(ManagedBuffer_get_section_entity_size)(
        begin, section_id, slot_size );

    SIXTRL_ASSERT( entity_size > ( buf_size_t )0u );
    SIXTRL_ASSERT( ( ( entity_size * num_elements ) +
        NS(ManagedBuffer_get_section_header_length)( begin, slot_size ) )
            <= max_section_size );

    #endif /* !defined( NDEBUG ) && !defined( _GPUCODE ) */

    return num_elements;
}

/* ========================================================================= */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(Object) const*
NS(ManagedBuffer_get_const_objects_index_begin)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Object) const* ptr_to_object_t;

    SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID = ( buf_size_t )4u;

    return ( ptr_to_object_t )NS(ManagedBuffer_get_const_ptr_to_section_data)(
        begin, OBJECTS_ID, slot_size );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(Object) const*
NS(ManagedBuffer_get_const_objects_index_end)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)                   buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Object) const* ptr_to_object_t;

    SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID = ( buf_size_t )4u;

    ptr_to_object_t end_ptr = ( ptr_to_object_t
        )NS(ManagedBuffer_get_const_ptr_to_section_data)(
            begin, OBJECTS_ID, slot_size );

    SIXTRL_ASSERT( end_ptr != SIXTRL_NULLPTR );

    return end_ptr + NS(ManagedBuffer_get_section_num_entities)(
        begin, OBJECTS_ID, slot_size );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(Object)*
NS(ManagedBuffer_get_objects_index_begin)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) slot_size )
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(Object)*
        )NS(ManagedBuffer_get_const_objects_index_begin)( begin, slot_size );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(Object)*
NS(ManagedBuffer_get_objects_index_end)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size )
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(Object)*
        )NS(ManagedBuffer_get_const_objects_index_end)( begin, slot_size );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(ManagedBuffer_get_num_objects)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t const OBJS_ID = ( buf_size_t )4u;
    return NS(ManagedBuffer_get_section_num_entities)(
        begin, OBJS_ID, slot_size );
}

/* ========================================================================= */

SIXTRL_INLINE bool NS(ManagedBuffer_needs_remapping)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_addr_t)   address_t;
    typedef NS(buffer_size_t)   buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC address_t const*    ptr_to_addr_t;

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

#endif /* SIXTRL_COMMON_BUFFER_BUFFER_MINIMAL_H__ */

/* end: sixtracklib/common/buffer/buffer_minimal.h */
