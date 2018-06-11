#ifndef SIXTRACKLIB_COMMON_MEMORY_POOL_H__
#define SIXTRACKLIB_COMMON_MEMORY_POOL_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/impl/alignment_impl.h"
#include "sixtracklib/common/alignment.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

#endif /* !defined( _GPUCODE ) */
    
/* -------------------------------------------------------------------------- */

typedef struct NS( AllocResult )
{
    unsigned char* p;
    SIXTRL_UINT64_T offset;
    SIXTRL_UINT64_T length;
} NS( AllocResult );

SIXTRL_STATIC bool NS( AllocResult_valid )( 
    const NS( AllocResult ) * const SIXTRL_RESTRICT result );

SIXTRL_STATIC bool NS(AllocResult_is_aligned)( 
    const NS(AllocResult) *const SIXTRL_RESTRICT result, 
    size_t alignment );

SIXTRL_STATIC NS( AllocResult ) *
    NS( AllocResult_preset )( NS( AllocResult ) * SIXTRL_RESTRICT result );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char* 
NS( AllocResult_get_pointer )( 
    const NS( AllocResult ) * const SIXTRL_RESTRICT result );

SIXTRL_STATIC SIXTRL_UINT64_T NS( AllocResult_get_offset )( 
    const NS( AllocResult ) * const SIXTRL_RESTRICT result );

SIXTRL_STATIC SIXTRL_UINT64_T NS( AllocResult_get_length )( 
    const NS( AllocResult ) * const SIXTRL_RESTRICT result );

SIXTRL_STATIC void NS(AllocResult_assign_ptr)( 
    NS(AllocResult)* SIXTRL_RESTRICT result, 
    unsigned char* SIXTRL_RESTRICT ptr );

SIXTRL_STATIC void NS(AllocResult_set_length)( 
    NS(AllocResult)* SIXTRL_RESTRICT result, SIXTRL_UINT64_T const length );

SIXTRL_STATIC void NS(AllocResult_set_offset)(
    NS(AllocResult)* SIXTRL_RESTRICT result, SIXTRL_UINT64_T const offset );

/* -------------------------------------------------------------------------- */

typedef struct NS( MemPool )
{
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT buffer;
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT begin_pos;

    SIXTRL_UINT64_T begin_offset;
    SIXTRL_UINT64_T capacity;
    SIXTRL_UINT64_T size;
    SIXTRL_UINT64_T chunk_size;
    
} NS(MemPool);

SIXTRL_STATIC NS( MemPool ) * NS( MemPool_preset )( 
    NS( MemPool ) * SIXTRL_RESTRICT pool );

SIXTRL_STATIC void NS( MemPool_init )( 
    NS( MemPool ) * SIXTRL_RESTRICT pool, SIXTRL_UINT64_T capacity,
    SIXTRL_UINT64_T const chunk_size );

/* ------------------------------------------------------------------------- */

void NS( MemPool_init_aligned)(
    NS( MemPool ) * SIXTRL_RESTRICT pool, SIXTRL_UINT64_T capacity,
    SIXTRL_UINT64_T const chunk_size,  SIXTRL_UINT64_T const begin_alignment );

SIXTRL_STATIC bool NS( MemPool_reserve )( 
    NS( MemPool ) * SIXTRL_RESTRICT pool, SIXTRL_UINT64_T new_capacity );

bool NS( MemPool_reserve_aligned )( 
    NS( MemPool ) * SIXTRL_RESTRICT pool,
    SIXTRL_UINT64_T const new_capacity, SIXTRL_UINT64_T const alignment );

bool NS(MemPool_clear_to_aligned_position)( 
    NS(MemPool)* SIXTRL_RESTRICT pool, SIXTRL_UINT64_T const alignment  );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC void NS( MemPool_free )( 
    NS( MemPool ) * SIXTRL_RESTRICT pool );

SIXTRL_STATIC void NS( MemPool_clear )( 
    NS( MemPool ) * SIXTRL_RESTRICT pool );

SIXTRL_STATIC bool NS( MemPool_is_empty )( 
    const NS( MemPool ) * const SIXTRL_RESTRICT pool );

SIXTRL_STATIC bool NS(MemPool_is_begin_aligned_with)( 
    const NS(MemPool) *const pool, SIXTRL_UINT64_T const alignment );

SIXTRL_STATIC SIXTRL_UINT64_T NS( MemPool_get_capacity )( 
    const NS( MemPool ) * const SIXTRL_RESTRICT pool );

SIXTRL_STATIC SIXTRL_UINT64_T NS( MemPool_get_buffer_capacity )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool );

SIXTRL_STATIC SIXTRL_UINT64_T NS( MemPool_get_size )( 
    const NS( MemPool ) * const SIXTRL_RESTRICT pool );

SIXTRL_STATIC SIXTRL_UINT64_T NS( MemPool_get_chunk_size )( 
    const NS( MemPool ) * const SIXTRL_RESTRICT pool );

SIXTRL_STATIC SIXTRL_UINT64_T NS( MemPool_get_begin_offset)( 
    const NS( MemPool ) *const SIXTRL_RESTRICT pool );

SIXTRL_STATIC SIXTRL_UINT64_T NS( MemPool_get_remaining_bytes )( 
    const NS( MemPool ) * const SIXTRL_RESTRICT pool );

SIXTRL_STATIC SIXTRL_UINT64_T NS( MemPool_get_next_begin_offset )( 
    const NS( MemPool ) * const pool, SIXTRL_UINT64_T block_alignment );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char* NS( MemPool_get_begin_pos )( 
    NS( MemPool ) * SIXTRL_RESTRICT pool );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char const* 
NS( MemPool_get_const_begin_pos )( 
    const NS( MemPool ) * const SIXTRL_RESTRICT pool );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char* 
NS( MemPool_get_next_begin_pointer )( 
    NS( MemPool ) * SIXTRL_RESTRICT pool, SIXTRL_UINT64_T block_alignment );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char* 
NS( MemPool_get_pointer_by_offset )( 
        NS( MemPool ) * SIXTRL_RESTRICT pool, SIXTRL_UINT64_T const offset );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char const* 
NS( MemPool_get_next_begin_const_pointer )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool, 
    SIXTRL_UINT64_T const alignment );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char const* 
NS( MemPool_get_const_pointer_by_offset )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool, 
    SIXTRL_UINT64_T const offset );

SIXTRL_STATIC void NS(MemPool_increment_size)(
    NS(MemPool)* SIXTRL_RESTRICT pool,
    SIXTRL_UINT64_T const new_size );

SIXTRL_STATIC NS( AllocResult ) NS( MemPool_append )( 
    NS( MemPool ) * SIXTRL_RESTRICT pool, 
    SIXTRL_UINT64_T const num_bytes );

SIXTRL_STATIC NS( AllocResult ) NS( MemPool_append_aligned )
( NS( MemPool ) * SIXTRL_RESTRICT pool,
    SIXTRL_UINT64_T const num_bytes, SIXTRL_UINT64_T const alignment );

/* ========================================================================= */
/* =====        Implementation of Inline Functions and Methods          ==== */
/* ========================================================================= */

SIXTRL_INLINE bool NS( AllocResult_valid )( const NS( AllocResult ) *
                              const SIXTRL_RESTRICT result )
{
    bool is_valid = false;

    if( result != 0 )
    {
        is_valid = ( ( result->p != 0 ) && 
                     ( result->length > ( SIXTRL_UINT64_T )0u ) );
    }

    return is_valid;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE bool NS(AllocResult_is_aligned)( 
    const NS(AllocResult) *const SIXTRL_RESTRICT result, 
    SIXTRL_SIZE_T const alignment )
{
    static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;    
    
    return ( 
        ( alignment > ZERO_SIZE ) && ( result != 0 ) && ( result->p != 0 ) && 
        ( ( ( ( uintptr_t )result->p ) % alignment ) == ZERO_SIZE ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS( AllocResult ) *
    NS( AllocResult_preset )( NS( AllocResult ) * SIXTRL_RESTRICT result )
{
    if( result != 0 )
    {
        result->p = 0;
        result->offset = ( SIXTRL_UINT64_T )0u;
        result->length = ( SIXTRL_UINT64_T )0u;
    }

    return result;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char* 
NS( AllocResult_get_pointer )( 
    const NS( AllocResult ) * const SIXTRL_RESTRICT result )
{
    return ( result != 0 ) ? result->p : 0;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_UINT64_T NS( AllocResult_get_offset )( 
    const NS( AllocResult ) * const SIXTRL_RESTRICT result )
{
    return ( result != 0 ) ? result->offset : ( SIXTRL_UINT64_T )0u;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_UINT64_T NS( AllocResult_get_length )( 
    const NS( AllocResult ) * const SIXTRL_RESTRICT result )
{
    return ( result != 0 ) ? result->length : ( SIXTRL_UINT64_T )0u;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(AllocResult_assign_ptr)( 
    NS(AllocResult)* SIXTRL_RESTRICT result, 
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT ptr )
{
    result->p = ptr;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(AllocResult_set_length)( 
    NS(AllocResult)* SIXTRL_RESTRICT result, SIXTRL_UINT64_T const length )
{
    result->length = length;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(AllocResult_set_offset)(
    NS(AllocResult)* SIXTRL_RESTRICT result, SIXTRL_UINT64_T const offset )
{
    result->offset = offset;
    return;
}

/* -------------------------------------------------------------------------- */

SIXTRL_INLINE NS( MemPool ) * NS( MemPool_preset )( 
    NS( MemPool ) * SIXTRL_RESTRICT pool )
{
    if( pool != 0 )
    {
        typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
        
        static SIXTRL_UINT64_T const ZERO = ( SIXTRL_UINT64_T )0u;

        pool->buffer        = ( g_ptr_uchar_t )0;
        pool->begin_pos     = ( g_ptr_uchar_t )0;

        pool->begin_offset  = ZERO;
        pool->capacity      = ZERO;
        pool->size          = ZERO;
        pool->chunk_size    = ZERO;
    }

    return pool;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS( MemPool_init )( 
    NS( MemPool ) * SIXTRL_RESTRICT pool,
    SIXTRL_UINT64_T capacity, SIXTRL_UINT64_T const chunk_size )
{
    NS( MemPool_init_aligned )( pool, capacity, chunk_size, chunk_size );
    return;
}

SIXTRL_INLINE bool NS( MemPool_reserve )( 
    NS( MemPool ) * SIXTRL_RESTRICT pool, SIXTRL_UINT64_T new_capacity )
{
    return NS(MemPool_reserve_aligned)(
        pool, new_capacity, NS(MemPool_get_chunk_size)( pool ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS( MemPool_free )( NS( MemPool ) * SIXTRL_RESTRICT pool )
{
    if( pool != 0 )
    {
        free( pool->buffer );
        NS( MemPool_preset )( pool );
    }

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS( MemPool_clear )( NS( MemPool ) * SIXTRL_RESTRICT pool )
{
    if( pool != 0 ) pool->size = (SIXTRL_UINT64_T)0u;
    return;
}

/* -------------------------------------------------------------------------- */

SIXTRL_INLINE bool NS( MemPool_is_empty )( 
    const NS( MemPool ) * const SIXTRL_RESTRICT pool )
{
    SIXTRL_ASSERT( NS( MemPool_get_const_begin_pos )( pool ) != 0 );
    return ( NS( MemPool_get_size )( pool ) == (SIXTRL_UINT64_T)0u );
}

/* -------------------------------------------------------------------------- */

SIXTRL_INLINE bool NS(MemPool_is_begin_aligned_with)( 
    const NS(MemPool) *const pool, SIXTRL_UINT64_T const alignment )
{
    return ( ( pool != 0 ) && 
             ( alignment > ( SIXTRL_UINT64_T )0u ) &&
             ( NS(MemPool_get_const_begin_pos)( pool ) != 0 ) &&
             ( ( ( ( uintptr_t )NS(MemPool_get_const_begin_pos)( pool ) ) %
                 alignment ) == ( SIXTRL_UINT64_T )0 ) );    
}

SIXTRL_INLINE SIXTRL_UINT64_T NS( MemPool_get_capacity )( 
    const NS( MemPool ) * const SIXTRL_RESTRICT pool )
{
    return ( pool != 0 ) ? pool->capacity : ( SIXTRL_UINT64_T )0u;
}

SIXTRL_INLINE SIXTRL_UINT64_T NS( MemPool_get_buffer_capacity )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool )
{
    return ( pool != 0 ) ? ( pool->capacity + pool->begin_offset ) 
                         : ( SIXTRL_UINT64_T )0u;
}

SIXTRL_INLINE SIXTRL_UINT64_T NS( MemPool_get_size )( 
    const NS( MemPool ) * const SIXTRL_RESTRICT pool )
{
    return ( pool != 0 ) ? ( pool->size ) : ( SIXTRL_UINT64_T )0u;
}

SIXTRL_INLINE SIXTRL_UINT64_T NS( MemPool_get_chunk_size )( 
    const NS( MemPool ) * const SIXTRL_RESTRICT pool )
{
    return ( pool != 0 ) ? ( pool->chunk_size ) : ( SIXTRL_UINT64_T )0u;
}

SIXTRL_INLINE SIXTRL_UINT64_T NS( MemPool_get_begin_offset)( 
    const NS( MemPool ) *const SIXTRL_RESTRICT pool )
{
    return ( pool != 0 ) ? ( pool->begin_offset ) : ( SIXTRL_UINT64_T )0u;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char* NS( MemPool_get_begin_pos )( 
    NS( MemPool ) * SIXTRL_RESTRICT pool )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
    return ( g_ptr_uchar_t )NS(MemPool_get_const_begin_pos)( pool );
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char const* 
NS( MemPool_get_const_begin_pos )( 
    const NS( MemPool ) * const SIXTRL_RESTRICT pool )
{
    return ( pool != 0 ) ? pool->begin_pos : 0;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_UINT64_T NS( MemPool_get_next_begin_offset )( 
    const NS( MemPool ) * const pool, SIXTRL_UINT64_T const alignment )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char const* g_ptr_uchar_t;    
    
    SIXTRL_STATIC SIXTRL_UINT64_T const ZERO = ( SIXTRL_UINT64_T )0u;
    
    SIXTRL_STATIC SIXTRL_UINT64_T const U64_MAX = 
        ( SIXTRL_UINT64_T )0xffffffffffffffff;
    
    SIXTRL_UINT64_T next_offset = U64_MAX;
    
    SIXTRL_UINT64_T const chunk_size = NS( MemPool_get_chunk_size )( pool );

    SIXTRL_UINT64_T const use_align = 
        NS(Alignment_calculate_common)( alignment, chunk_size );
        
    /* --------------------------------------------------------------------- */
    
    SIXTRL_ASSERT( ( chunk_size > ZERO ) && ( use_align >= chunk_size ) );
    
    if( pool != 0 )
    {
        SIXTRL_UINT64_T const current_size = NS( MemPool_get_size )( pool );
        g_ptr_uchar_t begin_pos = NS( MemPool_get_const_begin_pos )( pool );

        uintptr_t const current_offset_addr = 
            ( uintptr_t )( begin_pos + current_size );

        uintptr_t const addr_mod = current_offset_addr % use_align;

        uintptr_t const offset = 
            ( addr_mod != ZERO ) ? ( use_align - addr_mod ) : ZERO;

        SIXTRL_UINT64_T temp_next_offset = current_size;

        if( ( offset != ZERO ) && ( temp_next_offset < ( U64_MAX - offset ) ) )
        {
            temp_next_offset += offset;
        }
        
        if( temp_next_offset <= NS(MemPool_get_capacity)( pool ) )
        {
            next_offset = temp_next_offset;
        }
    }

    return next_offset;
}

/* -------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_UINT64_T NS( MemPool_get_remaining_bytes )( 
    const NS( MemPool ) * const SIXTRL_RESTRICT pool )
{
    SIXTRL_UINT64_T const capacity   = NS( MemPool_get_capacity )( pool );
    SIXTRL_UINT64_T const chunk_size = NS( MemPool_get_chunk_size )( pool );
    SIXTRL_UINT64_T const size       = NS( MemPool_get_size )( pool );

    assert( ( capacity >= size ) && ( chunk_size > (SIXTRL_UINT64_T)0u ) );
    return ( ( capacity - size ) / chunk_size ) * chunk_size;
}

/* -------------------------------------------------------------------------- */

SIXTRL_INLINE NS( AllocResult )
NS( MemPool_append )( NS( MemPool ) * SIXTRL_RESTRICT pool, 
                      SIXTRL_UINT64_T const num_bytes )
{
    return NS( MemPool_append_aligned )( pool, num_bytes, (SIXTRL_UINT64_T)1u );
}

/* -------------------------------------------------------------------------- */

SIXTRL_INLINE NS( AllocResult )
NS( MemPool_append_aligned )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                              SIXTRL_UINT64_T const num_bytes,
                              SIXTRL_UINT64_T const alignment )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
    
    static SIXTRL_UINT64_T const ZERO = ( SIXTRL_UINT64_T )0u;
    
    static SIXTRL_UINT64_T const U64_MAX = 
        ( SIXTRL_UINT64_T )0xffffffffffffffff;

    NS( AllocResult ) result;

    g_ptr_uchar_t begin_pos = NS( MemPool_get_begin_pos )( pool );
    
    SIXTRL_UINT64_T const next_offset = 
        NS( MemPool_get_next_begin_offset )( pool, alignment );

    SIXTRL_UINT64_T const chunk_size = NS( MemPool_get_chunk_size )( pool );
    SIXTRL_ASSERT( sizeof( unsigned char ) == (SIXTRL_UINT64_T)1u );

    NS( AllocResult_preset )( &result );

    if( ( begin_pos != 0 ) && ( num_bytes > ZERO ) &&
        ( next_offset != U64_MAX ) && ( chunk_size > ZERO ) )
    {
        SIXTRL_UINT64_T new_size = next_offset;
        SIXTRL_UINT64_T bytes_to_add = ( num_bytes / chunk_size ) * chunk_size;

        if( bytes_to_add < num_bytes ) bytes_to_add += chunk_size;
        SIXTRL_ASSERT( bytes_to_add >= num_bytes );

        if( new_size < ( U64_MAX - bytes_to_add ) )
        {
            new_size += bytes_to_add;

            if( new_size <= NS( MemPool_get_capacity )( pool ) )
            {
                begin_pos = begin_pos + next_offset;
                
                SIXTRL_ASSERT( ( ( ( uintptr_t )begin_pos ) % alignment ) 
                    == ZERO );

                result.p      = begin_pos;
                result.offset = next_offset;
                result.length = bytes_to_add;

                pool->size = new_size;
            }
        }
    }

    return result;
}

/* -------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char* 
NS( MemPool_get_pointer_by_offset )( 
    NS( MemPool ) *SIXTRL_RESTRICT pool, SIXTRL_UINT64_T const offset )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
    
    /* casting away const-ness is legal, so reuse the const-ptr version */
    return (g_ptr_uchar_t)NS( MemPool_get_const_pointer_by_offset )( 
        pool, offset );
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char const* 
NS( MemPool_get_const_pointer_by_offset )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool, 
    SIXTRL_UINT64_T const offset )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char const* g_ptr_uchar_t;
    
    g_ptr_uchar_t ptr = ( pool != 0 ) ? 
        NS( MemPool_get_const_begin_pos )( pool ) : 0;
    
    SIXTRL_ASSERT( 
        ( ( ptr != 0 ) && 
        ( NS( MemPool_get_chunk_size )( pool ) > (SIXTRL_UINT64_T)0u ) &&
        ( ( offset % NS( MemPool_get_chunk_size )( pool ) ) == 
          (SIXTRL_UINT64_T)0u ) ) || ( ptr == 0 ) );

    return ptr + offset;
}

SIXTRL_INLINE void NS(MemPool_increment_size)( 
    NS(MemPool)* SIXTRL_RESTRICT pool, SIXTRL_UINT64_T const new_size )
{
    assert( ( pool != 0 ) && ( new_size >= NS(MemPool_get_size)( pool ) ) &&
            ( new_size <= NS(MemPool_get_capacity)( pool ) ) );
    
    pool->size = new_size;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char* 
NS( MemPool_get_next_begin_pointer )( NS( MemPool ) * SIXTRL_RESTRICT pool,
    SIXTRL_UINT64_T const alignment )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;    
    return (g_ptr_uchar_t)NS( MemPool_get_next_begin_const_pointer )(
        pool, alignment );
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char const* 
NS( MemPool_get_next_begin_const_pointer )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool, 
    SIXTRL_UINT64_T const alignment )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char const* g_ptr_uchar_t;
    
    g_ptr_uchar_t begin_pos = NS( MemPool_get_const_begin_pos )( pool );
    
    if( begin_pos != 0 )
    {
        begin_pos = begin_pos + NS( MemPool_get_next_begin_offset )(
                                    pool, alignment );
    }

    return begin_pos;
}

#if !defined( _GPUCODE )

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_MEMORY_POOL_H__ */

/* end: sixtracklib/common/mem_pool.h */
