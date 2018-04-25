#include "sixtracklib/common/mem_pool.h"
#include "sixtracklib/_impl/definitions.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/common/details/tools.h"


/* ========================================================================= */

extern bool NS( AllocResult_valid )( const NS( AllocResult ) *
                                     const SIXTRL_RESTRICT result );

extern bool NS(AllocResult_is_aligned)( 
    const NS(AllocResult) *const SIXTRL_RESTRICT result, 
    SIXTRL_SIZE_T const alignment );

extern NS( AllocResult ) *
    NS( AllocResult_preset )( NS( AllocResult ) * SIXTRL_RESTRICT result );

extern unsigned char* NS( AllocResult_get_pointer )(
    const NS( AllocResult ) * const SIXTRL_RESTRICT result );

extern SIXTRL_UINT64_T NS( AllocResult_get_offset )( const NS( AllocResult ) *
                                              const SIXTRL_RESTRICT result );

extern SIXTRL_UINT64_T NS( AllocResult_get_length )( const NS( AllocResult ) *
                                              const SIXTRL_RESTRICT result );

/* ------------------------------------------------------------------------- */

extern NS( MemPool ) *
    NS( MemPool_preset )( NS( MemPool ) * SIXTRL_RESTRICT pool );

extern void NS( MemPool_init )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                SIXTRL_SIZE_T const capacity,
                                SIXTRL_SIZE_T const chunk_size );

extern void NS( MemPool_free )( NS( MemPool ) * SIXTRL_RESTRICT pool );

extern void NS( MemPool_clear )( NS( MemPool ) * SIXTRL_RESTRICT pool );

extern bool NS( MemPool_reserve )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                   SIXTRL_SIZE_T const new_capacity );

extern void NS( MemPool_clone )( 
    NS( MemPool ) * SIXTRL_RESTRICT dest, 
    const NS( MemPool ) * const SIXTRL_RESTRICT source );

extern bool NS( MemPool_is_empty )( 
    const NS( MemPool ) * const SIXTRL_RESTRICT pool );

extern SIXTRL_SIZE_T NS( MemPool_get_capacity )( 
    const NS( MemPool ) * const SIXTRL_RESTRICT pool );

extern SIXTRL_SIZE_T NS( MemPool_get_size )( 
    const NS( MemPool ) * const SIXTRL_RESTRICT pool );

extern SIXTRL_SIZE_T NS( MemPool_get_remaining_bytes )( 
    const NS( MemPool ) * const SIXTRL_RESTRICT pool );

extern SIXTRL_UINT64_T NS( MemPool_get_next_begin_offset )( 
    const NS( MemPool ) * const pool, SIXTRL_SIZE_T const alignment );

extern unsigned char* NS( MemPool_get_buffer )( NS( MemPool ) *
                                                SIXTRL_RESTRICT pool );

extern unsigned char*
    NS( MemPool_get_pointer_by_offset )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                         SIXTRL_UINT64_T const offset );

extern unsigned char*
    NS( MemPool_get_next_begin_pointer )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                          SIXTRL_SIZE_T const alignment );

extern unsigned char const* NS( MemPool_get_const_buffer )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool );

extern unsigned char const* NS( MemPool_get_const_pointer_by_offset )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool, 
    SIXTRL_UINT64_T const offset );

extern unsigned char const* NS( MemPool_get_next_begin_const_pointer )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool, 
    SIXTRL_SIZE_T const block_alignment );

extern NS( AllocResult ) NS( MemPool_append )( 
    NS( MemPool ) * SIXTRL_RESTRICT pool, SIXTRL_SIZE_T const num_bytes );

extern NS( AllocResult ) NS( MemPool_append_aligned )( 
    NS( MemPool ) * SIXTRL_RESTRICT pool, SIXTRL_SIZE_T const num_bytes,
    SIXTRL_SIZE_T const block_alignment );

/* ------------------------------------------------------------------------- */

static void NS( MemPool_set_capacity )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                        SIXTRL_SIZE_T const new_capacity );

static void NS( MemPool_set_size )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                    SIXTRL_SIZE_T const new_size );

static void NS( MemPool_set_chunk_size )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                          SIXTRL_SIZE_T const new_chunk_size );

static void NS( MemPool_set_buffer )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                      unsigned char* new_buffer );

/* ========================================================================= */

bool NS( AllocResult_valid )( const NS( AllocResult ) *
                              const SIXTRL_RESTRICT result )
{
    bool is_valid = false;

    if( result != 0 )
    {
        is_valid = ( ( result->p != 0 ) && ( result->length > UINT64_C( 0 ) ) );
    }

    return is_valid;
}

/* ------------------------------------------------------------------------- */

bool NS(AllocResult_is_aligned)( 
    const NS(AllocResult) *const SIXTRL_RESTRICT result, 
    SIXTRL_SIZE_T const alignment )
{
    static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;    
    
    return ( 
        ( alignment > ZERO_SIZE ) && ( result != 0 ) && ( result->p != 0 ) && 
        ( ( ( ( uintptr_t )result->p ) % alignment ) == ZERO_SIZE ) );
}

/* ------------------------------------------------------------------------- */

NS( AllocResult ) *
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

unsigned char* NS( AllocResult_get_pointer )( const NS( AllocResult ) *
                                              const SIXTRL_RESTRICT result )
{
    return ( result != 0 ) ? result->p : 0;
}

/* ------------------------------------------------------------------------- */

uint64_t NS( AllocResult_get_offset )( const NS( AllocResult ) *
                                       const SIXTRL_RESTRICT result )
{
    return ( result != 0 ) ? result->offset : ( SIXTRL_UINT64_T )0u;
}

/* ------------------------------------------------------------------------- */

uint64_t NS( AllocResult_get_length )( const NS( AllocResult ) *
                                       const SIXTRL_RESTRICT result )
{
    return ( result != 0 ) ? result->length : ( SIXTRL_UINT64_T )0u;
}

/* ========================================================================= */

NS( MemPool ) * NS( MemPool_preset )( NS( MemPool ) * SIXTRL_RESTRICT pool )
{
    if( pool != 0 )
    {
        static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;

        NS( MemPool_set_buffer )( pool, 0 );
        NS( MemPool_set_capacity )( pool, ZERO_SIZE );
        NS( MemPool_set_size )( pool, ZERO_SIZE );
        NS( MemPool_set_chunk_size )( pool, ZERO_SIZE );
    }

    return pool;
}

/* -------------------------------------------------------------------------- */

void NS( MemPool_init )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                         SIXTRL_SIZE_T capacity,
                         SIXTRL_SIZE_T const chunk_size )
{
    static size_t const ZERO_SIZE = (SIXTRL_SIZE_T)0u;
    NS( MemPool_preset( pool ) );

    if( ( pool != 0 ) && ( chunk_size > (SIXTRL_SIZE_T)0u ) )
    {
        unsigned char* new_buffer = 0;

        SIXTRL_SIZE_T const requested_capacity =
            ( capacity > chunk_size ) ? capacity : chunk_size;

        SIXTRL_SIZE_T const num_chunks = requested_capacity / chunk_size;
        SIXTRL_SIZE_T const calc_capacity = num_chunks * chunk_size;

        capacity = ( calc_capacity < requested_capacity )
                       ? calc_capacity + chunk_size
                       : calc_capacity;

        /* use sizeof( unsigned char ) because of some theoretical ambiguity
         * in C++ which allows for sizeof( unsigned char ) > sizeof( char ) */
        new_buffer =
            (unsigned char*)malloc( capacity * sizeof( unsigned char ) );

        if( new_buffer != 0 )
        {
            NS( MemPool_set_buffer )( pool, new_buffer );
            NS( MemPool_set_capacity )( pool, capacity );
            NS( MemPool_set_size )( pool, ZERO_SIZE );
            NS( MemPool_set_chunk_size )( pool, chunk_size );
        }
    }

    return;
}

/* -------------------------------------------------------------------------- */

void NS( MemPool_free )( NS( MemPool ) * SIXTRL_RESTRICT pool )
{
    if( pool != 0 )
    {
        free( pool->buffer );
        NS( MemPool_preset )( pool );
    }

    return;
}

/* -------------------------------------------------------------------------- */

void NS( MemPool_clear )( NS( MemPool ) * SIXTRL_RESTRICT pool )
{
    if( pool != 0 )
    {
        pool->size = (SIXTRL_SIZE_T)0u;
    }

    return;
}

bool NS(MemPool_clear_to_aligned_position)( 
    NS(MemPool)* SIXTRL_RESTRICT pool, SIXTRL_SIZE_T const alignment )
{
    bool success = false;
    
    NS(MemPool_clear)( pool );
    
    if( ( alignment > 0u ) && ( pool->chunk_size > 0u ) &&
        ( pool->buffer != 0 ) )
    {
        SIXTRL_SIZE_T const use_alignment = NS(least_common_multiple)( 
            alignment, pool->chunk_size );
        
        if( use_alignment == 0u ) return false;
        
        uintptr_t const buffer_addr = ( uintptr_t )pool->buffer;
        uintptr_t const buffer_addr_mod = ( buffer_addr % use_alignment );
        
        if( buffer_addr_mod != 0u )
        {
            SIXTRL_SIZE_T const offset = use_alignment - buffer_addr_mod;
            
            if( ( pool->size + offset ) < pool->capacity )
            {
                pool->size += offset;                
            }
        }
        
        success = ( ( ( ( uintptr_t )NS(MemPool_get_next_begin_pointer)( 
                pool, use_alignment ) ) % use_alignment ) == 0u );
    }
    
    return success;
}

/* -------------------------------------------------------------------------- */

bool NS( MemPool_reserve )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                            SIXTRL_SIZE_T const new_capacity )
{
    bool has_been_changed = false;

    static SIXTRL_SIZE_T ZERO_SIZE = (SIXTRL_SIZE_T)0u;

    SIXTRL_SIZE_T const current_capacity = NS( MemPool_get_capacity( pool ) );
    SIXTRL_SIZE_T const chunk_size = NS( MemPool_get_chunk_size( pool ) );

    if( ( new_capacity > current_capacity ) && ( chunk_size > ZERO_SIZE ) )
    {
        unsigned char* current_buffer = NS( MemPool_get_buffer )( pool );
        SIXTRL_SIZE_T const current_size = NS( MemPool_get_size( pool ) );

        NS( MemPool_preset( pool ) );
        assert( NS( MemPool_get_buffer( pool ) ) == 0 );

        NS( MemPool_init )
        ( pool, new_capacity, chunk_size );

        if( NS( MemPool_get_buffer( pool ) != 0 ) )
        {
            assert( NS( MemPool_get_capacity )( pool ) >= new_capacity );

            if( current_buffer != 0 )
            {
                if( current_size > ZERO_SIZE )
                {
                    memcpy( NS( MemPool_get_buffer( pool ) ),
                            current_buffer, current_size );
                    
                    NS( MemPool_set_size )( pool, current_size );
                }

                free( current_buffer );
                current_buffer = 0;
            }

            has_been_changed = true;
        } else
        {
            /* Rollback change as allocation was not successful! */

            NS( MemPool_set_buffer )( pool, current_buffer );
            NS( MemPool_set_capacity )( pool, current_capacity );
            NS( MemPool_set_size )( pool, current_size );
            NS( MemPool_set_chunk_size )( pool, chunk_size );
        }
    }

    return has_been_changed;
}

/* -------------------------------------------------------------------------- */

void NS( MemPool_clone )( NS( MemPool ) * SIXTRL_RESTRICT dest,
                          const NS( MemPool ) * const SIXTRL_RESTRICT source )
{
    NS( MemPool_free )( dest );

    unsigned char const* source_buffer =
        NS( MemPool_get_const_buffer )( source );

    if( source_buffer != 0 )
    {
        unsigned char* dest_buffer = 0;

        SIXTRL_SIZE_T const source_size = NS( MemPool_get_size )( source );
        SIXTRL_SIZE_T const source_capacity = NS( MemPool_get_capacity( source ) );
        SIXTRL_SIZE_T const source_chunk_size = NS( MemPool_get_size( source ) );

        assert( ( source_capacity >= source_size ) &&
                ( source_chunk_size > (SIXTRL_SIZE_T)0u ) );

        NS( MemPool_init )
        ( dest, source_capacity, source_chunk_size );
        dest_buffer = NS( MemPool_get_buffer )( dest );

        if( ( dest_buffer != 0 ) && ( source_size > (SIXTRL_SIZE_T)0u ) )
        {
            memcpy( dest_buffer, source_buffer, source_size );
            NS( MemPool_set_size )( dest, source_size );
        }
    }

    return;
}

/* -------------------------------------------------------------------------- */

bool NS( MemPool_is_empty )( const NS( MemPool ) * const SIXTRL_RESTRICT pool )
{
    assert( NS( MemPool_get_const_buffer )( pool ) != 0 );
    return ( NS( MemPool_get_size )( pool ) == (SIXTRL_SIZE_T)0u );
}

/* -------------------------------------------------------------------------- */

SIXTRL_SIZE_T NS( MemPool_get_capacity )( const NS( MemPool ) *
                                   const SIXTRL_RESTRICT pool )
{
    return ( pool != 0 ) ? pool->capacity : (SIXTRL_SIZE_T)0u;
}

/* -------------------------------------------------------------------------- */

SIXTRL_SIZE_T NS( MemPool_get_size )( const NS( MemPool ) *
                               const SIXTRL_RESTRICT pool )
{
    return ( pool != 0 ) ? pool->size : (SIXTRL_SIZE_T)0u;
}

/* -------------------------------------------------------------------------- */

SIXTRL_SIZE_T NS( MemPool_get_chunk_size )( const NS( MemPool ) *
                                     const SIXTRL_RESTRICT pool )
{
    return ( pool != 0 ) ? pool->chunk_size : (SIXTRL_SIZE_T)0u;
}

/* -------------------------------------------------------------------------- */

SIXTRL_UINT64_T NS( MemPool_get_next_begin_offset )( 
    const NS( MemPool ) * const pool, SIXTRL_SIZE_T const alignment )
{
    static size_t const ZERO_SIZE = (size_t)0u;
    SIXTRL_UINT64_T next_offset = UINT64_MAX;
    SIXTRL_SIZE_T use_alignment = alignment;
    
    unsigned char const* ptr_begin = NS( MemPool_get_const_buffer )( pool );
    SIXTRL_SIZE_T const chunk_size = NS( MemPool_get_chunk_size )( pool );

    if( use_alignment == (SIXTRL_SIZE_T)1u )
    {
        use_alignment = chunk_size;
    }
    
    if( ( use_alignment  != chunk_size ) &&
        ( ( use_alignment < chunk_size ) ||
          ( ( use_alignment > chunk_size ) &&
            ( ( use_alignment % chunk_size ) != ZERO_SIZE ) ) ) )
    {
        use_alignment = NS(least_common_multiple)( use_alignment, chunk_size );
    }

    /* --------------------------------------------------------------------- */
    
    assert( ( use_alignment >= alignment ) &&
            ( use_alignment >= chunk_size ) &&
            ( ( use_alignment % chunk_size ) == ZERO_SIZE ) &&
            ( ( alignment == ZERO_SIZE ) ||
              ( ( use_alignment % alignment ) == ZERO_SIZE ) ) );
    
    if( ( pool != 0 ) && ( chunk_size > ZERO_SIZE ) )
    {
        SIXTRL_SIZE_T const current_size = NS( MemPool_get_size )( pool );

        uintptr_t const current_offset_addr =
            ( uintptr_t )( ptr_begin + current_size );

        uintptr_t const addr_align_modulo = current_offset_addr % use_alignment;

        uintptr_t const addr_align_offset = ( addr_align_modulo != ZERO_SIZE )
                ? ( use_alignment - addr_align_modulo ) : ZERO_SIZE;

        next_offset = current_size;

        if( ( addr_align_offset != ZERO_SIZE ) &&
            ( next_offset < ( (SIXTRL_SIZE_T)UINT64_MAX - use_alignment ) ) )
        {
            next_offset += addr_align_offset;
        }
    }

    return next_offset;
}

/* -------------------------------------------------------------------------- */

size_t NS( MemPool_get_remaining_bytes )( const NS( MemPool ) *
                                          const SIXTRL_RESTRICT pool )
{
    #if !defined( NDEBUG ) 
    static SIXTRL_SIZE_T ZERO_SIZE = (SIXTRL_SIZE_T)0u;
    #endif /* !defined( NDEBUG ) */
    
    SIXTRL_SIZE_T const capacity = NS( MemPool_get_capacity )( pool );
    SIXTRL_SIZE_T const chunk_size = NS( MemPool_get_chunk_size )( pool );
    SIXTRL_SIZE_T const size = NS( MemPool_get_size )( pool );

    assert( ( capacity >= size ) && ( chunk_size > ZERO_SIZE ) &&
            ( ( size % chunk_size ) == ZERO_SIZE ) );

    return ( ( capacity - size ) / chunk_size ) * chunk_size;
}

/* -------------------------------------------------------------------------- */

unsigned char* NS( MemPool_get_buffer )( NS( MemPool ) * SIXTRL_RESTRICT pool )
{
    return ( pool != 0 ) ? ( pool->buffer ) : 0;
}

unsigned char const* NS( MemPool_get_const_buffer )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool )
{
    return ( pool != 0 ) ? ( pool->buffer ) : 0;
}

/* -------------------------------------------------------------------------- */

NS( AllocResult )
NS( MemPool_append )( NS( MemPool ) * SIXTRL_RESTRICT pool, 
                      SIXTRL_SIZE_T const num_bytes )
{
    return NS( MemPool_append_aligned )( pool, num_bytes, (SIXTRL_SIZE_T)1u );
}

/* -------------------------------------------------------------------------- */

NS( AllocResult )
NS( MemPool_append_aligned )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                              SIXTRL_SIZE_T const num_bytes,
                              SIXTRL_SIZE_T const alignment )
{
    static SIXTRL_SIZE_T const ZERO_SIZE = (SIXTRL_SIZE_T)0u;

    NS( AllocResult ) result;

    unsigned char* ptr_begin = NS( MemPool_get_buffer )( pool );
    
    SIXTRL_UINT64_T const next_offset =
        NS( MemPool_get_next_begin_offset )( pool, alignment );

    SIXTRL_SIZE_T const chunk_size = NS( MemPool_get_chunk_size )( pool );
    assert( sizeof( unsigned char ) == (SIXTRL_SIZE_T)1u );

    NS( AllocResult_preset )( &result );

    if( ( ptr_begin != 0 ) && ( num_bytes > ZERO_SIZE ) &&
        ( next_offset != UINT64_MAX ) && ( chunk_size > ZERO_SIZE ) )
    {
        SIXTRL_SIZE_T new_size = next_offset;
        SIXTRL_SIZE_T bytes_to_add = ( num_bytes / chunk_size ) * chunk_size;

        if( bytes_to_add < num_bytes ) bytes_to_add += chunk_size;
        assert( bytes_to_add >= num_bytes );

        if( new_size < ( ( SIXTRL_SIZE_T )(UINT64_MAX)-bytes_to_add ) )
        {
            new_size += bytes_to_add;

            if( new_size <= NS( MemPool_get_capacity )( pool ) )
            {
                ptr_begin = ptr_begin + next_offset;
                assert( ( (uintptr_t)ptr_begin % alignment ) == ZERO_SIZE );

                result.p = ptr_begin;
                result.offset = next_offset;
                result.length = bytes_to_add;

                NS( MemPool_set_size )
                ( pool, new_size );
            }
        }
    }

    return result;
}

/* -------------------------------------------------------------------------- */

unsigned char* NS( MemPool_get_pointer_by_offset )( NS( MemPool ) *
                                                        SIXTRL_RESTRICT pool,
                                                    SIXTRL_UINT64_T const offset )
{

    /* casting away const-ness is legal, so reuse the const-ptr version */
    return (unsigned char*)NS( MemPool_get_const_pointer_by_offset )( pool,
                                                                      offset );
}

unsigned char const* NS( MemPool_get_const_pointer_by_offset )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool, SIXTRL_UINT64_T const offset )
{
    unsigned char const* ptr =
        ( pool != 0 ) ? NS( MemPool_get_const_buffer )( pool ) : 0;
    
    assert( ( ( ptr != 0 ) && 
              ( NS( MemPool_get_chunk_size )( pool ) > (SIXTRL_SIZE_T)0u ) &&
              ( ( offset % NS( MemPool_get_chunk_size )( pool ) ) == 
                (SIXTRL_SIZE_T)0u ) ) || ( ptr == 0 ) );

    return ptr + offset;
}

/* ------------------------------------------------------------------------- */

unsigned char* NS( MemPool_get_next_begin_pointer )( NS( MemPool ) *
                                                         SIXTRL_RESTRICT pool,
                                                     SIXTRL_SIZE_T const alignment )
{
    return (unsigned char*)NS( MemPool_get_next_begin_const_pointer )(
        pool, alignment );
}

unsigned char const* NS( MemPool_get_next_begin_const_pointer )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool, SIXTRL_SIZE_T const alignment )
{
    unsigned char const* ptr_begin = NS( MemPool_get_const_buffer )( pool );
    if( ptr_begin != 0 )
    {
        ptr_begin = ptr_begin + NS( MemPool_get_next_begin_offset )(
                                    pool, alignment );
    }

    return ptr_begin;
}

/* ------------------------------------------------------------------------- */

void NS( MemPool_set_capacity )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                 SIXTRL_SIZE_T const new_capacity )
{
    assert( pool != 0 );
    pool->capacity = new_capacity;
    return;
}

/* -------------------------------------------------------------------------- */

void NS( MemPool_set_size )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                             SIXTRL_SIZE_T const new_size )
{
    assert( pool != 0 );
    pool->size = new_size;
    return;
}

/* -------------------------------------------------------------------------- */

void NS( MemPool_set_chunk_size )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                   SIXTRL_SIZE_T const new_chunk_size )
{
    assert( pool != 0 );
    pool->chunk_size = new_chunk_size;
    return;
}

/* -------------------------------------------------------------------------- */

void NS( MemPool_set_buffer )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                               unsigned char* new_buffer )
{
    assert( pool != 0 );
    pool->buffer = new_buffer;
    return;
}

/* ========================================================================= */

/* end: sixtracklib/common/details/mem_pool.c */
