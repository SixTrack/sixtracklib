#include "sixtracklib/common/details/mem_pool.h"

#include "sixtracklib/_impl/namespace_begin.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/common/restrict.h"

/* ========================================================================= */

extern bool NS( AllocResult_valid )( const NS( AllocResult ) *
                                     const SIXTRL_RESTRICT result );

extern NS( AllocResult ) *
    NS( AllocResult_preset )( NS( AllocResult ) * SIXTRL_RESTRICT result );

extern unsigned char* NS( AllocResult_get_pointer )(
    const NS( AllocResult ) * const SIXTRL_RESTRICT result );
extern uint64_t NS( AllocResult_get_offset )( const NS( AllocResult ) *
                                              const SIXTRL_RESTRICT result );

extern uint64_t NS( AllocResult_get_length )( const NS( AllocResult ) *
                                              const SIXTRL_RESTRICT result );

/* ------------------------------------------------------------------------- */

extern NS( MemPool ) *
    NS( MemPool_preset )( NS( MemPool ) * SIXTRL_RESTRICT pool );

extern void NS( MemPool_init )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                size_t capacity,
                                size_t chunk_size );

extern void NS( MemPool_free )( NS( MemPool ) * SIXTRL_RESTRICT pool );

extern void NS( MemPool_clear )( NS( MemPool ) * SIXTRL_RESTRICT pool );

extern bool NS( MemPool_reserve )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                   size_t new_capacity );

extern void NS( MemPool_clone )( NS( MemPool ) * SIXTRL_RESTRICT dest,
                                 const NS( MemPool ) *
                                     const SIXTRL_RESTRICT source );

extern bool NS( MemPool_is_empty )( const NS( MemPool ) *
                                    const SIXTRL_RESTRICT pool );

extern size_t NS( MemPool_get_capacity )( const NS( MemPool ) *
                                          const SIXTRL_RESTRICT pool );

extern size_t NS( MemPool_get_size )( const NS( MemPool ) *
                                      const SIXTRL_RESTRICT pool );

extern size_t NS( MemPool_get_remaining_bytes )( const NS( MemPool ) *
                                                 const SIXTRL_RESTRICT pool );

extern uint64_t NS( MemPool_get_next_begin_offset )( const NS( MemPool ) *
                                                         const pool,
                                                     size_t block_alignment );

extern unsigned char* NS( MemPool_get_buffer )( NS( MemPool ) *
                                                SIXTRL_RESTRICT pool );

extern unsigned char*
    NS( MemPool_get_pointer_by_offset )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                         uint64_t offset );

extern unsigned char*
    NS( MemPool_get_next_begin_pointer )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                          size_t block_alignment );

extern unsigned char const* NS( MemPool_get_const_buffer )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool );

extern unsigned char const* NS( MemPool_get_const_pointer_by_offset )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool, uint64_t offset );

extern unsigned char const* NS( MemPool_get_next_begin_const_pointer )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool, size_t block_alignment );

extern NS( AllocResult )
    NS( MemPool_append )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                          size_t num_bytes );

extern NS( AllocResult )
    NS( MemPool_append_aligned )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                  size_t num_bytes,
                                  size_t block_alignment );

/* ------------------------------------------------------------------------- */

static void NS( MemPool_set_capacity )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                        size_t new_capacity );
static void NS( MemPool_set_size )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                    size_t new_size );
static void NS( MemPool_set_chunk_size )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                          size_t new_chunk_size );
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

NS( AllocResult ) *
    NS( AllocResult_preset )( NS( AllocResult ) * SIXTRL_RESTRICT result )
{
    if( result != 0 )
    {
        result->p = 0;
        result->offset = UINT64_C( 0 );
        result->length = UINT64_C( 0 );
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
    return ( result != 0 ) ? result->offset : UINT64_C( 0 );
}

/* ------------------------------------------------------------------------- */

uint64_t NS( AllocResult_get_length )( const NS( AllocResult ) *
                                       const SIXTRL_RESTRICT result )
{
    return ( result != 0 ) ? result->length : UINT64_C( 0 );
}

/* ========================================================================= */

NS( MemPool ) * NS( MemPool_preset )( NS( MemPool ) * SIXTRL_RESTRICT pool )
{
    if( pool != 0 )
    {
        static size_t const ZERO_SIZE = (size_t)0u;

        NS( MemPool_set_buffer )( pool, 0 );
        NS( MemPool_set_capacity )( pool, ZERO_SIZE );
        NS( MemPool_set_size )( pool, ZERO_SIZE );
        NS( MemPool_set_chunk_size )( pool, ZERO_SIZE );
    }

    return pool;
}

/* -------------------------------------------------------------------------- */

void NS( MemPool_init )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                         size_t capacity,
                         size_t chunk_size )
{
    static size_t const ZERO_SIZE = (size_t)0u;
    NS( MemPool_preset( pool ) );

    if( ( pool != 0 ) && ( chunk_size > (size_t)0u ) )
    {
        unsigned char* new_buffer = 0;

        size_t const requested_capacity =
            ( capacity > chunk_size ) ? capacity : chunk_size;

        size_t const num_chunks = requested_capacity / chunk_size;
        size_t const calc_capacity = num_chunks * chunk_size;

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
        NS( MemPool_preset )
        ( pool );
    }

    return;
}

/* -------------------------------------------------------------------------- */

void NS( MemPool_clear )( NS( MemPool ) * SIXTRL_RESTRICT pool )
{
    if( pool != 0 )
    {
        pool->size = (size_t)0u;
    }

    return;
}

/* -------------------------------------------------------------------------- */

bool NS( MemPool_reserve )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                            size_t new_capacity )
{
    bool has_been_changed = false;

    static size_t ZERO_SIZE = (size_t)0u;

    size_t const current_capacity = NS( MemPool_get_capacity( pool ) );
    size_t const chunk_size = NS( MemPool_get_chunk_size( pool ) );

    if( ( new_capacity > current_capacity ) && ( chunk_size > ZERO_SIZE ) )
    {
        unsigned char* current_buffer = NS( MemPool_get_buffer )( pool );
        size_t const current_size = NS( MemPool_get_size( pool ) );

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
                            current_buffer,
                            current_size );
                    NS( MemPool_set_size )
                    ( pool, current_size );
                }

                free( current_buffer );
                current_buffer = 0;
            }

            has_been_changed = true;
        } else
        {
            /* Rollback change as allocation was not successful! */

            NS( MemPool_set_buffer )
            ( pool, current_buffer );
            NS( MemPool_set_capacity )
            ( pool, current_capacity );
            NS( MemPool_set_size )
            ( pool, current_size );
            NS( MemPool_set_chunk_size )
            ( pool, chunk_size );
        }
    }

    return has_been_changed;
}

/* -------------------------------------------------------------------------- */

void NS( MemPool_clone )( NS( MemPool ) * SIXTRL_RESTRICT dest,
                          const NS( MemPool ) * const SIXTRL_RESTRICT source )
{
    NS( MemPool_free )
    ( dest );

    unsigned char const* source_buffer =
        NS( MemPool_get_const_buffer )( source );

    if( source_buffer != 0 )
    {
        unsigned char* dest_buffer = 0;

        size_t const source_size = NS( MemPool_get_size )( source );
        size_t const source_capacity = NS( MemPool_get_capacity( source ) );
        size_t const source_chunk_size = NS( MemPool_get_size( source ) );

        assert( ( source_capacity >= source_size ) &&
                ( source_chunk_size > (size_t)0u ) );

        NS( MemPool_init )
        ( dest, source_capacity, source_chunk_size );
        dest_buffer = NS( MemPool_get_buffer )( dest );

        if( ( dest_buffer != 0 ) && ( source_size > (size_t)0u ) )
        {
            memcpy( dest_buffer, source_buffer, source_size );
            NS( MemPool_set_size )
            ( dest, source_size );
        }
    }

    return;
}

/* -------------------------------------------------------------------------- */

bool NS( MemPool_is_empty )( const NS( MemPool ) * const SIXTRL_RESTRICT pool )
{
    assert( NS( MemPool_get_const_buffer )( pool ) != 0 );
    return ( NS( MemPool_get_size )( pool ) == (size_t)0u );
}

/* -------------------------------------------------------------------------- */

size_t NS( MemPool_get_capacity )( const NS( MemPool ) *
                                   const SIXTRL_RESTRICT pool )
{
    return ( pool != 0 ) ? pool->capacity : (size_t)0u;
}

/* -------------------------------------------------------------------------- */

size_t NS( MemPool_get_size )( const NS( MemPool ) *
                               const SIXTRL_RESTRICT pool )
{
    return ( pool != 0 ) ? pool->size : (size_t)0u;
}

/* -------------------------------------------------------------------------- */

size_t NS( MemPool_get_chunk_size )( const NS( MemPool ) *
                                     const SIXTRL_RESTRICT pool )
{
    return ( pool != 0 ) ? pool->chunk_size : (size_t)0u;
}

/* -------------------------------------------------------------------------- */

uint64_t NS( MemPool_get_next_begin_offset )( const NS( MemPool ) * const pool,
                                              size_t alignment )
{
    uint64_t next_offset = UINT64_MAX;

    static size_t const ZERO_SIZE = (size_t)0u;

    unsigned char const* ptr_begin = NS( MemPool_get_const_buffer )( pool );
    size_t const chunk_size = NS( MemPool_get_chunk_size )( pool );

    if( alignment == (size_t)1u )
        alignment = chunk_size;

    if( ( pool != 0 ) && ( chunk_size > ZERO_SIZE ) &&
        ( ( alignment % chunk_size ) == ZERO_SIZE ) )
    {
        size_t const current_size = NS( MemPool_get_size )( pool );

        uintptr_t const current_offset_addr =
            ( uintptr_t )( ptr_begin + current_size );

        uintptr_t const addr_align_modulo = current_offset_addr % alignment;

        uintptr_t const addr_align_offset =
            ( addr_align_modulo != ZERO_SIZE )
                ? ( alignment - addr_align_modulo )
                : ZERO_SIZE;

        next_offset = current_size;

        if( ( addr_align_offset != ZERO_SIZE ) &&
            ( next_offset < ( (size_t)UINT64_MAX - alignment ) ) )
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
    static size_t ZERO_SIZE = (size_t)0u;
    size_t const capacity = NS( MemPool_get_capacity )( pool );
    size_t const chunk_size = NS( MemPool_get_chunk_size )( pool );
    size_t const size = NS( MemPool_get_size )( pool );

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
NS( MemPool_append )( NS( MemPool ) * SIXTRL_RESTRICT pool, size_t num_bytes )
{
    return NS( MemPool_append_aligned )( pool, num_bytes, (size_t)1u );
}

/* -------------------------------------------------------------------------- */

NS( AllocResult )
NS( MemPool_append_aligned )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                              size_t num_bytes,
                              size_t alignment )
{
    static size_t const ZERO_SIZE = (size_t)0u;

    NS( AllocResult )
    result;

    unsigned char* ptr_begin = NS( MemPool_get_buffer )( pool );
    uint64_t const next_offset =
        NS( MemPool_get_next_begin_offset )( pool, alignment );

    size_t const chunk_size = NS( MemPool_get_chunk_size )( pool );
    assert( sizeof( unsigned char ) == (size_t)1u );

    NS( AllocResult_preset )
    ( &result );

    if( ( ptr_begin != 0 ) && ( num_bytes > ZERO_SIZE ) &&
        ( next_offset != UINT64_MAX ) && ( chunk_size > ZERO_SIZE ) )
    {
        size_t new_size = next_offset;
        size_t bytes_to_add = ( num_bytes / chunk_size ) * chunk_size;

        if( bytes_to_add < num_bytes )
            bytes_to_add += chunk_size;
        assert( bytes_to_add >= num_bytes );

        if( new_size < ( ( size_t )(UINT64_MAX)-bytes_to_add ) )
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
                                                    uint64_t offset )
{

    /* casting away const-ness is legal, so reuse the const-ptr version */
    return (unsigned char*)NS( MemPool_get_const_pointer_by_offset )( pool,
                                                                      offset );
}

unsigned char const* NS( MemPool_get_const_pointer_by_offset )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool, uint64_t offset )
{

    static size_t const ZERO_SIZE = (size_t)0u;

    unsigned char const* ptr =
        ( pool != 0 ) ? NS( MemPool_get_const_buffer )( pool ) : 0;
    size_t const chunk_size = NS( MemPool_get_chunk_size )( pool );

    assert( ( ( ptr != 0 ) && ( chunk_size > ZERO_SIZE ) &&
              ( ( offset % chunk_size ) == ZERO_SIZE ) ) ||
            ( ptr == 0 ) );

    return ptr + offset;
}

/* ------------------------------------------------------------------------- */

unsigned char* NS( MemPool_get_next_begin_pointer )( NS( MemPool ) *
                                                         SIXTRL_RESTRICT pool,
                                                     size_t block_alignment )
{
    return (unsigned char*)NS( MemPool_get_next_begin_const_pointer )(
        pool, block_alignment );
}

unsigned char const* NS( MemPool_get_next_begin_const_pointer )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool, size_t block_alignment )
{
    unsigned char const* ptr_begin = NS( MemPool_get_const_buffer )( pool );
    if( ptr_begin != 0 )
    {
        ptr_begin = ptr_begin + NS( MemPool_get_next_begin_offset )(
                                    pool, block_alignment );
    }

    return ptr_begin;
}

/* ------------------------------------------------------------------------- */

void NS( MemPool_set_capacity )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                 size_t new_capacity )
{
    assert( pool != 0 );
    pool->capacity = new_capacity;
    return;
}

/* -------------------------------------------------------------------------- */

void NS( MemPool_set_size )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                             size_t new_size )
{
    assert( pool != 0 );
    pool->size = new_size;
    return;
}

/* -------------------------------------------------------------------------- */

void NS( MemPool_set_chunk_size )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                   size_t new_chunk_size )
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
