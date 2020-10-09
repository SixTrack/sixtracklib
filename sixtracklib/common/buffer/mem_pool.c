#include "sixtracklib/common/buffer/mem_pool.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer/alignment.h"

/* ========================================================================= */

extern void NS( MemPool_init_aligned)(
    NS( MemPool ) * SIXTRL_RESTRICT pool, SIXTRL_UINT64_T capacity,
    SIXTRL_UINT64_T const chunk_size, SIXTRL_UINT64_T const begin_alignment );

extern bool NS(MemPool_clear_to_aligned_position)(
    NS(MemPool)* SIXTRL_RESTRICT pool, SIXTRL_UINT64_T const alignment  );

extern bool NS( MemPool_reserve_aligned )(
    NS( MemPool ) * SIXTRL_RESTRICT pool,
    SIXTRL_UINT64_T const new_capacity, SIXTRL_UINT64_T const alignment );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char*
NS( MemPool_get_buffer )( NS( MemPool ) * SIXTRL_RESTRICT pool );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char const*
NS( MemPool_get_const_buffer )( const NS( MemPool ) *const SIXTRL_RESTRICT pool );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC void NS( MemPool_set_capacity )(
    NS( MemPool ) * SIXTRL_RESTRICT pool, SIXTRL_UINT64_T const new_capacity );

SIXTRL_STATIC void NS( MemPool_set_size )(
    NS( MemPool ) * SIXTRL_RESTRICT pool, SIXTRL_UINT64_T const new_size );

SIXTRL_STATIC void NS( MemPool_set_buffer )(
    NS( MemPool ) * SIXTRL_RESTRICT pool,
    SIXTRL_GLOBAL_DEC unsigned char* new_buffer );

SIXTRL_STATIC void NS( MemPool_set_begin_pos )(
    NS( MemPool )* SIXTRL_RESTRICT pool,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT begin_pos );

SIXTRL_STATIC void NS( MemPool_set_begin_offset )(
    NS( MemPool )* SIXTRL_RESTRICT pool,
    SIXTRL_UINT64_T const begin_offset );

/* ========================================================================= */

void NS( MemPool_init_aligned)(
    NS( MemPool ) * SIXTRL_RESTRICT pool, SIXTRL_UINT64_T capacity,
    SIXTRL_UINT64_T const chunk_size, SIXTRL_UINT64_T const begin_alignment )
{
    static size_t const ZERO = (SIXTRL_UINT64_T)0u;

    NS( MemPool_preset( pool ) );

    if( ( pool != 0 ) && ( chunk_size > ZERO ) && ( begin_alignment > ZERO ) )
    {
        typedef unsigned char uchar_t;
        typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;

        g_ptr_uchar_t new_buffer = 0;

        SIXTRL_UINT64_T requested_capacity = begin_alignment +
            ( ( capacity > chunk_size ) ? capacity : chunk_size );

        SIXTRL_UINT64_T const num_chunks = requested_capacity / chunk_size;
        SIXTRL_UINT64_T const calc_capacity = num_chunks * chunk_size;

        capacity = ( calc_capacity < requested_capacity )
                 ? ( calc_capacity + chunk_size ) : ( calc_capacity );

        /* use sizeof( unsigned char ) because of some theoretical ambiguity
         * in C++ which allows for sizeof( unsigned char ) > sizeof( char ) */
        new_buffer = ( g_ptr_uchar_t )malloc( capacity * sizeof( uchar_t ) );

        if( new_buffer != 0 )
        {
            uintptr_t const buffer_addr = ( uintptr_t )new_buffer;
            uintptr_t const addr_mod    = buffer_addr % begin_alignment;

            SIXTRL_UINT64_T const begin_offset = ( addr_mod != ( uintptr_t )0 )
                ? ( SIXTRL_UINT64_T )( begin_alignment - addr_mod ) : ZERO;

            g_ptr_uchar_t begin_pos = new_buffer + begin_offset;

            if( begin_offset > ZERO )
            {
                SIXTRL_ASSERT( capacity > begin_offset );
                capacity -= begin_offset;
            }

            NS( MemPool_set_buffer )( pool, new_buffer );
            NS( MemPool_set_begin_pos )( pool, begin_pos );
            NS( MemPool_set_capacity )( pool, capacity );
            NS( MemPool_set_begin_offset )( pool, begin_offset );
            NS( MemPool_set_size )( pool, ZERO );
            NS( MemPool_set_chunk_size )( pool, chunk_size );
        }
    }

    return;
}

/* -------------------------------------------------------------------------- */

bool NS(MemPool_clear_to_aligned_position)(
    NS(MemPool)* SIXTRL_RESTRICT pool, SIXTRL_UINT64_T const alignment )
{
    bool success = false;

    typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;

    SIXTRL_STATIC SIXTRL_UINT64_T const ZERO = ( SIXTRL_UINT64_T )0u;

    SIXTRL_UINT64_T const chunk_size =
        NS(MemPool_get_chunk_size)( pool );

    SIXTRL_UINT64_T const buffer_capacity =
        NS(MemPool_get_buffer_capacity)( pool );

    g_ptr_uchar_t buffer = NS(MemPool_get_buffer)( pool );

    NS(MemPool_clear)( pool );

    if( ( alignment > ZERO ) && ( chunk_size > ZERO ) && ( buffer != 0 ) )
    {
        SIXTRL_UINT64_T const use_align =
            NS(least_common_multiple)( alignment, chunk_size );

        if( use_align == 0u ) return false;

        SIXTRL_ASSERT( NS(MemPool_get_buffer_capacity)( pool ) >=
                       NS(MemPool_get_capacity)( pool ) );

        if( use_align > buffer_capacity )
        {
            uintptr_t const addr_mod = ( ( uintptr_t )buffer ) % use_align;

            if( addr_mod != ( uintptr_t )0u )
            {
                SIXTRL_UINT64_T const offset = use_align - addr_mod;

                if( offset == ZERO )
                {
                    NS(MemPool_set_begin_pos)( pool, buffer );
                    NS(MemPool_set_capacity)( pool, buffer_capacity );
                }
                else if( ( offset != ZERO ) && ( buffer_capacity > offset ) )
                {
                    NS(MemPool_set_begin_pos)( pool, buffer + offset );
                    NS(MemPool_set_capacity)( pool, buffer_capacity - offset );
                }
                else
                {
                    NS(MemPool_set_begin_pos)( pool, 0 );
                    NS(MemPool_set_capacity)( pool, ZERO );
                }

                success = true;
            }
        }
    }

    return success;
}

/* -------------------------------------------------------------------------- */

bool NS( MemPool_reserve_aligned )(
    NS( MemPool ) * SIXTRL_RESTRICT pool,
    SIXTRL_UINT64_T const new_capacity, SIXTRL_UINT64_T const align )
{
    bool has_been_changed = false;

    static SIXTRL_UINT64_T ZERO = (SIXTRL_UINT64_T)0u;

    SIXTRL_UINT64_T const current_capacity = NS( MemPool_get_capacity( pool ) );
    SIXTRL_UINT64_T const chunk_size = NS( MemPool_get_chunk_size( pool ) );

    if( ( pool != 0 ) && ( new_capacity > current_capacity ) &&
        ( chunk_size > ZERO ) )
    {
        typedef SIXTRL_GLOBAL_DEC unsigned char const* g_const_ptr_uchar_t;

        NS(MemPool) rollback_copy = *pool;

        NS( MemPool_preset( pool ) );
        SIXTRL_ASSERT( NS( MemPool_get_const_buffer( pool ) ) == 0 );

        NS( MemPool_init_aligned )( pool, new_capacity, chunk_size, align );

        if( NS( MemPool_get_const_buffer( pool ) != 0 ) )
        {
            SIXTRL_ASSERT( NS( MemPool_is_begin_aligned_with)( pool, align ) );
            SIXTRL_ASSERT( NS( MemPool_get_capacity )( pool ) >= new_capacity );

            g_const_ptr_uchar_t prev_begin_pos =
                NS(MemPool_get_const_begin_pos)( &rollback_copy );

            if( prev_begin_pos != 0 )
            {
                SIXTRL_UINT64_T const prev_size =
                    NS(MemPool_get_size)( &rollback_copy );

                if( prev_size > ZERO )
                {
                    SIXTRL_ASSERT( prev_size < new_capacity );

                    memcpy( NS(MemPool_get_begin_pos)( pool ),
                            prev_begin_pos, prev_size );

                    NS( MemPool_set_size )( pool, prev_size );
                }

                NS(MemPool_free)( &rollback_copy );
            }

            has_been_changed = true;
        }
        else
        {
            *pool = rollback_copy;
        }
    }

    return has_been_changed;
}

/* -------------------------------------------------------------------------- */

SIXTRL_GLOBAL_DEC unsigned char*
NS( MemPool_get_buffer )( NS( MemPool ) * SIXTRL_RESTRICT pool )
{
    return ( pool != 0 ) ? ( pool->buffer ) : 0;
}

SIXTRL_GLOBAL_DEC unsigned char const*
NS( MemPool_get_const_buffer )( const NS( MemPool ) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? ( p->buffer ) : 0;
}

/* -------------------------------------------------------------------------- */

void NS( MemPool_set_capacity )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                                 SIXTRL_UINT64_T const new_capacity )
{
    assert( pool != 0 );
    pool->capacity = new_capacity;
    return;
}

/* -------------------------------------------------------------------------- */

void NS( MemPool_set_size )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                             SIXTRL_UINT64_T const new_size )
{
    assert( pool != 0 );
    pool->size = new_size;
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

/* -------------------------------------------------------------------------- */

void NS( MemPool_set_begin_pos )(
    NS( MemPool )* SIXTRL_RESTRICT pool,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT begin_pos )
{
    if( pool != 0 ) pool->begin_pos = begin_pos;
    return;
}

/* -------------------------------------------------------------------------- */

void NS( MemPool_set_begin_offset )( NS( MemPool )* SIXTRL_RESTRICT pool,
    SIXTRL_UINT64_T const begin_offset )
{
    if( pool != 0 ) pool->begin_offset = begin_offset;
    return;
}

/* ========================================================================= */

/* end: sixtracklib/common/buffer/mem_pool.c */
