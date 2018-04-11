#ifndef SIXTRACKLIB_COMMON_MEMORY_POOL_H__
#define SIXTRACKLIB_COMMON_MEMORY_POOL_H__

#include "sixtracklib/_impl/namespace_begin.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/restrict.h"

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

/* -------------------------------------------------------------------------- */

typedef struct NS( AllocResult )
{
    unsigned char* p;
    uint64_t offset;
    uint64_t length;
} NS( AllocResult );

bool NS( AllocResult_valid )( 
    const NS( AllocResult ) * const SIXTRL_RESTRICT result );

bool NS(AllocResult_is_aligned)( 
    const NS(AllocResult) *const SIXTRL_RESTRICT result, 
    size_t alignment );

NS( AllocResult ) *
    NS( AllocResult_preset )( NS( AllocResult ) * SIXTRL_RESTRICT result );

unsigned char* NS( AllocResult_get_pointer )( const NS( AllocResult ) *
                                              const SIXTRL_RESTRICT result );

uint64_t NS( AllocResult_get_offset )( const NS( AllocResult ) *
                                       const SIXTRL_RESTRICT result );

uint64_t NS( AllocResult_get_length )( const NS( AllocResult ) *
                                       const SIXTRL_RESTRICT result );

/* -------------------------------------------------------------------------- */

typedef struct NS( MemPool )
{
    unsigned char* buffer;

    size_t capacity;
    size_t size;
    size_t chunk_size;
} NS(MemPool);

NS( MemPool ) * NS( MemPool_preset )( NS( MemPool ) * SIXTRL_RESTRICT pool );

void NS( MemPool_init )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                         size_t capacity,
                         size_t chunk_size );

void NS( MemPool_free )( NS( MemPool ) * SIXTRL_RESTRICT pool );

void NS( MemPool_clear )( NS( MemPool ) * SIXTRL_RESTRICT pool );

bool NS( MemPool_reserve )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                            size_t new_capacity );

void NS( MemPool_clone )( NS( MemPool ) * SIXTRL_RESTRICT dest,
                          const NS( MemPool ) * const SIXTRL_RESTRICT source );

bool NS( MemPool_is_empty )( const NS( MemPool ) * const SIXTRL_RESTRICT pool );

size_t NS( MemPool_get_capacity )( const NS( MemPool ) *
                                   const SIXTRL_RESTRICT pool );

size_t NS( MemPool_get_size )( const NS( MemPool ) *
                               const SIXTRL_RESTRICT pool );

size_t NS( MemPool_get_chunk_size )( const NS( MemPool ) *
                                     const SIXTRL_RESTRICT pool );

size_t NS( MemPool_get_remaining_bytes )( const NS( MemPool ) *
                                          const SIXTRL_RESTRICT pool );

uint64_t NS( MemPool_get_next_begin_offset )( const NS( MemPool ) * const pool,
                                              size_t block_alignment );

unsigned char* NS( MemPool_get_buffer )( NS( MemPool ) * SIXTRL_RESTRICT pool );

unsigned char* NS( MemPool_get_next_begin_pointer )( NS( MemPool ) *
                                                         SIXTRL_RESTRICT pool,
                                                     size_t block_alignment );

unsigned char* NS( MemPool_get_pointer_by_offset )( NS( MemPool ) *
                                                        SIXTRL_RESTRICT pool,
                                                    uint64_t offset );

unsigned char const* NS( MemPool_get_const_buffer )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool );

unsigned char const* NS( MemPool_get_next_begin_const_pointer )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool, size_t block_alignment );

unsigned char const* NS( MemPool_get_const_pointer_by_offset )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool, uint64_t offset );

NS( AllocResult )
NS( MemPool_append )( NS( MemPool ) * SIXTRL_RESTRICT pool, size_t num_bytes );

NS( AllocResult )
NS( MemPool_append_aligned )
( NS( MemPool ) * SIXTRL_RESTRICT pool,
  size_t num_bytes,
  size_t block_alignment );

/* -------------------------------------------------------------------------- */

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_MEMORY_POOL_H__ */

/* end: sixtracklib/common/mem_pool.h */
