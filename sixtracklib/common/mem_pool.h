#ifndef SIXTRACKLIB_COMMON_MEMORY_POOL_H__
#define SIXTRACKLIB_COMMON_MEMORY_POOL_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

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

bool NS( AllocResult_valid )( 
    const NS( AllocResult ) * const SIXTRL_RESTRICT result );

bool NS(AllocResult_is_aligned)( 
    const NS(AllocResult) *const SIXTRL_RESTRICT result, 
    size_t alignment );

NS( AllocResult ) *
    NS( AllocResult_preset )( NS( AllocResult ) * SIXTRL_RESTRICT result );

unsigned char* NS( AllocResult_get_pointer )( const NS( AllocResult ) *
                                              const SIXTRL_RESTRICT result );

SIXTRL_UINT64_T NS( AllocResult_get_offset )( const NS( AllocResult ) *
                                       const SIXTRL_RESTRICT result );

SIXTRL_UINT64_T NS( AllocResult_get_length )( const NS( AllocResult ) *
                                       const SIXTRL_RESTRICT result );

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
    unsigned char* buffer;

    SIXTRL_SIZE_T capacity;
    SIXTRL_SIZE_T size;
    SIXTRL_SIZE_T chunk_size;
} NS(MemPool);

NS( MemPool ) * NS( MemPool_preset )( NS( MemPool ) * SIXTRL_RESTRICT pool );

void NS( MemPool_init )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                         SIXTRL_SIZE_T capacity,
                         SIXTRL_SIZE_T const chunk_size );

void NS( MemPool_free )( NS( MemPool ) * SIXTRL_RESTRICT pool );

void NS( MemPool_clear )( NS( MemPool ) * SIXTRL_RESTRICT pool );

bool NS( MemPool_reserve )( NS( MemPool ) * SIXTRL_RESTRICT pool,
                            SIXTRL_SIZE_T new_capacity );

void NS( MemPool_clone )( NS( MemPool ) * SIXTRL_RESTRICT dest,
                          const NS( MemPool ) * const SIXTRL_RESTRICT source );

bool NS( MemPool_is_empty )( const NS( MemPool ) * const SIXTRL_RESTRICT pool );

bool NS(MemPool_clear_to_aligned_position)( 
    NS(MemPool)* SIXTRL_RESTRICT pool, SIXTRL_SIZE_T const alignment  );

SIXTRL_SIZE_T NS( MemPool_get_capacity )( const NS( MemPool ) *
                                   const SIXTRL_RESTRICT pool );

SIXTRL_SIZE_T NS( MemPool_get_size )( const NS( MemPool ) *
                               const SIXTRL_RESTRICT pool );

SIXTRL_SIZE_T NS( MemPool_get_chunk_size )( const NS( MemPool ) *
                                     const SIXTRL_RESTRICT pool );

SIXTRL_SIZE_T NS( MemPool_get_remaining_bytes )( const NS( MemPool ) *
                                          const SIXTRL_RESTRICT pool );

SIXTRL_UINT64_T NS( MemPool_get_next_begin_offset )( const NS( MemPool ) * const pool,
                                              SIXTRL_SIZE_T block_alignment );

unsigned char* NS( MemPool_get_buffer )( NS( MemPool ) * SIXTRL_RESTRICT pool );

unsigned char* NS( MemPool_get_next_begin_pointer )( NS( MemPool ) *
                                                         SIXTRL_RESTRICT pool,
                                                     SIXTRL_SIZE_T block_alignment );

unsigned char* NS( MemPool_get_pointer_by_offset )( NS( MemPool ) *
                                                        SIXTRL_RESTRICT pool,
                                                    SIXTRL_UINT64_T const offset );

unsigned char const* NS( MemPool_get_const_buffer )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool );

unsigned char const* NS( MemPool_get_next_begin_const_pointer )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool, 
    SIXTRL_SIZE_T const alignment );

unsigned char const* NS( MemPool_get_const_pointer_by_offset )(
    const NS( MemPool ) * const SIXTRL_RESTRICT pool, 
    SIXTRL_UINT64_T const offset );

NS( AllocResult )
NS( MemPool_append )( NS( MemPool ) * SIXTRL_RESTRICT pool, 
                      SIXTRL_SIZE_T const num_bytes );

NS( AllocResult )
NS( MemPool_append_aligned )
( NS( MemPool ) * SIXTRL_RESTRICT pool,
  SIXTRL_SIZE_T const num_bytes,
  SIXTRL_SIZE_T const alignment );

/* -------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(AllocResult_assign_ptr)( 
    NS(AllocResult)* SIXTRL_RESTRICT result, 
    unsigned char* SIXTRL_RESTRICT ptr )
{
    result->p = ptr;
    return;
}

SIXTRL_INLINE void NS(AllocResult_set_length)( 
    NS(AllocResult)* SIXTRL_RESTRICT result, SIXTRL_UINT64_T const length )
{
    result->length = length;
    return;
}

SIXTRL_INLINE void NS(AllocResult_set_offset)(
    NS(AllocResult)* SIXTRL_RESTRICT result, SIXTRL_UINT64_T const offset )
{
    result->offset = offset;
    return;
}

/* -------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_MEMORY_POOL_H__ */

/* end: sixtracklib/common/mem_pool.h */
