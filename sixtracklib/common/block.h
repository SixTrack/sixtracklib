#ifndef SIXTRACKLIB_COMMON_BLOCK_BASE_H__
#define SIXTRACKLIB_COMMON_BLOCK_BASE_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

struct NS(Block);
struct NS(MemPool);

/* ------------------------------------------------------------------------- */

SIXTRL_SIZE_T NS(Block_predict_required_mempool_capacity_for_packing)(
    const struct NS(MemPool) *const SIXTRL_RESTRICT pool,
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_alignment, 
    SIXTRL_SIZE_T const num_of_elements, 
    SIXTRL_SIZE_T const* ptr_attributes_sizes,
    SIXTRL_SIZE_T const num_of_attributes );

SIXTRL_SIZE_T NS(Block_predict_required_capacity_for_packing)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin,
    SIXTRL_SIZE_T const alignment, 
    SIXTRL_SIZE_T const num_of_elements,
    SIXTRL_SIZE_T const* ptr_attributes_sizes,
    SIXTRL_SIZE_T const num_of_attributes );
    
/* ------------------------------------------------------------------------- */
/*
struct NS(Block)* NS(Block_new)();

struct NS(Block)* NS(Block_new_on_mempool)( 
    struct NS(MemPool)* SIXTRL_RESTRICT pool );

void NS(Block_clear)( struct NS(Block)* SIXTRL_RESTRICT pool );

void NS(Block_free)( struct NS(Block)* SIXTRL_RESTRICT pool );
*/

/* ------------------------------------------------------------------------- */

/*
bool NS( Block_manages_own_memory )( const struct NS( Block )) *
                                         const SIXTRL_RESTRICT p );

bool NS( Block_uses_mempool )( const struct NS( Block )) *
                                   const SIXTRL_RESTRICT p );

bool NS( Block_uses_flat_memory )( 
    const struct NS(Particles )* const SIXTRL_RESTRICT p );

struct NS( MemPool ) const* NS( Block_get_const_mem_pool )(
    const struct NS( Block )) * const SIXTRL_RESTRICT p );

unsigned char const* NS( Block_get_const_flat_memory )(
    const struct NS( Block )) * const SIXTRL_RESTRICT p );
*/

/* ------------------------------------------------------------------------- */

/*
bool NS( Block_has_defined_alignment )( const struct NS( Block ) *
                                            const SIXTRL_RESTRICT block );

bool NS( Block_is_aligned )( const struct NS( Block ) *
                                     const SIXTRL_RESTRICT block,
                                 size_t alignment );

bool NS( Block_check_alignment )( const struct NS( Block ) *
                                          const SIXTRL_RESTRICT block,
                                      size_t alignment );

size_t NS( Block_get_alignment )( const struct NS( Block ) *
                                    const SIXTRL_RESTRICT block );

void NS(Block_set_alignment)( struct NS(Block)* SIXTRL_RESTRICT block, 
                              size_t new_alignment );
*/

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_BLOCK_BASE_H__ */

/* end: sixtracklib/sixtracklib/common/block.h */
