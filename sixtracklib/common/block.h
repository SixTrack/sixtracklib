#ifndef SIXTRACKLIB_COMMON_BLOCK_BASE_H__
#define SIXTRACKLIB_COMMON_BLOCK_BASE_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/impl/block_type.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

struct NS(Drift);
struct NS(MemPool);

/* ------------------------------------------------------------------------- */

SIXTRL_SIZE_T NS(Block_predict_required_num_bytes_on_mempool_for_packing)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin,
    SIXTRL_SIZE_T const chunk_size, 
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_alignment, 
    SIXTRL_SIZE_T const num_of_elements, 
    SIXTRL_SIZE_T const num_of_attributes,
    SIXTRL_SIZE_T const* ptr_attributes_sizes
                                                                         );

SIXTRL_SIZE_T NS(Block_predict_required_num_bytes_for_packing)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin,
    SIXTRL_SIZE_T const alignment, 
    SIXTRL_SIZE_T const num_of_elements,
    SIXTRL_SIZE_T const num_of_attributes,
    SIXTRL_SIZE_T const* ptr_attributes_sizes );
    
/* ------------------------------------------------------------------------- */

NS(Block)* NS(Block_new)( SIXTRL_SIZE_T const capacity );

NS(Block)* NS(Block_new_on_mempool)( SIXTRL_SIZE_T const capacity, 
    struct NS(MemPool)* SIXTRL_RESTRICT pool );

void NS(Block_clear)( NS(Block)* SIXTRL_RESTRICT pool );

void NS(Block_free)( NS(Block)* SIXTRL_RESTRICT pool );

/* ------------------------------------------------------------------------- */

bool NS( Block_manages_own_memory )( const NS(Block) *
                                         const SIXTRL_RESTRICT block );

bool NS( Block_uses_mempool )( const NS(Block) *
                                   const SIXTRL_RESTRICT block );

bool NS( Block_uses_flat_memory )( 
    const NS(Block)* const SIXTRL_RESTRICT block );

struct NS( MemPool ) const* NS( Block_get_const_mem_pool )(
    const NS(Block) * const SIXTRL_RESTRICT block );

unsigned char const* NS( Block_get_const_flat_memory )(
    const NS(Block) * const SIXTRL_RESTRICT block );

/* ------------------------------------------------------------------------- */

bool NS( Block_has_defined_alignment )( const NS(Block) *
                                            const SIXTRL_RESTRICT block );

/*
bool NS( Block_is_aligned )( const NS(Block) *
                                     const SIXTRL_RESTRICT block,
                                 size_t alignment );

bool NS( Block_check_alignment )( const NS(Block) *
                                          const SIXTRL_RESTRICT block,
                                      size_t alignment );
*/
size_t NS( Block_get_alignment )( const NS(Block) *
                                    const SIXTRL_RESTRICT block );

void NS(Block_set_alignment)( NS(Block)* SIXTRL_RESTRICT block, 
                              size_t new_alignment );


/* ========================================================================== */

SIXTRL_INT64_T NS(Block_append_drift_aligned)( NS(Block)* SIXTRL_RESTRICT block,
    NS(BeamElementType) const type_id, SIXTRL_REAL_T const length, 
    SIXTRL_SIZE_T alignment );

SIXTRL_STATIC SIXTRL_INT64_T NS(Block_append_drift)( 
    NS(Block)* SIXTRL_RESTRICT block, 
    NS(BeamElementType) const type_id, SIXTRL_REAL_T const length );

SIXTRL_STATIC SIXTRL_INT64_T NS(Block_append_drift_copy)( 
    NS(Block)* SIXTRL_RESTRICT block,
    const struct NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_INT64_T NS(Block_append_drift_copy_aligned)( 
    NS(Block)* SIXTRL_RESTRICT block,
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    SIXTRL_SIZE_T alignment );

/* ========================================================================== */
                                       
/* ************************************************************************** */
/* *****             Implementation of inline functions                  **** */
/* ************************************************************************** */

#if !defined( _GPUCODE )
#include "sixtracklib/common/impl/block_drift_type.h"
#endif /* !defined( _GPUCODE ) */

SIXTRL_INLINE SIXTRL_INT64_T NS(Block_append_drift)( 
    NS(Block)* SIXTRL_RESTRICT block,
    NS(BeamElementType) const type_id, SIXTRL_REAL_T const length )
{
    SIXTRL_SIZE_T alignment = NS(BLOCK_DEFAULT_MEMPOOL_ALIGNMENT);
    
    if( NS(Block_has_defined_alignment)( block ) )
    {
        alignment = NS(Block_get_alignment)( block );        
    }
    
    return NS(Block_append_drift_aligned)( block, type_id, length, alignment );
}

SIXTRL_INLINE SIXTRL_STATIC SIXTRL_INT64_T NS(Block_append_drift_copy)( 
    NS(Block)* SIXTRL_RESTRICT block,
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_SIZE_T alignment = NS(BLOCK_DEFAULT_MEMPOOL_ALIGNMENT);
    
    if( NS(Block_has_defined_alignment)( block ) )
    {
        alignment = NS(Block_get_alignment)( block );        
    }
    
    return NS(Block_append_drift_copy_aligned)( block, drift, alignment );
}

SIXTRL_INLINE SIXTRL_STATIC SIXTRL_INT64_T NS(Block_append_drift_copy_aligned)( 
    NS(Block)* SIXTRL_RESTRICT block,
    const NS(Drift) *const SIXTRL_RESTRICT drift, SIXTRL_SIZE_T alignment )
{
    return NS(Block_append_drift_aligned)(
        block, ( NS(BeamElementType) )NS(Drift_get_type_id)( drift ), 
        NS(Drift_get_length)( drift ), alignment );
}

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_BLOCK_BASE_H__ */

/* end: sixtracklib/sixtracklib/common/block.h */
