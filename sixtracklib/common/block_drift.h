#ifndef SIXTRACKLIB_COMMON_BLOCK_DRIFT_H__
#define SIXTRACKLIB_COMMON_BLOCK_DRIFT_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/impl/block_type.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */
    
struct NS(Drift);
struct NS(AllocResult);
struct NS(MemPool);

struct NS(Drift)* NS(Drift_preset)( 
    struct NS(Drift)* SIXTRL_RESTRICT drift );

/* ------------------------------------------------------------------------- */

SIXTRL_SIZE_T NS(Drift_predict_required_mempool_capacity_for_packing)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    const struct NS(MemPool) *const SIXTRL_RESTRICT pool, 
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_alignment );

SIXTRL_SIZE_T NS(Drift_predict_required_capacity_for_packing)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin, 
    SIXTRL_SIZE_T const alignment );

/* ------------------------------------------------------------------------- */

SIXTRL_SIZE_T NS(Drift_map_to_pack_flat_memory_aligned)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    unsigned char* mem_begin, SIXTRL_SIZE_T const alignment );

SIXTRL_STATIC  SIXTRL_SIZE_T NS(Drift_map_to_pack_flat_memory)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    unsigned char* mem_begin );

struct NS(AllocResult) NS(Drift_map_to_pack_aligned)( 
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    struct NS(MemPool)* SIXTRL_RESTRICT pool, SIXTRL_SIZE_T alignment );

SIXTRL_STATIC struct NS(AllocResult) NS(Drift_map_to_pack)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    struct NS(MemPool)* SIXTRL_RESTRICT pool );

/* -------------------------------------------------------------------------- */
/* -----                         Inline functions                             */
/* -------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

#include "sixtracklib/common/mem_pool.h"

#endif /* !defined( _GPUCODE ) */

SIXTRL_INLINE SIXTRL_SIZE_T NS(Drift_map_to_pack_flat_memory)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    unsigned char* mem_begin )
{
    return NS(Drift_map_to_pack_flat_memory_aligned)(
        drift, mem_begin, NS(BLOCK_DEFAULT_MEMPOOL_ALIGNMENT) );
}

SIXTRL_INLINE struct NS(AllocResult) NS(Drift_map_to_pack)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    NS(MemPool)* SIXTRL_RESTRICT pool )
{
    return NS(Drift_map_to_pack_aligned)(
        drift, pool, NS(BLOCK_DEFAULT_MEMPOOL_ALIGNMENT) );
}

#if !defined( _GPUCODE ) 

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */
                                   
#endif /* !defined( _GPUCODE ) */
                                   
#endif /* SIXTRACKLIB_COMMON_BLOCK_DRIFT_H__ */

/* end: sixtracklib/common/block_drift.h */
