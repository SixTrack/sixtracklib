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

SIXTRL_SIZE_T NS(Drift_predict_required_num_bytes_on_mempool_for_packing)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin, 
    SIXTRL_SIZE_T const chunk_size, 
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_alignment );

SIXTRL_STATIC SIXTRL_SIZE_T 
NS(Drift_predict_required_size_on_mempool_for_packing)(
    SIXTRL_SIZE_T const chunk_size );

SIXTRL_SIZE_T NS(Drift_predict_required_num_bytes_for_packing)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin,     
    SIXTRL_SIZE_T const alignment );

SIXTRL_STATIC SIXTRL_SIZE_T NS(Drift_predict_required_size_for_packing)();
    
/* ------------------------------------------------------------------------- */

SIXTRL_SIZE_T NS(Drift_pack_to_flat_memory_aligned)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    unsigned char* mem_begin, SIXTRL_SIZE_T const alignment );

SIXTRL_STATIC  SIXTRL_SIZE_T NS(Drift_pack_to_flat_memory)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    unsigned char* mem_begin );

struct NS(AllocResult) NS(Drift_pack_aligned)( 
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    struct NS(MemPool)* SIXTRL_RESTRICT pool, SIXTRL_SIZE_T alignment );

SIXTRL_STATIC struct NS(AllocResult) NS(Drift_pack)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    struct NS(MemPool)* SIXTRL_RESTRICT pool );

/* -------------------------------------------------------------------------- */
/* -----                         Inline functions                             */
/* -------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

#include "sixtracklib/common/mem_pool.h"

#endif /* !defined( _GPUCODE ) */

SIXTRL_INLINE SIXTRL_SIZE_T NS(Drift_predict_required_size_for_packing)()
{
    SIXTRL_STATIC SIXTRL_SIZE_T const U64_SIZE = sizeof( SIXTRL_UINT64_T );    
    return U64_SIZE * 6 + sizeof( SIXTRL_REAL_T ) + sizeof( SIXTRL_INT64_T );    
}

SIXTRL_INLINE SIXTRL_SIZE_T 
NS(Drift_predict_required_size_on_mempool_for_packing)(
    SIXTRL_SIZE_T const chunk_size )
{
    SIXTRL_SIZE_T required_size = ( SIXTRL_SIZE_T )0u;
    
    SIXTRL_ASSERT( chunk_size > ( SIXTRL_SIZE_T )0u );
    SIXTRL_STATIC const SIXTRL_SIZE_T U64_SIZE = sizeof( SIXTRL_UINT64_T );
    SIXTRL_SIZE_T const header_size = U64_SIZE * 6u;
    SIXTRL_SIZE_T const length_block_size     = sizeof( SIXTRL_REAL_T );
    SIXTRL_SIZE_T const element_id_block_size = sizeof( SIXTRL_INT64_T );
    
    SIXTRL_SIZE_T temp = ( header_size / chunk_size ) * chunk_size;
    
    required_size  = ( temp < header_size ) ? ( temp + chunk_size ) : header_size;
    
    temp = ( length_block_size / chunk_size ) * chunk_size;
    
    required_size += ( temp < length_block_size ) ? ( temp + chunk_size ) : length_block_size;
    
    temp = ( element_id_block_size / chunk_size ) * chunk_size;
    
    required_size += ( temp < element_id_block_size ) ? ( temp + chunk_size ) : element_id_block_size;
    
    return required_size;
}

SIXTRL_INLINE SIXTRL_SIZE_T NS(Drift_pack_to_flat_memory)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    unsigned char* mem_begin )
{
    SIXTRL_SIZE_T const alignment = NS(BLOCK_DEFAULT_ALIGNMENT);
    return NS(Drift_pack_to_flat_memory_aligned)( drift, mem_begin, alignment );
}

SIXTRL_INLINE struct NS(AllocResult) NS(Drift_pack)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    NS(MemPool)* SIXTRL_RESTRICT pool )
{
    SIXTRL_SIZE_T const alignment = NS(BLOCK_DEFAULT_MEMPOOL_ALIGNMENT);
    return NS(Drift_pack_aligned)( drift, pool, alignment );
}

#if !defined( _GPUCODE ) 

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */
                                   
#endif /* !defined( _GPUCODE ) */
                                   
#endif /* SIXTRACKLIB_COMMON_BLOCK_DRIFT_H__ */

/* end: sixtracklib/common/block_drift.h */
