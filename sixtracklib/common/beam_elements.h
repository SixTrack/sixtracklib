#ifndef SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__
#define SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__

#if !defined( _GPUCODE )

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/blocks.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

struct NS(Blocks);

/* ************************************************************************* */

typedef struct NS(Drift)
{
    SIXTRL_REAL_T length __attribute__(( aligned( 8 ) ));
}
NS(Drift);

SIXTRL_STATIC NS(block_size_t) NS(Drift_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const num_of_blocks );

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(Drift)* NS(Blocks_add_drift)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks, SIXTRL_REAL_T const length );

SIXTRL_GLOBAL_DEC NS(Drift)* NS(Blocks_reserve_drift)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks );

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

typedef struct NS(DriftExact)
{
    SIXTRL_REAL_T length __attribute__(( aligned( 8 ) ));
}
NS(DriftExact);

SIXTRL_STATIC NS(block_size_t) NS(DriftExact_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const num_of_blocks );

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(DriftExact)* NS(Blocks_add_drift_exact)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks, SIXTRL_REAL_T const length );

SIXTRL_GLOBAL_DEC NS(DriftExact)* NS(Blocks_reserve_drift_exact)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks );

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

typedef struct NS(MultiPole)
{
    SIXTRL_REAL_T   length  __attribute__(( aligned( 8 ) ));
    SIXTRL_REAL_T   hxl     __attribute__(( aligned( 8 ) ));
    SIXTRL_REAL_T   hyl     __attribute__(( aligned( 8 ) ));
    SIXTRL_INT64_T  order   __attribute__(( aligned( 8 ) ));
    
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
        SIXTRL_RESTRICT bal __attribute__(( aligned( 8 ) ));
}
NS(MultiPole);

SIXTRL_STATIC NS(block_size_t) NS(MultiPole_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const num_of_blocks, NS(block_size_t) const max_order );

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(MultiPole)* NS(Blocks_add_multipole)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, SIXTRL_REAL_T const length, 
    SIXTRL_REAL_T const hxl, SIXTRL_REAL_T const hyl, 
    SIXTRL_INT64_T const order, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT bal );
    
SIXTRL_GLOBAL_DEC NS(MultiPole)* NS(Blocks_reserve_multipole)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, 
    SIXTRL_INT64_T const order );

#endif /* !defined( _GPUCODE ) */

/* ========================================================================= */
/* =====             Implementation of inline functions                ===== */
/* ========================================================================= */

#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/impl/beam_elements_api.h"

SIXTRL_INLINE NS(block_size_t) NS(Drift_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const num_of_blocks )
{
    NS(block_size_t) attr_data_capacity = ( NS(block_size_t) )0u;
    
    if( ( blocks != 0 ) && ( num_of_blocks > 0u ) )
    {
        SIXTRL_STATIC NS(block_size_t) const NUM_REAL_ATTRIBUTES = 1u;        
        
        NS(block_size_t) const REAL_ATTRIBUTE_SIZE = sizeof( SIXTRL_REAL_T );
        
        NS(block_size_t) const alignment = 
            NS(Blocks_get_data_alignment)( blocks );
            
        attr_data_capacity = alignment + num_of_blocks * (
            ( NUM_REAL_ATTRIBUTES * REAL_ATTRIBUTE_SIZE ) +
            alignment + sizeof( NS(Drift) ) + alignment   + 
            NUM_REAL_ATTRIBUTES * sizeof( SIXTRL_GLOBAL_DEC void** ) );
    }
        
    return attr_data_capacity;
}

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(Drift)* NS(Blocks_add_drift)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks, SIXTRL_REAL_T const length )
{
    SIXTRL_GLOBAL_DEC NS(Drift)* drift = NS(Blocks_reserve_drift)( blocks );
    if( drift != 0 ) NS(Drift_set_length)( drift, length );
    
    return drift;
}
#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(block_size_t) NS(DriftExact_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const num_of_blocks )
{
    NS(block_size_t) attr_data_capacity = ( NS(block_size_t) )0u;
    
    if( ( blocks != 0 ) && ( num_of_blocks > 0u ) )
    {
        SIXTRL_STATIC NS(block_size_t) const NUM_REAL_ATTRIBUTES = 1u;        
        
        NS(block_size_t) const REAL_ATTRIBUTE_SIZE = sizeof( SIXTRL_REAL_T );
        
        NS(block_size_t) const alignment = 
            NS(Blocks_get_data_alignment)( blocks );
            
        attr_data_capacity = alignment + num_of_blocks * (
            ( NUM_REAL_ATTRIBUTES * REAL_ATTRIBUTE_SIZE ) +
            alignment + sizeof( NS(DriftExact) ) + alignment   + 
            NUM_REAL_ATTRIBUTES * sizeof( SIXTRL_GLOBAL_DEC void** ) );
    }
        
    return attr_data_capacity;
}

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(DriftExact)* NS(Blocks_add_drift_exact)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks, SIXTRL_REAL_T const length )
{
    SIXTRL_GLOBAL_DEC NS(DriftExact)* drift = 
        NS(Blocks_reserve_drift_exact)( blocks );
    
    if( drift != 0 ) NS(DriftExact_set_length)( drift, length );
    
    return drift;
}

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(block_size_t) NS(MultiPole_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const num_of_blocks, NS(block_size_t) const max_order )
{
    NS(block_size_t) attr_data_capacity   = ( NS(block_size_t) )0u;    
    
    if( ( blocks != 0 ) && ( num_of_blocks > 0u ) && ( max_order > 0u ) )
    {
        SIXTRL_STATIC NS(block_size_t) const 
            NUM_REAL_ATTRIBUTES = 1u;        
        
        NS(block_size_t) const NUM_OF_BAL_VALUES = max_order * 2u + 2u;
            
        NS(block_size_t) const REAL_ATTRIBUTE_SIZE = 
            NUM_OF_BAL_VALUES * sizeof( SIXTRL_REAL_T );
        
        NS(block_size_t) const alignment = 
            NS(Blocks_get_data_alignment)( blocks );
            
        attr_data_capacity = alignment + num_of_blocks * (
            ( NUM_REAL_ATTRIBUTES * REAL_ATTRIBUTE_SIZE ) +
            alignment + sizeof( NS(Drift) ) + alignment   + 
            NUM_REAL_ATTRIBUTES * sizeof( SIXTRL_GLOBAL_DEC void** ) );
    }
        
    return attr_data_capacity;
}

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(MultiPole)* NS(Blocks_add_multipole)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, SIXTRL_REAL_T const length, 
    SIXTRL_REAL_T const hxl, SIXTRL_REAL_T const hyl, 
    SIXTRL_INT64_T const order, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT bal )
{
    SIXTRL_GLOBAL_DEC NS(MultiPole)* multipole = 
        NS(Blocks_reserve_multipole)( blocks, order );
    
    if( multipole != 0 )
    {
        SIXTRL_ASSERT( NS(MultiPole_get_order)( multipole ) == order );
        
        NS(MultiPole_set_length)( multipole, length );
        NS(MultiPole_set_hxl)( multipole, hxl );
        NS(MultiPole_set_hyl)( multipole, hyl );
        NS(MultiPole_set_bal)( multipole, order, bal );
    }
    
    return multipole;
}

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__ */

/* end: sixtracklib/sixtracklib/common/beam_elements.h */
