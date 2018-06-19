#ifndef SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__
#define SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__

#if !defined( _GPUCODE )

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/impl/beam_elements_type.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

struct NS(Blocks);

/* ************************************************************************* */

SIXTRL_FN SIXTRL_STATIC 
NS(block_size_t) NS(Drift_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const num_of_blocks );

#if !defined( _GPUCODE )

SIXTRL_HOST_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(Drift)* NS(Blocks_add_drift)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks, SIXTRL_REAL_T const length );

SIXTRL_HOST_FN SIXTRL_GLOBAL_DEC NS(Drift)* NS(Blocks_reserve_drift)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks );

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(block_size_t) 
NS(DriftExact_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const num_of_blocks );

#if !defined( _GPUCODE )

SIXTRL_HOST_FN  SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(DriftExact)* 
NS(Blocks_add_drift_exact)( NS(Blocks)* SIXTRL_RESTRICT blocks, 
                            SIXTRL_REAL_T const length );

SIXTRL_HOST_FN  SIXTRL_GLOBAL_DEC NS(DriftExact)* 
NS(Blocks_reserve_drift_exact)( NS(Blocks)* SIXTRL_RESTRICT blocks );

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(block_size_t) NS(MultiPole_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const num_of_blocks, NS(block_size_t) const max_order );

#if !defined( _GPUCODE )

SIXTRL_HOST_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(MultiPole)* NS(Blocks_add_multipole)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, SIXTRL_REAL_T const length, 
    SIXTRL_REAL_T const hxl, SIXTRL_REAL_T const hyl, 
    SIXTRL_INT64_T const order, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT bal );
    
SIXTRL_HOST_FN SIXTRL_GLOBAL_DEC NS(MultiPole)* NS(Blocks_reserve_multipole)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, 
    SIXTRL_INT64_T const order );

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(block_size_t) 
NS(BeamBeam_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks,
    NS(block_size_t) const num_of_blocks, 
    NS(block_num_elements_t) const max_num_slices );

#if !defined( _GPUCODE )

SIXTRL_HOST_FN SIXTRL_STATIC 
SIXTRL_GLOBAL_DEC NS(BeamBeam)* NS(Blocks_add_beam_beam)(
    NS(Blocks)* SIXTRL_RESTRICT blocks,
    const NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost_data,
    const NS(BeamBeamSigmas)    *const SIXTRL_RESTRICT sigmas,
    NS(block_num_elements_t) const num_of_slices, 
    SIXTRL_REAL_T const* SIXTRL_RESTRICT n_part_per_slice_begin,
    SIXTRL_REAL_T const* SIXTRL_RESTRICT x_slices_star_begin,
    SIXTRL_REAL_T const* SIXTRL_RESTRICT y_slices_star_begin,
    SIXTRL_REAL_T const* SIXTRL_RESTRICT sigma_slices_star_begin,
    SIXTRL_REAL_T const q_part, SIXTRL_REAL_T const min_sigma_diff, 
    SIXTRL_REAL_T const treshold_singular );
    
SIXTRL_HOST_FN SIXTRL_GLOBAL_DEC NS(BeamBeam)* NS(Blocks_reserve_beam_beam)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, 
    NS(block_num_elements_t) const num_of_slices );

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(block_size_t) 
NS(Cavity_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks,
    NS(block_size_t) const num_of_blocks );

#if !defined( _GPUCODE )

SIXTRL_HOST_FN SIXTRL_STATIC 
SIXTRL_GLOBAL_DEC NS(Cavity)* NS(Blocks_add_cavity)(
    NS(Blocks)* SIXTRL_RESTRICT blocks,
    SIXTRL_REAL_T const voltage, 
    SIXTRL_REAL_T const frequency, 
    SIXTRL_REAL_T const lag );
    
SIXTRL_HOST_FN SIXTRL_GLOBAL_DEC NS(Cavity)* 
NS(Blocks_reserve_cavity)( NS(Blocks)* SIXTRL_RESTRICT blocks );

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(block_size_t) 
NS(Align_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks,
    NS(block_size_t) const num_of_blocks );

#if !defined( _GPUCODE )

SIXTRL_HOST_FN SIXTRL_STATIC 
SIXTRL_GLOBAL_DEC NS(Align)* NS(Blocks_add_align)(
    NS(Blocks)* SIXTRL_RESTRICT blocks,
    SIXTRL_REAL_T const tilt, 
    SIXTRL_REAL_T const cz,  SIXTRL_REAL_T const sz,
    SIXTRL_REAL_T const dx,  SIXTRL_REAL_T const dy 
);
    
SIXTRL_HOST_FN SIXTRL_GLOBAL_DEC NS(Align)* 
NS(Blocks_reserve_align)( NS(Blocks)* SIXTRL_RESTRICT blocks );

#endif /* !defined( _GPUCODE ) */

/* ========================================================================= */
/* =====             Implementation of inline functions                ===== */
/* ========================================================================= */

#if !defined( _GPUCODE )
#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/impl/beam_elements_api.h"
#endif /* !defined( _GPUCODE ) */

SIXTRL_INLINE NS(block_size_t) NS(Drift_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const num_of_blocks )
{
    NS(block_size_t) attr_data_capacity = ( NS(block_size_t) )0u;
    
    if( ( blocks != 0 ) && ( num_of_blocks > 0u ) )
    {
        SIXTRL_STATIC_VAR NS(block_size_t) const NUM_REAL_ATTRIBUTES = 1u;        
        
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
        SIXTRL_STATIC_VAR NS(block_size_t) const NUM_REAL_ATTRIBUTES = 1u;        
        
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
        SIXTRL_STATIC_VAR NS(block_size_t) const 
            NUM_REAL_ATTRIBUTES = 1u;        
        
        NS(block_size_t) const NUM_OF_BAL_VALUES = max_order * 2u + 2u;
            
        NS(block_size_t) const REAL_ATTRIBUTE_SIZE = 
            NUM_OF_BAL_VALUES * sizeof( SIXTRL_REAL_T );
        
        NS(block_size_t) const alignment = 
            NS(Blocks_get_data_alignment)( blocks );
            
        attr_data_capacity = alignment + num_of_blocks * (
            ( NUM_REAL_ATTRIBUTES * REAL_ATTRIBUTE_SIZE ) +
            alignment + sizeof( NS(MultiPole) ) + alignment   + 
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

SIXTRL_INLINE NS(block_size_t) NS(BeamBeam_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks,
    NS(block_size_t) const num_of_blocks, 
    NS(block_num_elements_t) const max_num_slices )
{
    NS(block_size_t) attr_data_capacity   = ( NS(block_size_t) )0u;    
    
    if( ( blocks != 0 ) && ( num_of_blocks > 0u ) && ( max_num_slices > 0u ) )
    {
        SIXTRL_STATIC_VAR NS(block_size_t) const NUM_SLICE_ATTRIBUTES = 4u;
        SIXTRL_STATIC_VAR NS(block_size_t) const NUM_DATA_POINTERS    = 6u;
            
        NS(block_size_t) const BEAM_BEAM_HANDLE_SIZE = 
            sizeof( NS(BeamBeam) );
            
        NS(block_size_t) const BEAM_BEAM_BOOST_DATA_SIZE = 
            sizeof( NS(BeamBeamBoostData) );
            
        NS(block_size_t) const BEAM_BEAM_SIGMA_MATRIX_SIZE =
            sizeof( NS(BeamBeamSigmas) );
            
        NS(block_size_t) const SLICE_ATTRIBUTE_SIZE = 
            sizeof( SIXTRL_REAL_T ) * max_num_slices;
            
        NS(block_size_t) const alignment = 
            NS(Blocks_get_data_alignment)( blocks );
            
        attr_data_capacity = alignment + num_of_blocks * (
            alignment + BEAM_BEAM_HANDLE_SIZE +
            alignment + BEAM_BEAM_BOOST_DATA_SIZE +
            alignment + BEAM_BEAM_SIGMA_MATRIX_SIZE +
            alignment + ( NUM_SLICE_ATTRIBUTES * SLICE_ATTRIBUTE_SIZE ) +
            alignment + NUM_DATA_POINTERS * sizeof( SIXTRL_GLOBAL_DEC void** ) );
    }
        
    return attr_data_capacity;
}

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(BeamBeam)* NS(Blocks_add_beam_beam)(
    NS(Blocks)* SIXTRL_RESTRICT blocks,
    const NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost_data,
    const NS(BeamBeamSigmas)    *const SIXTRL_RESTRICT sigmas,
    NS(block_num_elements_t) const num_of_slices, 
    SIXTRL_REAL_T const* SIXTRL_RESTRICT n_part_per_slice_begin,
    SIXTRL_REAL_T const* SIXTRL_RESTRICT x_slices_star_begin,
    SIXTRL_REAL_T const* SIXTRL_RESTRICT y_slices_star_begin,
    SIXTRL_REAL_T const* SIXTRL_RESTRICT sigma_slices_star_begin,
    SIXTRL_REAL_T const q_part, SIXTRL_REAL_T const min_sigma_diff, 
    SIXTRL_REAL_T const treshold_singular )
{
    SIXTRL_GLOBAL_DEC NS(BeamBeam)* beam_beam = 
        NS(Blocks_reserve_beam_beam)( blocks, num_of_slices );
    
    if( beam_beam != 0 )
    {
        SIXTRL_ASSERT( NS(BeamBeam_get_num_of_slices)( beam_beam ) == 
                       num_of_slices );
        
        NS(BeamBeam_set_boost_data_value)( beam_beam, boost_data );
        NS(BeamBeam_set_sigmas_matrix_value)( beam_beam, sigmas );
        NS(BeamBeam_set_n_part_per_slice)( beam_beam, n_part_per_slice_begin );
        NS(BeamBeam_set_x_slices_star)(    beam_beam, x_slices_star_begin );
        NS(BeamBeam_set_y_slices_star)(    beam_beam, y_slices_star_begin );
        
        NS(BeamBeam_set_sigma_slices_star)(
                beam_beam, sigma_slices_star_begin );
        
        NS(BeamBeam_set_q_part)( beam_beam, q_part );
        NS(BeamBeam_set_min_sigma_diff)( beam_beam, min_sigma_diff );
        NS(BeamBeam_set_treshold_singular)( beam_beam, treshold_singular );
    }
    
    return beam_beam;
}

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(block_size_t) NS(Cavity_predict_blocks_data_capacity)( 
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const num_of_blocks )
{
    NS(block_size_t) attr_data_capacity = ( NS(block_size_t) )0u;
    
    if( ( blocks != 0 ) && ( num_of_blocks > 0u ) )
    {
        SIXTRL_STATIC_VAR NS(block_size_t) const CAVITY_SIZE = 
            sizeof( NS(Cavity ) );
        
        NS(block_size_t) const alignment = 
            NS(Blocks_get_data_alignment)( blocks );
            
        attr_data_capacity = num_of_blocks * ( alignment + CAVITY_SIZE );
    }
        
    return attr_data_capacity;
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(Cavity)* NS(Blocks_add_cavity)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, SIXTRL_REAL_T const voltage, 
    SIXTRL_REAL_T const frequency, SIXTRL_REAL_T const lag )
{
    SIXTRL_GLOBAL_DEC NS(Cavity)* cavity = NS(Blocks_reserve_cavity)( blocks );
    
    if( cavity != 0 )
    {
        NS(Cavity_set_voltage)( cavity, voltage );
        NS(Cavity_set_frequency)( cavity, frequency );
        NS(Cavity_set_lag)( cavity, lag );
    }
    
    return cavity;
}
    
#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(block_size_t) NS(Align_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks,
    NS(block_size_t) const num_of_blocks )
{
    NS(block_size_t) attr_data_capacity = ( NS(block_size_t) )0u;
    
    if( ( blocks != 0 ) && ( num_of_blocks > 0u ) )
    {
        SIXTRL_STATIC_VAR NS(block_size_t) const 
            ALIGN_SIZE = sizeof( NS(Align ) );
        
        NS(block_size_t) const alignment = 
            NS(Blocks_get_data_alignment)( blocks );
            
        attr_data_capacity = num_of_blocks * ( alignment + ALIGN_SIZE );
    }
        
    return attr_data_capacity;
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(Align)* NS(Blocks_add_align)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, SIXTRL_REAL_T const tilt,
    SIXTRL_REAL_T const cz,  SIXTRL_REAL_T const sz,
    SIXTRL_REAL_T const dx,  SIXTRL_REAL_T const dy )
{
    SIXTRL_GLOBAL_DEC NS(Align)* align = NS(Blocks_reserve_align)( blocks );
    
    if( align != 0 )
    {
        NS(Align_set_tilt)( align, tilt );
        NS(Align_set_cz)( align, cz );
        NS(Align_set_sz)( align, sz );
        NS(Align_set_dx)( align, dx );
        NS(Align_set_dy)( align, dy );
    }
    
    return align;
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
