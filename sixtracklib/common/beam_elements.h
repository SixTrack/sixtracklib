#ifndef SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__
#define SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/impl/block_info_impl.h"
#include "sixtracklib/common/blocks_container.h"
#include "sixtracklib/common/be_drift.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* ------------------------------------------------------------------------- */

typedef struct NS(BlocksContainer) NS(BeamElements);

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC NS(BeamElements)* NS(BeamElements_preset)( 
    NS(BeamElements)* beam_elements );

SIXTRL_STATIC void NS(BeamElements_clear)( 
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements );

SIXTRL_STATIC void NS(BeamElements_free)( 
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements );

SIXTRL_STATIC int NS(BeamElements_init)( 
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_size_t) const blocks_capacity, 
    NS(block_size_t) const data_capacity );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC int NS(BeamElements_set_info_begin_alignment)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_alignment_t) const begin_alignment );

SIXTRL_STATIC int NS(BeamElements_set_data_begin_alignment)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_alignment_t) const begin_alignment );

SIXTRL_STATIC int NS(BeamElements_set_data_alignment)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_alignment_t) const alignment );

SIXTRL_STATIC int NS(BeamElements_set_info_alignment )(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_alignment_t) const alignment );

SIXTRL_STATIC void NS(BeamElements_reserve_num_blocks)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_size_t) const new_block_capacity );

SIXTRL_STATIC void NS(BeamElements_reserve_for_data)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_size_t) const new_data_capacity );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC NS(block_alignment_t) NS(BeamElements_get_info_alignment)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements );

SIXTRL_STATIC NS(block_alignment_t) NS(BeamElements_get_data_alignment)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements );

SIXTRL_STATIC NS(block_alignment_t) NS(BeamElements_get_info_begin_alignment)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements );

SIXTRL_STATIC NS(block_alignment_t) NS(BeamElements_get_data_begin_alignment)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements );

SIXTRL_STATIC NS(block_size_t) NS(BeamElements_get_data_capacity)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements );

SIXTRL_STATIC NS(block_size_t) NS(BeamElements_get_data_size)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements );

SIXTRL_STATIC NS(block_size_t) NS(BeamElements_get_block_capacity)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements );

SIXTRL_STATIC NS(block_size_t) NS(BeamElements_get_num_of_blocks)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC unsigned char const* NS(BeamElements_get_const_ptr_data_begin)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements );

SIXTRL_STATIC unsigned char* NS(BeamElements_get_ptr_data_begin)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements );

SIXTRL_STATIC NS(BlockInfo) const* NS(BeamElements_get_const_block_infos_begin)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements );

SIXTRL_STATIC NS(BlockInfo) const* NS(BeamElements_get_const_block_infos_end)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements );

SIXTRL_STATIC NS(BlockInfo)* NS(BeamElements_get_block_infos_begin)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements );

SIXTRL_STATIC NS(BlockInfo)* NS(BeamElements_get_infos_end)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements );

SIXTRL_STATIC NS(BlockInfo) NS(BeamElements_get_block_info_by_index)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements, 
    NS(block_size_t) const block_index );

SIXTRL_STATIC NS(BlockInfo) const* NS(BeamElements_get_const_ptr_to_block_info_by_index)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements, 
    NS(block_size_t) const block_index );

SIXTRL_STATIC NS(BlockInfo)* NS(BeamElements_get_ptr_to_block_info_by_index)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_size_t) const block_index );

/* ------------------------------------------------------------------------- */

NS(Drift) NS(BeamElements_add_drift)( 
    NS(BeamElements)* beam_elements, SIXTRL_REAL_T  const length, 
    SIXTRL_INT64_T const element_id );

NS(Drift) NS(BeamElements_add_drift_exact)( 
    NS(BeamElements)* beam_elements, SIXTRL_REAL_T const length, 
    SIXTRL_INT64_T const element_id );

NS(Drift) NS(BeamElements_add_drifts)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    SIXTRL_UINT64_T const num_of_drifts, 
    SIXTRL_REAL_T   const* SIXTRL_RESTRICT lengths, 
    SIXTRL_INT64_T  const* SIXTRL_RESTRICT element_ids );

NS(Drift) NS(BeamElements_add_drifts_exact)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    SIXTRL_UINT64_T const num_of_drifts, 
    SIXTRL_REAL_T   const* SIXTRL_RESTRICT lengths, 
    SIXTRL_INT64_T  const* SIXTRL_RESTRICT element_ids );

/* ************************************************************************ */
/* *********     Implementation of inline functions and methods     ******* */
/* ************************************************************************ */

SIXTRL_INLINE NS(BeamElements)* NS(BeamElements_preset)( 
    NS(BeamElements)* beam_elements )
{
    return NS(BlocksContainer_preset)( beam_elements );
}

SIXTRL_INLINE void NS(BeamElements_clear)( 
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements )
{
    NS(BlocksContainer_clear)( beam_elements );
    return;
}

SIXTRL_INLINE void NS(BeamElements_free)( 
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements )
{
    NS(BlocksContainer_free)( beam_elements );
    return;
}

SIXTRL_INLINE int NS(BeamElements_init)( 
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_size_t) const blocks_capacity, 
    NS(block_size_t) const data_capacity )
{
    return NS(BlocksContainer_init)( 
        beam_elements, blocks_capacity, data_capacity );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(BeamElements_set_info_begin_alignment)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_alignment_t) const begin_alignment )
{
    return NS(BlocksContainer_set_info_begin_alignment)( 
        beam_elements, begin_alignment );    
}

SIXTRL_INLINE int NS(BeamElements_set_data_begin_alignment)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_alignment_t) const begin_alignment )
{
    return NS(BlocksContainer_set_data_begin_alignment)( 
        beam_elements, begin_alignment );    
}

SIXTRL_INLINE int NS(BeamElements_set_data_alignment)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_alignment_t) const alignment )
{
    return NS(BlocksContainer_set_data_alignment)( 
        beam_elements, alignment );    
}

SIXTRL_INLINE int NS(BeamElements_set_info_alignment )(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_alignment_t) const alignment )
{
    return NS(BlocksContainer_set_info_alignment)( 
        beam_elements, alignment );    
}

SIXTRL_INLINE void NS(BeamElements_reserve_num_blocks)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_size_t) const new_block_capacity )
{
    NS(BlocksContainer_reserve_num_blocks)( beam_elements, new_block_capacity );
    return;
}

SIXTRL_INLINE void NS(BeamElements_reserve_for_data)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_size_t) const new_data_capacity )
{
    NS(BlocksContainer_reserve_for_data)( beam_elements, new_data_capacity );
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(block_alignment_t) 
NS(BeamElements_get_info_alignment)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements )
{
    return NS(BlocksContainer_get_info_alignment)( beam_elements );
}

SIXTRL_INLINE NS(block_alignment_t) NS(BeamElements_get_data_alignment)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements )
{
    return NS(BlocksContainer_get_data_alignment)( beam_elements );
}

SIXTRL_INLINE NS(block_alignment_t) NS(BeamElements_get_info_begin_alignment)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements )
{
    return NS(BlocksContainer_get_info_begin_alignment)( beam_elements );
}

SIXTRL_INLINE NS(block_alignment_t) NS(BeamElements_get_data_begin_alignment)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements )
{
    return NS(BlocksContainer_get_data_begin_alignment)( beam_elements );
}

SIXTRL_INLINE NS(block_size_t) NS(BeamElements_get_data_capacity)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements )
{
    return NS(BlocksContainer_get_data_capacity)( beam_elements );
}

SIXTRL_INLINE NS(block_size_t) NS(BeamElements_get_data_size)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements )
{
    return NS(BlocksContainer_get_data_size)( beam_elements );
}

SIXTRL_INLINE NS(block_size_t) NS(BeamElements_get_block_capacity)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements )
{
    return NS(BlocksContainer_get_block_capacity)( beam_elements );
}

SIXTRL_INLINE NS(block_size_t) NS(BeamElements_get_num_of_blocks)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements )
{
    return NS(BlocksContainer_get_num_of_blocks)( beam_elements );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE unsigned char const* 
NS(BeamElements_get_const_ptr_data_begin)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements )
{
    return NS(BlocksContainer_get_const_ptr_data_begin)( beam_elements );
}

SIXTRL_INLINE unsigned char* NS(BeamElements_get_ptr_data_begin)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements )
{
    return NS(BlocksContainer_get_ptr_data_begin)( beam_elements );
}

SIXTRL_INLINE NS(BlockInfo) const* NS(BeamElements_get_const_block_infos_begin)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements )
{
    return NS(BlocksContainer_get_const_block_infos_begin)( beam_elements );
}

SIXTRL_INLINE NS(BlockInfo) const* NS(BeamElements_get_const_block_infos_end)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements )
{
    return NS(BlocksContainer_get_const_block_infos_end)( beam_elements );
}


SIXTRL_INLINE NS(BlockInfo)* NS(BeamElements_get_block_infos_begin)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements )
{
    return NS(BlocksContainer_get_block_infos_begin)( beam_elements );
}

SIXTRL_INLINE NS(BlockInfo)* NS(BeamElements_get_infos_end)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements )
{
    return NS(BlocksContainer_get_infos_end)( beam_elements );
}

SIXTRL_INLINE NS(BlockInfo) NS(BeamElements_get_block_info_by_index)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements, 
    NS(block_size_t) const block_index )
{
    return NS(BlocksContainer_get_block_info_by_index)( 
        beam_elements, block_index );
}

SIXTRL_INLINE NS(BlockInfo) const* NS(BeamElements_get_const_ptr_to_block_info_by_index)(
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements, 
    NS(block_size_t) const block_index )
{
    return NS(BlocksContainer_get_const_ptr_to_block_info_by_index)(
        beam_elements, block_index );
}

SIXTRL_INLINE NS(BlockInfo)* NS(BeamElements_get_ptr_to_block_info_by_index)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_size_t) const block_index )
{
    return NS(BlocksContainer_get_ptr_to_block_info_by_index)(
        beam_elements, block_index );
}

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__ */
