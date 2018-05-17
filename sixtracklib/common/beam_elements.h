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

SIXTRL_STATIC int NS(BeamElements_assemble)(
    NS(BeamElements)* SIXTRL_RESTRICT container,
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT block_infos_begin,
    NS(block_size_t) const num_of_blocks,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT data_mem_begin,
    NS(block_size_t) const data_num_of_bytes );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC int NS(BeamElements_has_info_store)(
    const NS(BeamElements) *const SIXTRL_RESTRICT container );

SIXTRL_STATIC int NS(BeamElements_has_data_store)(
    const NS(BeamElements) *const SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(MemPool) const* 
NS(BeamElements_get_const_ptr_info_store)(
    NS(BeamElements)* SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(MemPool) const* 
NS(BeamElements_get_const_ptr_data_store)(
    NS(BeamElements)* SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(MemPool)* 
NS(BeamElements_get_ptr_info_store)(
    NS(BeamElements)* SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(MemPool)* 
NS(BeamElements_get_ptr_data_store)(
    NS(BeamElements)* SIXTRL_RESTRICT container );

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

SIXTRL_STATIC NS(BlockInfo)* NS(BeamElements_get_block_infos_end)(
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

SIXTRL_STATIC NS(BlockInfo)* NS(BeamElements_create_beam_element)(
    void* SIXTRL_RESTRICT ptr_beam_element,
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(BlockType) const type_id );

SIXTRL_STATIC int NS(BeamElements_get_beam_element)(
    void* SIXTRL_RESTRICT ptr_beam_element,
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_num_elements_t) const block_index );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC NS(Drift) NS(BeamElements_add_drift)( 
    NS(BeamElements)* beam_elements, SIXTRL_REAL_T  const length, 
    SIXTRL_INT64_T const element_id );

SIXTRL_STATIC NS(Drift) NS(BeamElements_add_drift_exact)( 
    NS(BeamElements)* beam_elements, SIXTRL_REAL_T const length, 
    SIXTRL_INT64_T const element_id );

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

SIXTRL_INLINE int NS(BeamElements_assemble)(
    NS(BeamElements)* SIXTRL_RESTRICT container,
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT block_infos_begin,
    NS(block_size_t) const num_of_blocks,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT data_mem_begin,
    NS(block_size_t) const data_num_of_bytes )
{
    return NS(BlocksContainer_assemble)( ( NS(BlocksContainer)* ) container,
        block_infos_begin, num_of_blocks, data_mem_begin, data_num_of_bytes );
}

/* ------------------------------------------------------------------------- */


SIXTRL_INLINE int NS(BeamElements_has_info_store)(
    const NS(BeamElements) *const SIXTRL_RESTRICT container )
{
    return NS(BlocksContainer_has_info_store)( container );
}

SIXTRL_INLINE int NS(BeamElements_has_data_store)(
    const NS(BeamElements) *const SIXTRL_RESTRICT container )
{
    return NS(BlocksContainer_has_data_store)( container );
}

SIXTRL_INLINE NS(MemPool) const* 
NS(BeamElements_get_const_ptr_info_store)(
    NS(BeamElements)* SIXTRL_RESTRICT container )
{
    return NS(BlocksContainer_get_const_ptr_info_store)( container );
}

SIXTRL_INLINE NS(MemPool) const* NS(BeamElements_get_const_ptr_data_store)(
    NS(BeamElements)* SIXTRL_RESTRICT container )
{
    return NS(BlocksContainer_get_const_ptr_data_store)( container );
}

SIXTRL_INLINE NS(MemPool)* NS(BeamElements_get_ptr_info_store)(
    NS(BeamElements)* SIXTRL_RESTRICT container )
{
    return NS(BlocksContainer_get_ptr_info_store)( container );
}

SIXTRL_INLINE NS(MemPool)* NS(BeamElements_get_ptr_data_store)(
    NS(BeamElements)* SIXTRL_RESTRICT container )
{
    return NS(BlocksContainer_get_ptr_data_store)( container );
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

SIXTRL_INLINE NS(BlockInfo)* NS(BeamElements_get_block_infos_end)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements )
{
    return NS(BlocksContainer_get_block_infos_end)( beam_elements );
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

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(Drift) NS(BeamElements_add_drift)( 
    NS(BeamElements)* beam_elements, SIXTRL_REAL_T  const length, 
    SIXTRL_INT64_T const element_id )
{
    NS(Drift) drift;
    NS(BlockInfo)* block_info = NS(BeamElements_create_beam_element)(
        &drift, beam_elements, NS(BLOCK_TYPE_DRIFT) );
    
    if( block_info != 0 )
    {
        NS(Drift_set_length_value)( &drift, length );
        NS(Drift_set_element_id_value)( &drift, element_id );
    }
    else
    {
        NS(Drift_preset)( &drift );
    }
    
    return drift;
}

SIXTRL_INLINE NS(Drift) NS(BeamElements_add_drift_exact)( 
    NS(BeamElements)* beam_elements, SIXTRL_REAL_T const length, 
    SIXTRL_INT64_T const element_id )
{
    NS(Drift) drift;
    NS(BlockInfo)* block_info = NS(BeamElements_create_beam_element)(
        &drift, beam_elements, NS(BLOCK_TYPE_DRIFT_EXACT) );
    
    if( block_info != 0 )
    {
        NS(Drift_set_length_value)( &drift, length );
        NS(Drift_set_element_id_value)( &drift, element_id );
    }
    else
    {
        NS(Drift_preset)( &drift );
    }
    
    return drift;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(BeamElements_get_beam_element)(
    void* SIXTRL_RESTRICT ptr_beam_element,
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    NS(block_num_elements_t) const block_index )
{
    int success = -1;
    
    NS(BlockInfo) const* block_info = 
        NS(BeamElements_get_const_ptr_to_block_info_by_index)(
            beam_elements, block_index );
        
    NS(BlockType) const type_id = NS(BlockInfo_get_type_id)( block_info );
        
    if( ( ptr_beam_element != 0 ) && ( block_info != 0 ) && 
        ( type_id != NS(BLOCK_TYPE_INVALID) ) )
    {
        switch( type_id )
        {
            case NS(BLOCK_TYPE_DRIFT):
            case NS(BLOCK_TYPE_DRIFT_EXACT):
            {
                NS(Drift)* drift = ( NS(Drift)* )ptr_beam_element;
                NS(Drift_preset)( drift );
                
                success = NS(Drift_remap_from_memory)( drift, block_info,
                    NS(BeamElements_get_ptr_data_begin)( beam_elements ),
                    NS(BeamElements_get_data_capacity)( beam_elements ) );
                
                break;
            }
            
            default:
            {
                success = -1;
            }
        };
    }
    
    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(BlockInfo)* NS(BeamElements_create_beam_element)(
    void* SIXTRL_RESTRICT ptr_beam_element,
    NS(BeamElements)* SIXTRL_RESTRICT elements, NS(BlockType) const type_id )
{
    NS(BlockInfo)* block_info = 0;
    
    NS(MemPool)* ptr_info_store = 0;
    NS(MemPool)* ptr_data_store = 0;
    
    NS(MemPool) rollback_info_store;
    NS(MemPool) rollback_data_store;
        
    if( ( !NS(BeamElements_has_data_store)( elements ) ) ||
        ( !NS(BeamElements_has_info_store)( elements ) ) )
    {
        return block_info;
    }
    
    ptr_info_store = NS(BeamElements_get_ptr_info_store)( elements );    
    ptr_data_store = NS(BeamElements_get_ptr_data_store)( elements );
    
    SIXTRL_ASSERT( ( ptr_info_store != 0 ) && ( ptr_data_store != 0 ) );
        
    rollback_info_store = *ptr_data_store;
    rollback_data_store = *ptr_info_store;    
    
    if( ( elements != 0 ) && ( ptr_beam_element != 0 ) && 
        ( type_id != NS(BLOCK_TYPE_INVALID) ) )
    {
        int success = -1;
        
        NS(block_alignment_t) const data_align =
            NS(BeamElements_get_data_alignment)( elements );
        
        NS(block_alignment_t) const info_align = 
            NS(BeamElements_get_info_alignment)( elements );
            
        NS(AllocResult) info_result = NS(MemPool_append_aligned)(
            ptr_info_store, sizeof( NS(BlockInfo) ), info_align );
        
        SIXTRL_GLOBAL_DEC unsigned char* mem_begin =
            NS(BeamElements_get_ptr_data_begin)( elements );
        
        NS(block_size_t) const mem_offset = NS(MemPool_get_next_begin_offset)( 
            ptr_data_store, data_align );
        
        NS(block_size_t) const max_num_bytes_on_mem = 
            NS(MemPool_get_capacity)( ptr_data_store );
            
        if( ( NS(AllocResult_valid)( &info_result ) ) && ( mem_begin != 0 ) && 
            ( max_num_bytes_on_mem > mem_offset ) )
        {
            SIXTRL_STATIC NS(block_num_elements_t) const ONLY_ONE_ELEMENT = 
                ( NS(block_num_elements_t) )1u;
            
            block_info = ( NS(BlockInfo)* )NS(AllocResult_get_pointer)( 
                &info_result );
            
            NS(BlockInfo_set_mem_offset)( block_info, mem_offset );
            NS(BlockInfo_set_type_id)( block_info, type_id );
            NS(BlockInfo_set_common_alignment)( block_info, data_align );            
            NS(BlockInfo_set_num_elements)( block_info, ONLY_ONE_ELEMENT );
            
            switch( type_id )
            {
                case NS(BLOCK_TYPE_DRIFT):
                case NS(BLOCK_TYPE_DRIFT_EXACT):
                {
                    NS(Drift)* drift = ( NS(Drift)* )ptr_beam_element;
                    NS(Drift_preset)( drift );
                    NS(Drift_set_type_id)( drift, type_id );
                    
                    success = NS(Drift_create_on_memory)(
                        drift, block_info, mem_begin, max_num_bytes_on_mem );
                    
                    break;
                }
                
                default:
                {
                    success = -1;
                }
            };
        }
        
        if( ( success == 0 ) && ( block_info != 0 ) )
        {
            
            NS(MemPool_increment_size)( ptr_data_store, 
                NS(BlockInfo_get_mem_offset)( block_info ) + 
                NS(BlockInfo_get_num_of_bytes)( block_info ) );
            
            ++elements->num_blocks;
            elements->data_raw_size = 
                NS(MemPool_get_size)( ptr_data_store );
            
            return block_info;
        }
        else if( ( ptr_data_store != 0 ) && ( ptr_info_store != 0 ) )
        {
            *ptr_info_store = rollback_info_store;
            *ptr_data_store = rollback_data_store;
        
            block_info = 0;
        }
    }
    
    return block_info;
}

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__ */
