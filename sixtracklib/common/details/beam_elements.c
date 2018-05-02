#include "sixtracklib/common/beam_elements.h"

#include "sixtracklib/_impl/definitions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/common/mem_pool.h"
#include "sixtracklib/common/block_info.h"
#include "sixtracklib/common/impl/block_info_impl.h"

extern NS(Drift) NS(BeamElements_add_drift)( 
    NS(BeamElements)* beam_elements, SIXTRL_REAL_T  const length, 
    SIXTRL_INT64_T const element_id );

extern NS(Drift) NS(BeamElements_add_drift_exact)( 
    NS(BeamElements)* beam_elements, SIXTRL_REAL_T const length, 
    SIXTRL_INT64_T const element_id );

extern NS(Drift) NS(BeamElements_add_drifts)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    SIXTRL_UINT64_T const num_of_drifts, 
    SIXTRL_REAL_T   const* SIXTRL_RESTRICT lengths, 
    SIXTRL_INT64_T  const* SIXTRL_RESTRICT element_ids );

extern NS(Drift) NS(BeamElements_add_drifts_exact)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    SIXTRL_UINT64_T const num_of_drifts, 
    SIXTRL_REAL_T   const* SIXTRL_RESTRICT lengths, 
    SIXTRL_INT64_T  const* SIXTRL_RESTRICT element_ids );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC NS(Drift) NS(BeamElements_add_drift_type_block)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements, 
    SIXTRL_REAL_T  const length, 
    SIXTRL_INT64_T const element_id, 
    NS(BlockType)  const type_id );

SIXTRL_STATIC NS(Drift) NS(BeamElements_add_multi_drifts_type_block)(
    NS(BeamElements)* beam_elements, 
    SIXTRL_REAL_T   const* SIXTRL_RESTRICT length, 
    NS(element_id_t) const* SIXTRL_RESTRICT element_id, 
    NS(block_num_elements_t) const  num_elements,
    NS(BlockType)   const  type_id );

/* ------------------------------------------------------------------------- */

NS(Drift) NS(BeamElements_add_drift_type_block)(
    NS(BeamElements)* SIXTRL_RESTRICT elements, SIXTRL_REAL_T const length, 
    NS(element_id_t) const element_id, NS(BlockType) const type_id )
{
    NS(Drift) drift;
    
    if( ( elements != 0 ) && 
        ( NS(BeamElements_get_block_capacity)( elements ) > 
          NS(BeamElements_get_num_of_blocks)(  elements ) ) &&
        ( ( type_id == NS(BLOCK_TYPE_DRIFT) ) ||
          ( type_id == NS(BLOCK_TYPE_DRIFT_EXACT ) ) ) )
    {
        NS(block_alignment_t) const info_align = 
            NS(BeamElements_get_info_alignment)( elements );
            
        NS(block_alignment_t) const data_align =
            NS(BeamElements_get_data_alignment)( elements );
        
        NS(MemPool) rollback_info_store = elements->info_store;
        NS(MemPool) rollback_data_store = elements->data_store;
        
        NS(AllocResult) info_result = NS(MemPool_append_aligned)(
            &elements->info_store, sizeof( NS(BlockInfo) ), info_align );
        
        NS(block_size_t) const mem_offset = NS(MemPool_get_next_begin_offset)( 
            &elements->data_store, data_align );
        
        NS(block_size_t) const max_num_bytes_on_mem = 
            NS(MemPool_get_capacity)( &elements->data_store );
        
        if( ( NS(AllocResult_valid)( &info_result ) ) &&
            ( max_num_bytes_on_mem > mem_offset ) )
        {
            NS(BlockInfo)* block_info = 
                ( NS(BlockInfo)* )NS(AllocResult_get_pointer)( &info_result );
            
            unsigned char* data_mem_begin = 
                NS(BeamElements_get_ptr_data_begin)( elements );
                
            NS(BlockInfo_set_mem_offset)( block_info, mem_offset );
            NS(BlockInfo_set_type_id)( block_info, type_id );
            NS(BlockInfo_set_common_alignment)( block_info, data_align );
            NS(BlockInfo_set_num_elements)( 
                block_info, ( NS(block_num_elements_t) )1 );
            
            NS(Drift_preset)( &drift );
            NS(Drift_set_num_elements)( &drift, ( NS(block_num_elements_t) )1 );
            NS(Drift_set_type_id)( &drift, type_id );
            
            if( 0 == NS(Drift_create_one_on_memory)( &drift, block_info, 0, 
                data_mem_begin, max_num_bytes_on_mem, length, element_id ) )
            {
                ++elements->num_blocks;
                
                NS(MemPool_increment_size)( 
                    &elements->data_store, 
                    NS(BlockInfo_get_mem_offset)( block_info ) + 
                    NS(BlockInfo_get_num_of_bytes)( block_info ) );
                
                return drift;
            }
        }
        
        /* if we are here, something went wrong -> rollback and return an 
         * invalid drift! */
        
        SIXTRL_ASSERT( NS(Drift_get_type_id)( &drift ) == 
                       NS(BLOCK_TYPE_INVALID ) );
        
        elements->info_store = rollback_info_store;
        elements->data_store = rollback_data_store;
    }
    
    NS(Drift_preset)( &drift );
    return drift;
}

NS(Drift) NS(BeamElements_add_multi_drifts_type_block)(
    NS(BeamElements)* SIXTRL_RESTRICT elements, 
    SIXTRL_REAL_T const* SIXTRL_RESTRICT lengths, 
    NS(element_id_t) const* SIXTRL_RESTRICT element_ids, 
    NS(block_num_elements_t) const num_elements, 
    NS(BlockType)   const type_id )
{
    NS(Drift) drift;
    
    if( ( elements != 0 ) && 
        ( num_elements != ( NS(block_num_elements_t) )0u ) &&
        ( NS(BeamElements_get_block_capacity)( elements ) > 
          NS(BeamElements_get_num_of_blocks)(  elements ) ) &&        
        ( ( type_id == NS(BLOCK_TYPE_DRIFT) ) ||
          ( type_id == NS(BLOCK_TYPE_DRIFT_EXACT ) ) ) )
    {
        NS(MemPool) rollback_info_store = elements->info_store;
        NS(MemPool) rollback_data_store = elements->data_store;
        
        NS(block_alignment_t) const info_align =
            NS(BeamElements_get_info_alignment)( elements );
        
        NS(AllocResult) info_result = NS(MemPool_append_aligned)(
            &elements->info_store, sizeof( NS(BlockInfo) ), info_align );
        
        NS(block_alignment_t) const data_align = 
            NS(BeamElements_get_data_alignment)( elements );
        
        NS(block_size_t) const mem_offset = NS(MemPool_get_next_begin_offset)( 
            &elements->data_store, data_align );
        
        NS(block_size_t) const max_num_bytes_on_mem = 
            NS(MemPool_get_capacity)( &elements->data_store );
            
        if( ( NS(AllocResult_valid)( &info_result ) ) &&
            ( max_num_bytes_on_mem > mem_offset ) )
        {
            NS(BlockInfo)* block_info = 
                ( NS(BlockInfo)* )NS(AllocResult_get_pointer)( &info_result );
            
            unsigned char* data_mem_begin = 
                NS(BeamElements_get_ptr_data_begin)( elements );
                
            NS(BlockInfo_set_mem_offset)( block_info, mem_offset );
            NS(BlockInfo_set_type_id)( block_info, type_id );
            NS(BlockInfo_set_common_alignment)( block_info, data_align );
            NS(BlockInfo_set_num_elements)( block_info, num_elements );
            
            NS(Drift_preset)( &drift );
            NS(Drift_set_num_elements)( &drift, num_elements );
            NS(Drift_set_type_id)( &drift, type_id );
            
            if( 0 == NS(Drift_create_on_memory)( &drift, block_info, 0, 
                data_mem_begin, max_num_bytes_on_mem, lengths, element_ids ) )
            {
                ++elements->num_blocks;
                
                NS(MemPool_increment_size)( 
                    &elements->data_store, 
                    NS(BlockInfo_get_mem_offset)( block_info ) + 
                    NS(BlockInfo_get_num_of_bytes)( block_info ) );
                
                return drift;
            }
        }
        
        /* if we are here, something went wrong -> rollback and return an 
         * invalid drift! */
        
        SIXTRL_ASSERT( NS(Drift_get_type_id)( &drift ) == 
                       NS(BLOCK_TYPE_INVALID ) );
        
        elements->info_store = rollback_info_store;
        elements->data_store = rollback_data_store;
    }
    
    NS(Drift_preset)( &drift );
    return drift;
}
        
NS(Drift) NS(BeamElements_add_drift_exact)( 
    NS(BeamElements)* SIXTRL_RESTRICT elements, 
    SIXTRL_REAL_T  const length, 
    SIXTRL_INT64_T const element_id )
{
    return NS(BeamElements_add_drift_type_block)( 
        elements, length, element_id, NS(BLOCK_TYPE_DRIFT_EXACT) );
}

NS(Drift) NS(BeamElements_add_drift)( 
    NS(BeamElements)* SIXTRL_RESTRICT elements, 
    SIXTRL_REAL_T  const length, 
    SIXTRL_INT64_T const element_id )
{
    return NS(BeamElements_add_drift_type_block)( 
        elements, length, element_id, NS(BLOCK_TYPE_DRIFT) );
}

NS(Drift) NS(BeamElements_add_drifts)(
    NS(BeamElements)* SIXTRL_RESTRICT elements, 
    SIXTRL_UINT64_T const num_of_drifts, 
    SIXTRL_REAL_T   const* SIXTRL_RESTRICT lengths, 
    SIXTRL_INT64_T  const* SIXTRL_RESTRICT element_ids )
{
    return NS(BeamElements_add_multi_drifts_type_block)( 
        elements, lengths, element_ids, num_of_drifts, NS(BLOCK_TYPE_DRIFT ) );
}

NS(Drift) NS(BeamElements_BeamElements_add_drifts_exact)(
    NS(BeamElements)* SIXTRL_RESTRICT elements, 
    SIXTRL_UINT64_T const num_of_drifts, 
    SIXTRL_REAL_T   const* SIXTRL_RESTRICT lengths, 
    SIXTRL_INT64_T  const* SIXTRL_RESTRICT element_ids )
{
    return NS(BeamElements_add_multi_drifts_type_block)( elements, lengths, 
        element_ids, num_of_drifts, NS(BLOCK_TYPE_DRIFT_EXACT ) );
}

/* end: sixtracklib/common/details/beam_elements.c */
