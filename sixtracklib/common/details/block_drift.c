#include "sixtracklib/common/block_drift.h"

#include <assert.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/block.h"
#include "sixtracklib/common/mem_pool.h"
#include "sixtracklib/common/impl/block_type.h"
#include "sixtracklib/common/impl/block_drift_type.h"


extern struct NS(Drift)* NS(Drift_preset)( 
    struct NS(Drift)* SIXTRL_RESTRICT drift );

/* ------------------------------------------------------------------------- */

extern SIXTRL_SIZE_T NS(Drift_predict_required_mempool_capacity_for_packing)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin, 
    SIXTRL_SIZE_T const chunk_size, SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_alignment );

extern SIXTRL_SIZE_T NS(Drift_predict_required_capacity_for_packing)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin, 
    SIXTRL_SIZE_T const alignment );

/* ------------------------------------------------------------------------- */

extern SIXTRL_SIZE_T NS(Drift_pack_to_flat_memory_aligned)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    unsigned char* mem_begin, SIXTRL_SIZE_T const alignment );

extern struct NS(AllocResult) NS(Drift_pack_aligned)( 
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    NS(MemPool)* SIXTRL_RESTRICT pool, SIXTRL_SIZE_T const alignment );

/* ========================================================================= */
/* ========================================================================= */

NS(Drift)* NS(Drift_preset)( NS(Drift)* SIXTRL_RESTRICT drift )
{
    if( drift != 0 )
    {
        drift->type_id    = NS(ELEMENT_TYPE_NONE);
        drift->length     = 0;
        drift->element_id = 0;
    }
    
    return drift;
}

/* ------------------------------------------------------------------------- */

SIXTRL_SIZE_T NS(Drift_predict_required_num_bytes_on_mempool_for_packing)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin,
    SIXTRL_SIZE_T const chunk_size, SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_alignment )
{
    SIXTRL_STATIC SIXTRL_SIZE_T const NUM_ELEMENTS   = ( SIXTRL_SIZE_T )1u;
    SIXTRL_STATIC SIXTRL_SIZE_T const NUM_ATTRIBUTES = ( SIXTRL_SIZE_T )2u;
    
    return NS(Block_predict_required_num_bytes_on_mempool_for_packing)( 
        ptr_mem_begin, chunk_size, ptr_alignment, NUM_ELEMENTS, NUM_ATTRIBUTES,
        (SIXTRL_SIZE_T[]){ sizeof( SIXTRL_REAL_T ), sizeof( SIXTRL_INT64_T ) } );
}

SIXTRL_SIZE_T NS(Drift_predict_required_num_bytes_for_packing)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin, 
    SIXTRL_SIZE_T const alignment )
{
    SIXTRL_STATIC SIXTRL_SIZE_T const NUM_ELEMENTS   = ( SIXTRL_SIZE_T )1u;
    SIXTRL_STATIC SIXTRL_SIZE_T const NUM_ATTRIBUTES = ( SIXTRL_SIZE_T )2u;
    
    return NS(Block_predict_required_num_bytes_for_packing)( 
        ptr_mem_begin, alignment, NUM_ELEMENTS, NUM_ATTRIBUTES,
        (SIXTRL_SIZE_T[]){ sizeof( SIXTRL_REAL_T ), sizeof( SIXTRL_INT64_T ) } );
}

/* ------------------------------------------------------------------------- */

SIXTRL_SIZE_T NS(Drift_pack_to_flat_memory_aligned)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    unsigned char* mem, SIXTRL_SIZE_T const alignment )
{
    SIXTRL_STATIC SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;    
    SIXTRL_STATIC SIXTRL_SIZE_T const U64_SIZE  = sizeof( SIXTRL_UINT64_T );
    SIXTRL_SIZE_T serial_len = ZERO_SIZE;
    
    if( ( drift != 0 ) && ( mem != 0 ) && ( alignment != ZERO_SIZE ) &&
        ( alignment >= U64_SIZE ) && ( ( alignment % U64_SIZE ) == ZERO_SIZE ) &&
        ( ( ( ( uintptr_t )mem ) % alignment ) == ZERO_SIZE ) )
    {
        NS(Drift) mapped;
        NS(BeamElementType) const type_id = NS(Drift_get_type_id)( drift );
        
        serial_len = NS(Drift_map_to_flat_memory)( &mapped, mem, type_id, alignment );
            
        if( serial_len > ZERO_SIZE )
        {
            SIXTRL_ASSERT( NS(Drift_get_type_id)( &mapped ) == type_id );
            SIXTRL_ASSERT( *( ( SIXTRL_UINT64_T const* )mem ) == serial_len );
            
            NS(Drift_set_length)( &mapped, NS(Drift_get_length)( drift ) );
            NS(Drift_set_element_id)( &mapped, NS(Drift_get_element_id)( drift ) );
        }
    }
    
    return serial_len;
}

/* ------------------------------------------------------------------------- */

NS(AllocResult) NS(Drift_pack_aligned)( 
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    NS(MemPool)* SIXTRL_RESTRICT pool, SIXTRL_SIZE_T alignment )
{
    static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
    
    NS(AllocResult) result;    
    NS(AllocResult_preset)( &result );
    
    if( ( drift != 0 ) && ( pool != 0 ) && ( alignment != ZERO_SIZE ) )
    {
        SIXTRL_SIZE_T const chunk_size = NS(MemPool_get_chunk_size)( pool );
        
        unsigned char* ptr_begin = 
            NS(MemPool_get_next_begin_pointer)(pool, alignment);
            
        SIXTRL_SIZE_T const predicted_num_bytes = 
            NS(Drift_predict_required_num_bytes_on_mempool_for_packing)(
                ptr_begin, chunk_size, &alignment );
            
        SIXTRL_SIZE_T const remaining_num_bytes = 
            NS(MemPool_get_remaining_bytes)( pool );
            
        if( ( predicted_num_bytes > ZERO_SIZE ) && 
            ( alignment != ZERO_SIZE ) && 
            ( remaining_num_bytes >= predicted_num_bytes ) )
        {
            NS(MemPool) const rollback_mem_pool = *pool;
            assert( ( alignment % chunk_size ) == ZERO_SIZE );
            
            result = NS(MemPool_append_aligned)( 
                pool, predicted_num_bytes, alignment );
            
            if( NS(AllocResult_valid)( &result ) )
            {
                SIXTRL_SIZE_T const serial_len = 
                    NS(Drift_pack_to_flat_memory_aligned)( drift,
                        NS(AllocResult_get_pointer)( &result ), alignment );
                    
                if( ( serial_len >  ZERO_SIZE ) && 
                    ( serial_len <= remaining_num_bytes ) )
                {
                    NS(AllocResult_set_length)( &result, serial_len );
                }
                else
                {
                    *pool = rollback_mem_pool;
                    NS(AllocResult_preset)( &result );
                }
            }
        }
    }
    
    return result;
}

/* ------------------------------------------------------------------------- */
