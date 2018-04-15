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
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    const NS(MemPool) *const SIXTRL_RESTRICT pool, 
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_alignment );

extern SIXTRL_SIZE_T NS(Drift_predict_required_capacity_for_packing)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin, 
    SIXTRL_SIZE_T const alignment );

/* ------------------------------------------------------------------------- */

extern SIXTRL_SIZE_T NS(Drift_map_to_pack_flat_memory_aligned)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    unsigned char* mem_begin, SIXTRL_SIZE_T const alignment );

extern struct NS(AllocResult) NS(Drift_map_to_pack_aligned)( 
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    NS(MemPool)* SIXTRL_RESTRICT pool, SIXTRL_SIZE_T const alignment );

/* ========================================================================= */
/* ========================================================================= */

NS(Drift)* NS(Drift_preset)( NS(Drift)* SIXTRL_RESTRICT drift )
{
    if( drift != 0 )
    {
        drift->num_elem   = ( SIXTRL_SIZE_T )0u;
        drift->type_id    = NS(ELEMENT_TYPE_NONE);
        drift->length     = 0;
        drift->element_id = 0;
    }
    
    return drift;
}

/* ------------------------------------------------------------------------- */

SIXTRL_SIZE_T NS(Drift_predict_required_mempool_capacity_for_packing)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    const NS(MemPool) *const SIXTRL_RESTRICT pool, 
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_alignment )
{
    SIXTRL_STATIC SIXTRL_SIZE_T const num_of_attributes = ( SIXTRL_SIZE_T )2u;
    SIXTRL_SIZE_T const num_of_elements = NS(Drift_get_size)( drift );
    SIXTRL_SIZE_T const attr_sizes[] = 
    {
        sizeof( SIXTRL_REAL_T ), sizeof( SIXTRL_INT64_T ) 
    };
    
    return NS(Block_predict_required_mempool_capacity_for_packing)( pool, 
        ptr_alignment, num_of_elements, &attr_sizes[ 0 ], num_of_attributes );
}

SIXTRL_SIZE_T NS(Drift_predict_required_capacity_for_packing)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin, 
    SIXTRL_SIZE_T const alignment )
{
    SIXTRL_SIZE_T const num_of_attributes = ( SIXTRL_SIZE_T )2u;
    SIXTRL_SIZE_T const num_of_elements = NS(Drift_get_size)( drift );
    SIXTRL_SIZE_T const attr_sizes[] = 
    {
        sizeof( SIXTRL_REAL_T ), sizeof( SIXTRL_INT64_T ) 
    };
    
    return NS(Block_predict_required_capacity_for_packing)( ptr_mem_begin, 
        alignment, num_of_elements, &attr_sizes[ 0 ], num_of_attributes );
}

/* ------------------------------------------------------------------------- */

SIXTRL_SIZE_T NS(Drift_map_to_pack_flat_memory_aligned)(
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    unsigned char* mem_begin, SIXTRL_SIZE_T const alignment )
{
    static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
    SIXTRL_SIZE_T serialized_length = ZERO_SIZE;
    
    if( ( drift != 0 ) && ( mem_begin != 0 ) && ( alignment != ZERO_SIZE ) &&
        ( ( ( ( uintptr_t )mem_begin ) % alignment ) == ZERO_SIZE ) )
    {
        SIXTRL_SIZE_T temp;
        SIXTRL_SIZE_T attr_block_length;
        SIXTRL_SIZE_T current_size = ZERO_SIZE;
        SIXTRL_SIZE_T const u64_size = sizeof( SIXTRL_UINT64_T );
        
        unsigned char* it = mem_begin;
        unsigned char* offset_info_ptr = 0;
        ptrdiff_t distance_from_begin;
                
        SIXTRL_UINT64_T const pack_index = NS(Drift_get_type_id)( drift );
        SIXTRL_UINT64_T const num_of_elements = NS(Drift_get_size)( drift );
        SIXTRL_UINT64_T const num_of_attributes = ( SIXTRL_UINT64_T )2u;
        SIXTRL_UINT64_T offset = UINT64_C( 0 );
                
        /* preset the "length" field in the header with 0 -> this will be over-
         * written with the proper length at the end of the serialization */
        memset( it, ( int )0, u64_size ); 
        it = it + u64_size;
        current_size += u64_size;
        
        memcpy( it, &pack_index, u64_size );
        it = it + u64_size;
        current_size += u64_size;
        
        memcpy( it, &num_of_elements, u64_size );
        it = it + u64_size;
        current_size += u64_size;
        
        memcpy( it, &num_of_attributes, u64_size );
        it = it + u64_size;
        current_size += u64_size;
        
        temp = u64_size * num_of_attributes;
        
        offset_info_ptr = it;
        it = it + temp;
        current_size += temp;
        
        temp = ( current_size / alignment ) * alignment;
        
        if( temp < current_size )
        {
            current_size = temp + alignment;
            it = mem_begin + current_size;
        }
        
        assert( ( current_size % alignment ) == ZERO_SIZE );
        assert( ( ( ( uintptr_t )it ) % alignment ) == ZERO_SIZE );
        
        /* ----------------------------------------------------------------- */
        /* length: */
        
        distance_from_begin = ( it - mem_begin );
        assert( distance_from_begin > 0 );
        
        offset = ( SIXTRL_UINT64_T )distance_from_begin;
        memcpy( offset_info_ptr, &offset, u64_size );
        offset_info_ptr   = offset_info_ptr + u64_size;
        attr_block_length = num_of_elements * sizeof( SIXTRL_REAL_T );
        
        temp = ( attr_block_length / alignment ) * alignment;
        
        if( attr_block_length > temp )
        {
            attr_block_length = temp + alignment;
        }
        
        it = it + attr_block_length;
        
        /* ----------------------------------------------------------------- */
        /* elem_id: */
        
        distance_from_begin = ( it - mem_begin );
        assert( distance_from_begin > 0 );
        
        offset = ( SIXTRL_UINT64_T )distance_from_begin;
        memcpy( offset_info_ptr, &offset, u64_size );
        offset_info_ptr = offset_info_ptr + u64_size;
        
        attr_block_length = num_of_elements * sizeof( SIXTRL_INT64_T );
        temp = ( attr_block_length / alignment ) * alignment;
        
        if( temp < attr_block_length )
        {
            attr_block_length = temp + alignment;
        }
        
        it = it + attr_block_length;
        
        distance_from_begin = ( it - mem_begin );
        
        if( distance_from_begin > 0 )
        {
            /* update the length part of the header */
            SIXTRL_UINT64_T const len = ( SIXTRL_UINT64_T )distance_from_begin;
            serialized_length = len;
            memcpy( mem_begin, &len, u64_size );
        }
    }
    
    return serialized_length;
}

/* ------------------------------------------------------------------------- */

NS(AllocResult) NS(Drift_map_to_pack_aligned)( 
    const struct NS(Drift) *const SIXTRL_RESTRICT drift, 
    NS(MemPool)* SIXTRL_RESTRICT pool, SIXTRL_SIZE_T alignment )
{
    static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
    
    NS(AllocResult) result;    
    NS(AllocResult_preset)( &result );
    
    if( ( drift != 0 ) && ( pool != 0 )  )        
    {
        SIXTRL_SIZE_T chunk_size = NS(MemPool_get_chunk_size)( pool );
        
        SIXTRL_SIZE_T const required_capacity = 
            NS(Drift_predict_required_mempool_capacity_for_packing)(
                drift, pool, &alignment );
            
        if( ( required_capacity > ZERO_SIZE ) && ( alignment != ZERO_SIZE ) )
        {
            NS(MemPool) const rollback_mem_pool = *pool;
            
            assert( ( alignment % chunk_size ) == ZERO_SIZE );
            
            result = NS(MemPool_append_aligned)( 
                pool, required_capacity, alignment );
            
            if( NS(AllocResult_valid)( &result ) )
            {
                SIXTRL_SIZE_T const serialized_length = 
                    NS(Drift_map_to_pack_flat_memory_aligned)( drift,
                        NS(AllocResult_get_pointer)( &result ), alignment );
                    
                if( ( serialized_length >  ZERO_SIZE ) && 
                    ( serialized_length <= required_capacity ) )
                {
                    NS(AllocResult_set_length)( &result, serialized_length );
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
