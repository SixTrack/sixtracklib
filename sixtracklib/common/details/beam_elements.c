#include "sixtracklib/common/beam_elements.h"

#include "sixtracklib/_impl/definitions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/common/mem_pool.h"

extern NS(BeamElements)* NS(BeamElements_preset)( NS(BeamElements)* beam_elements );

extern NS(BeamElements)* NS(BeamElements_new)( 
    SIXTRL_SIZE_T const elements_capacity, size_t const   raw_capacity, 
    SIXTRL_SIZE_T const element_alignment, SIXTRL_SIZE_T const begin_alignment );

extern void NS(BeamElements_free)( NS(BeamElements)* beam_elements );

extern bool NS(BeamElements_add_drift)( 
    NS(BeamElements)* beam_elements, SIXTRL_REAL_T  const length, 
    SIXTRL_INT64_T const element_id );

extern bool NS(BeamElements_add_drift_exact)( 
    NS(BeamElements)* beam_elements, SIXTRL_REAL_T const length, 
    SIXTRL_INT64_T const element_id );

SIXTRL_STATIC bool NS(BeamElements_add_drift_type)(
    NS(BeamElements)* beam_elements, SIXTRL_REAL_T const length, 
    SIXTRL_INT64_T const element_id, SIXTRL_UINT64_T const type_id );


NS(BeamElements)* NS(BeamElements_preset)( NS(BeamElements)* elements )
{
    if( elements != 0 )
    {
        elements->info_begin   = 0;
        elements->data_begin   = 0;
                    
        NS(MemPool_preset)( &elements->info_store );
        NS(MemPool_preset)( &elements->data_store );
        
        elements->num_elements         = ( SIXTRL_SIZE_T )0u;
        elements->elements_capacity    = ( SIXTRL_SIZE_T )0u;
        
        elements->raw_size             = ( SIXTRL_SIZE_T )0u;
        elements->raw_capacity         = ( SIXTRL_SIZE_T )0u;
        
        elements->begin_alignment      = ( SIXTRL_SIZE_T )0u;
        elements->element_alignment    = ( SIXTRL_SIZE_T )0u;
        
    }
    
    return elements;
}

NS(BeamElements)* NS(BeamElements_new)( 
    SIXTRL_SIZE_T const elements_capacity, size_t const   raw_capacity, 
    SIXTRL_SIZE_T const element_alignment, SIXTRL_SIZE_T const begin_alignment )
{    
    bool success = false;
    
    NS(BeamElements)* elements = NS(BeamElements_preset)(
        ( NS(BeamElements)* )malloc( sizeof( NS(BeamElements) ) ) );
    
    if( elements != 0 )
    {
        SIXTRL_SIZE_T const INFO_CAPACITY = 
            sizeof( NS(BeamElemInfo) ) * elements_capacity + begin_alignment;
            
        NS(MemPool_init)( &elements->info_store, INFO_CAPACITY, 
                          sizeof( NS(BeamElemInfo) ) );
        
        NS(MemPool_init)( &elements->data_store, 
                          raw_capacity + begin_alignment, 8u );
        
        elements->begin_alignment   = begin_alignment;
        elements->element_alignment = element_alignment;
        
        elements->elements_capacity = elements_capacity;
        elements->raw_capacity = raw_capacity;
        
        success = NS(MemPool_clear_to_aligned_position)(
            &elements->info_store, elements->begin_alignment );
        
        success &= NS(MemPool_clear_to_aligned_position)(
            &elements->data_store, elements->begin_alignment );
        
        if( success )
        {
            elements->info_begin = ( NS(BeamElemInfo)* )NS(MemPool_get_next_begin_pointer)( 
                &elements->info_store, elements->begin_alignment );
            
            elements->data_begin = ( unsigned char* )NS(MemPool_get_next_begin_pointer)( 
                &elements->data_store, elements->begin_alignment );
            
            success = ( ( elements->info_begin != 0 ) && 
                        ( elements->data_begin != 0 ) );
        }
    }
    
    if( !success )
    {
        NS(BeamElements_free)( elements );
        free( elements );
        elements = 0;
    }
    
    return elements;
}

void NS(BeamElements_free)( NS(BeamElements)* elements )
{
    if( elements != 0 )
    {
        NS(MemPool_free)( &elements->info_store );
        NS(MemPool_free)( &elements->data_store );
        
        NS(BeamElements_preset)( elements );        
    }
    
    return;    
}

bool NS(BeamElements_add_drift_type)(
    NS(BeamElements)* elements, SIXTRL_REAL_T const length, 
    SIXTRL_INT64_T const element_id, SIXTRL_UINT64_T const type_id )
{
    bool success = false;

    if( ( elements != 0 ) && 
        ( elements->elements_capacity > elements->num_elements ) )
    {
        NS(MemPool) rollback_info_store = elements->info_store;
        NS(MemPool) rollback_data_store = elements->data_store;
        
        NS(AllocResult) info_result = NS(MemPool_append)(
            &elements->info_store, sizeof( NS(BeamElemInfo) ) );
        
        NS(AllocResult) data_result = NS(MemPool_append)(
            &elements->data_store, sizeof( SIXTRL_REAL_T ) );
        
        if( ( NS(AllocResult_valid)( &info_result ) ) &&
            ( NS(AllocResult_valid)( &data_result ) ) )
        {
            NS(BeamElemInfo)* ptr_info = 
                ( NS(BeamElemInfo)* )NS(AllocResult_get_pointer)( &info_result );
            
            NS(AllocResult) add_data_result = NS(MemPool_append)(
                &elements->data_store, sizeof( SIXTRL_INT64_T ) );
            
            SIXTRL_UINT64_T num_bytes = 
                NS(AllocResult_get_length)( &data_result );
            
            unsigned char* data_begin = 
                NS(AllocResult_get_pointer)( &data_result );
            
            unsigned char* data_buffer_begin = elements->data_begin;
            
            unsigned char* data_end =
                NS(AllocResult_get_pointer)( &add_data_result );
                            
            if( ( NS(AllocResult_valid)( &add_data_result ) ) &&
                ( num_bytes >= sizeof( SIXTRL_REAL_T ) ) && ( ptr_info != 0 ) &&
                ( data_begin != 0 ) && ( data_end != 0 ) && 
                ( data_buffer_begin != 0 ) &&
                ( 0 <= ( data_begin - data_buffer_begin ) ) &&
                ( 0 <  ( data_end - data_begin ) ) )
            {
                unsigned char* ptr_store_length = data_begin;
                unsigned char* ptr_store_elemid = data_end;
                
                ptr_info->mem_offset = 
                    ( SIXTRL_UINT64_T )( data_begin - data_buffer_begin );
                
                data_end = data_end + 
                    NS(AllocResult_get_length)( &add_data_result );
                    
                ptr_info->num_bytes = 
                    ( SIXTRL_UINT64_T )( data_end - data_begin );
                    
                ptr_info->type_id    = type_id;
                ptr_info->element_id = element_id;
                
                memcpy( ptr_store_length, &length,     sizeof( SIXTRL_REAL_T  ) );
                memcpy( ptr_store_elemid, &element_id, sizeof( SIXTRL_INT64_T ) );
                
                success = true;
            }
        }        
        
        if( success )
        {
            ++elements->num_elements;
        }
        else
        {
            elements->info_store = rollback_info_store;
            elements->data_store = rollback_data_store;
        }
    }
    
    return success;
}

bool NS(BeamElements_add_drift_exact)( 
    NS(BeamElements)* elements, SIXTRL_REAL_T const length, 
    SIXTRL_INT64_T const element_id )
{
    return NS(BeamElements_add_drift_type)( elements, length, element_id, 3u );
}

bool NS(BeamElements_add_drift)( 
    NS(BeamElements)* elements, SIXTRL_REAL_T const length, 
    SIXTRL_INT64_T const element_id )
{
    return NS(BeamElements_add_drift_type)( elements, length, element_id, 2u );
}




