#include "sixtracklib/common/be_drift.h"
#include "sixtracklib/_impl/definitions.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/_impl/namespace_begin.h"
#include "sixtracklib/common/impl/block_info_impl.h"
#include "sixtracklib/common/impl/be_drift_impl.h"
#include "sixtracklib/common/impl/particles_impl.h"
#include "sixtracklib/common/block_info.h"
#include "sixtracklib/common/track.h"

/* -------------------------------------------------------------------------- */

extern int NS(Drift_create_from_single_drift)( 
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(DriftSingle)* SIXTRL_RESTRICT single_drift );

extern void NS(DriftSingle_track_particle)( 
    struct NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_UINT64_T const particle_index,
    const NS(DriftSingle) *const SIXTRL_RESTRICT single_drift  );

extern void NS(Drift_exact_track_particle)( 
    struct NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_UINT64_T const particle_index,
    const NS(DriftSingle) *const SIXTRL_RESTRICT single_drift  );

extern int NS(Drift_is_valid)( const NS(Drift) *const SIXTRL_RESTRICT drift );

extern int NS(Drift_write_to_bin_file)( 
    FILE* fp, const NS(Drift) *const SIXTRL_RESTRICT drift );

extern int NS(Drift_read_from_bin_file)( 
    FILE* fp, NS(Drift)* SIXTRL_RESTRICT drift );


/* -------------------------------------------------------------------------- */

int NS(Drift_create_from_single_drift)( 
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(DriftSingle)* SIXTRL_RESTRICT single_drift )
{
    int success = -1;
    
    if( ( drift != 0 ) && ( single_drift != 0 ) )
    {
        NS(Drift_set_type_id_num)( 
            drift, NS(DriftSingle_get_type_id_num)( single_drift ) );
        
        NS(Drift_assign_ptr_to_element_id)( 
            drift, NS(DriftSingle_get_element_id_ptr)( single_drift ) );
        
        NS(Drift_assign_ptr_to_length)( 
            drift, NS(DriftSingle_get_length_ptr)( single_drift ) );
        
        success = 0;
    }
    else if( drift != 0 )
    {
        NS(Drift_preset)( drift );
    }
    
    return success;
}

int NS(Drift_is_valid)( const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    int is_valid = 0;
    
    NS(BlockType) const type_id = NS(Drift_get_type_id)( drift );
        
    if( ( type_id == NS(BLOCK_TYPE_DRIFT )  ) ||
        ( type_id == NS(BLOCK_TYPE_DRIFT_EXACT ) ) )
    {
        is_valid = ( ( NS(Drift_get_element_id_value)( drift ) >= 0 ) &&
            ( NS(Drift_get_length_value)( drift ) >= ( SIXTRL_REAL_T )0.0 ) );
    }
    
    return is_valid;
}


int NS(Drift_write_to_bin_file)( 
    FILE* fp, const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    int success = -1;
    
    NS(BlockType) const type_id = NS(Drift_get_type_id)( drift );
    
    if( ( fp != 0 ) && ( drift != 0 ) &&
        ( ( type_id == NS(BLOCK_TYPE_DRIFT ) ) ||
          ( type_id == NS(BLOCK_TYPE_DRIFT_EXACT ) ) ) )
    {
        static size_t const ONE = ( size_t )1u;        
        size_t const  REAL_SIZE = sizeof( SIXTRL_REAL_T );
        size_t const  I64_SIZE  = sizeof( SIXTRL_INT64_T );
        
        NS(block_size_t) const num_attributes = ( NS(block_size_t) )2u;            
        NS(block_size_t) const attr_sizes[]   = { I64_SIZE, REAL_SIZE };        
        NS(block_size_t) const attr_counts[]  = { ONE, ONE };
        NS(block_num_elements_t) num_elements = 1;
        
        void* attr[] =
        {
            ( void* )NS(Drift_get_const_element_id)( drift ), 
            ( void* )NS(Drift_get_const_length)( drift )
        };
        
        success = NS(Block_write_to_binary_file)( fp, NS(BLOCK_TYPE_PARTICLE), 
            num_elements, num_attributes, attr, attr_sizes, attr_counts );
    }
    
    return success;
}


int NS(Drift_read_from_bin_file)( FILE* fp, NS(Drift)* SIXTRL_RESTRICT drift )
{
    int success = -1;
    
    SIXTRL_STATIC NS(block_size_t) const ONE      = ( NS(block_size_t) )1u;
    SIXTRL_STATIC NS(block_size_t) const ZERO     = ( NS(block_size_t) )0u;
    SIXTRL_STATIC SIXTRL_UINT64_T  const U64_ZERO = ( SIXTRL_UINT64_T  )0u;
    
    NS(block_size_t) const NUM_ATTRIBUTES = ( NS(block_size_t) )2u;
    NS(block_size_t) const REAL_SIZE = sizeof( SIXTRL_REAL_T );
    NS(block_size_t) const I64_SIZE  = sizeof( SIXTRL_INT64_T );
    
    NS(block_size_t) attr_sizes[]  = { ZERO, ZERO };
    NS(block_size_t) attr_counts[] = {  ONE,  ONE };
    
    void* attr[] = { 0, 0 };
    
    NS(BlockType) type_id = NS(BLOCK_TYPE_INVALID);
    NS(block_num_elements_t) num_particles = 0;
    NS(block_size_t) num_attrs = ZERO;
    SIXTRL_UINT64_T binary_length = U64_ZERO;
    SIXTRL_INT64_T success_flag  = 0;
    
    if( ( 0 == NS(Block_peak_at_next_block_in_binary_file)( fp, &binary_length, 
            &success_flag, &type_id, &num_particles, &num_attrs, 
                &attr_sizes[ 0 ], &attr_counts[ 0 ], NUM_ATTRIBUTES ) ) &&
        ( num_attrs == NUM_ATTRIBUTES ) &&
        ( type_id == NS(BLOCK_TYPE_PARTICLE) ) && ( success_flag == 0 ) && 
        ( binary_length > ZERO ) && ( num_particles > ZERO ) &&
        ( num_particles == ( NS(block_size_t) )1u ) )
    {
        NS(block_num_elements_t) num_elements = 0;
        
        attr[  0 ] = ( void* )NS(Drift_get_element_id)( drift );
        attr[  1 ] = ( void* )NS(Drift_get_length)( drift );
        
        success = NS(Block_read_structure_from_binary_file)( 
            fp, &binary_length, &success_flag, &type_id, &num_elements,
            &num_attrs, attr, &attr_sizes[ 0 ], &attr_counts[ 0 ] );
        
        SIXTRL_ASSERT( 
            ( success == 0 ) ||
            ( ( success_flag == 0    ) && 
              ( binary_length > ZERO ) && 
              ( num_elements == ( NS(block_num_elements_t) )1u ) &&
              ( attr_counts[  0 ] == ONE ) && 
              ( attr_counts[  1 ] == ONE ) && 
              ( attr_sizes[   0 ] == I64_SIZE  ) && 
              ( attr_sizes[   1 ] == REAL_SIZE ) ) );
    }
    
    return success;
}

/* end: sixtracklib/common/details/be_drift.c */
