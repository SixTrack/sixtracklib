#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <random>

#include <gtest/gtest.h>

#if defined( __NAMESPACE )
    #define __SAVED_NAMESPACE __NAMESPACE
    #undef  __NAMESPACE     
#endif /* !defiend( __NAMESPACE ) */

#if !defined( __NAMESPACE )
    #define __NAMESPACE st_    
#endif /* !defiend( __NAMESPACE ) */

#include "sixtracklib/common/impl/be_drift_impl.h"
#include "sixtracklib/common/be_drift.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/details/random.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

TEST( CommonTestsBeamElements, CreateDriftsOnMemoryRemapAndCompare )
{
    
    st_block_num_elements_t ii = 0;
    st_block_num_elements_t const NUM_BEAM_ELEMENTS = 1000u;
    st_block_size_t const BEAM_ELEMENTS_DATA_CAPACITY = NUM_BEAM_ELEMENTS * ( 
            sizeof( SIXTRL_REAL_T ) + sizeof( SIXTRL_INT64_T ) );
    
    std::vector< SIXTRL_REAL_T > cmp_drift_lengths( 
        NUM_BEAM_ELEMENTS, SIXTRL_REAL_T{ 0.0 } );
    
    std::vector< st_Drift > cmp_drifts( NUM_BEAM_ELEMENTS, st_Drift{} );
    
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
    
    static SIXTRL_REAL_T const DRIFT_CMP_EPS = SIXTRL_REAL_T{ 1e-16 };
    
    static SIXTRL_REAL_T const MIN_DRIFT_LEN = SIXTRL_REAL_T{ 0.01 };
    static SIXTRL_REAL_T const MAX_DRIFT_LEN = SIXTRL_REAL_T{ 0.10 };
    SIXTRL_REAL_T const DRIFT_LEN_RANGE = MAX_DRIFT_LEN - MIN_DRIFT_LEN;
    
    cmp_drift_lengths.clear();
    cmp_drifts.clear();
    
    st_BeamElements beam_elements;
    st_BeamElements_preset( &beam_elements );
    
    int ret = st_BeamElements_init( 
        &beam_elements, NUM_BEAM_ELEMENTS, BEAM_ELEMENTS_DATA_CAPACITY );
    
    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( st_BeamElements_get_num_of_blocks( &beam_elements ) == 0u );
    ASSERT_TRUE( st_BeamElements_has_info_store( &beam_elements ) );
    ASSERT_TRUE( st_BeamElements_has_data_store( &beam_elements ) );
    
    ASSERT_TRUE( st_BeamElements_get_const_block_infos_begin( 
        &beam_elements ) != nullptr );    
    
    ASSERT_TRUE( st_BeamElements_get_const_ptr_data_begin( 
        &beam_elements ) != nullptr );
    
    /* --------------------------------------------------------------------- */
    /* creating drifts on memory: */
    
    for( ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        SIXTRL_REAL_T const len = MIN_DRIFT_LEN +
            DRIFT_LEN_RANGE * st_Random_genrand64_real1();
        
        st_Drift drift;
        st_Drift_preset( &drift );
            
        st_BlockInfo* block_info_it = st_BeamElements_create_beam_element( 
            &drift, &beam_elements, st_BLOCK_TYPE_DRIFT );
        
        ASSERT_TRUE( block_info_it != nullptr );
        ASSERT_TRUE( st_BlockInfo_get_type_id( block_info_it ) == 
                     st_BLOCK_TYPE_DRIFT );
        
        ASSERT_TRUE( st_Drift_get_type_id( &drift ) == 
                     st_BLOCK_TYPE_DRIFT );
        
        st_Drift_set_element_id_value( &drift, ii );
        st_Drift_set_length_value( &drift, len );
        
        cmp_drift_lengths.push_back( len );
        cmp_drifts.push_back( drift );
    }
    
    /* --------------------------------------------------------------------- */
    /* Remap from memory. */
    
    st_BlockInfo const* block_info_it  = 
        st_BeamElements_get_const_block_infos_begin( &beam_elements );
    
    st_BlockInfo const* block_info_end  = 
        st_BeamElements_get_const_block_infos_end( &beam_elements );
    
    SIXTRL_GLOBAL_DEC unsigned char* mem_begin = 
        st_BeamElements_get_ptr_data_begin( &beam_elements );
        
    st_block_size_t const mem_max_bytes = 
        st_BeamElements_get_data_size( &beam_elements );
        
    ASSERT_TRUE( block_info_it  != nullptr );
    ASSERT_TRUE( block_info_end != nullptr );
    ASSERT_TRUE( std::distance( block_info_it, block_info_end ) == 
                 static_cast< std::ptrdiff_t >( NUM_BEAM_ELEMENTS ) );
    
    ASSERT_TRUE( mem_begin != nullptr );
    ASSERT_TRUE( mem_max_bytes >= BEAM_ELEMENTS_DATA_CAPACITY );
    
    ASSERT_TRUE( cmp_drifts.size() == NUM_BEAM_ELEMENTS );
    ASSERT_TRUE( cmp_drift_lengths.size() == NUM_BEAM_ELEMENTS );
    
    ii = 0;
        
    for( ; block_info_it != block_info_end ; ++block_info_it, ++ii )
    {
        /*  remap using block_info_it and compare -> */
        
        st_Drift const& cmp_drift = cmp_drifts[ ii ];
        
        st_Drift remapped_drift;
        st_Drift_preset( &remapped_drift );
        
        ret = st_Drift_remap_from_memory( 
            &remapped_drift, block_info_it, mem_begin, mem_max_bytes );
        
        ASSERT_TRUE( ret == 0 );
        ASSERT_TRUE( st_Drift_get_type_id( &remapped_drift ) == 
                     st_BLOCK_TYPE_DRIFT );
        
        ASSERT_TRUE( st_Drift_get_element_id_value( &remapped_drift ) == 
                     st_Drift_get_element_id_value( &cmp_drift ) );
        
        ASSERT_TRUE( st_Drift_get_const_element_id( &remapped_drift ) ==
                     st_Drift_get_const_element_id( &cmp_drift ) );
        
        ASSERT_TRUE( std::fabs( st_Drift_get_length_value( &remapped_drift ) -
                     cmp_drift_lengths[ ii ] ) < DRIFT_CMP_EPS );
        
        ASSERT_TRUE( st_Drift_get_const_length( &remapped_drift ) ==
                     st_Drift_get_const_length( &cmp_drift ) );
        
        /*  remap using st_BeamElements convenience function and compare -> */
        
        st_Drift other_remapped_drift;
        st_Drift_preset( &other_remapped_drift );
        
        ret = st_BeamElements_get_beam_element( 
            &other_remapped_drift, &beam_elements, ii );
        
        ASSERT_TRUE( ret == 0 );
        
        ASSERT_TRUE( st_Drift_get_type_id( &other_remapped_drift) == 
                     st_BLOCK_TYPE_DRIFT );
        
        ASSERT_TRUE( st_Drift_get_const_element_id( &other_remapped_drift ) ==
                     st_Drift_get_const_element_id( &cmp_drift ) );
        
        ASSERT_TRUE( st_Drift_get_const_length( &other_remapped_drift ) ==
                     st_Drift_get_const_length( &cmp_drift ) );
    }
    
    /* Free BeamElements buffer: */
    
    st_BeamElements_free( &beam_elements );
}

/* end: sixtracklib/common/tests/test_beam_elements.cpp */
