#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#if defined( __NAMESPACE )
    #define __SAVED_NAMESPACE __NAMESPACE
    #undef  __NAMESPACE     
#endif /* !defiend( __NAMESPACE ) */

#if !defined( __NAMESPACE )
    #define __NAMESPACE st_    
#endif /* !defiend( __NAMESPACE ) */

#include "sixtracklib/common/details/random.h"
#include "sixtracklib/common/beam_elements.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */


TEST( CommonTestsBeamElements, CreateAndRandomInitDriftsUnserializeCompare )
{
    st_Blocks beam_elements;
    st_Blocks_preset( &beam_elements );
    
    SIXTRL_STATIC st_block_size_t const NUM_BLOCKS = 1000u;
    
    SIXTRL_STATIC st_block_size_t const BEAM_ELEMENTS_DATA_CAPACITY =
        st_Blocks_predict_data_capacity_for_num_blocks( 
            &beam_elements, NUM_BLOCKS ) +
        st_Drift_predict_blocks_data_capacity( 
            &beam_elements, NUM_BLOCKS );
        
    SIXTRL_STATIC SIXTRL_REAL_T const MIN_DRIFT_LEN = ( SIXTRL_REAL_T )0.05L;
    SIXTRL_STATIC SIXTRL_REAL_T const MAX_DRIFT_LEN = ( SIXTRL_REAL_T )1.00L;
    
    int ret = st_Blocks_init( 
        &beam_elements, NUM_BLOCKS, BEAM_ELEMENTS_DATA_CAPACITY );
    
    ASSERT_TRUE( ret == 0 );
    
    SIXTRL_REAL_T const DRIFT_LEN_RANGE = 
        MAX_DRIFT_LEN - MIN_DRIFT_LEN;
    
    SIXTRL_REAL_T const DRIFT_LEN_CMP_EPS = 
        std::numeric_limits< SIXTRL_REAL_T >::epsilon();
        
    /* --------------------------------------------------------------------- */
        
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
        
    std::vector< SIXTRL_REAL_T > cmp_drift_lengths( NUM_BLOCKS, 0.0 );
    std::vector< SIXTRL_GLOBAL_DEC st_Drift const* > 
        ptr_drifts( NUM_BLOCKS, nullptr );
    
    st_block_size_t ii = 0u;
    
    for( ; ii < NUM_BLOCKS ; ++ii )
    {
        cmp_drift_lengths[ ii ] = MIN_DRIFT_LEN + 
            DRIFT_LEN_RANGE * st_Random_genrand64_real1();
        
        ptr_drifts[ ii ] = st_Blocks_add_drift( 
            &beam_elements, cmp_drift_lengths[ ii ] );
    }
    
    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &beam_elements ) == NUM_BLOCKS );
    ASSERT_TRUE( !st_Blocks_are_serialized( &beam_elements ) );
    
    ASSERT_TRUE( st_Blocks_serialize( &beam_elements ) == 0 );
    ASSERT_TRUE( st_Blocks_are_serialized( &beam_elements ) );
    
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &beam_elements ) == NUM_BLOCKS );
    
    SIXTRL_GLOBAL_DEC unsigned char* data_mem_begin =
        st_Blocks_get_data_begin( &beam_elements );
        
    ASSERT_TRUE( data_mem_begin != nullptr );
    
    st_Blocks ref_beam_elements;
    st_Blocks_preset( &ref_beam_elements );
    
    ret = st_Blocks_unserialize( &ref_beam_elements, data_mem_begin );
    
    ASSERT_TRUE( ret == 0 );    
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( 
        &ref_beam_elements ) == NUM_BLOCKS );
    
    SIXTRL_GLOBAL_DEC st_BlockInfo const* blocks_it  = 
        st_Blocks_get_const_block_infos_begin( &ref_beam_elements );
        
    SIXTRL_GLOBAL_DEC st_BlockInfo const* blocks_end =
        st_Blocks_get_const_block_infos_end( &ref_beam_elements );
        
    ASSERT_TRUE( std::distance( blocks_it, blocks_end ) == 
        static_cast< std::ptrdiff_t >( NUM_BLOCKS ) );
    
    /* --------------------------------------------------------------------- */
    
    for( ii = 0 ; blocks_it != blocks_end ; ++blocks_it, ++ii )
    {
        ASSERT_TRUE( st_BlockInfo_get_type_id( blocks_it ) == 
                     st_BLOCK_TYPE_DRIFT );
        
        ASSERT_TRUE( st_BlockInfo_get_const_ptr_begin( blocks_it ) != 
                     nullptr );  
        
        st_Drift const* drift = st_Blocks_get_const_drift( blocks_it );
        ASSERT_TRUE( drift != nullptr );
        ASSERT_TRUE( std::distance( drift, ptr_drifts[ ii ] ) == 0 );
        ASSERT_TRUE( DRIFT_LEN_CMP_EPS >= std::fabs( 
            st_Drift_get_length( drift ) - cmp_drift_lengths[ ii ] ) );
    }
    
    /* --------------------------------------------------------------------- */
    
    std::vector< unsigned char > copy_data_buffer(
        st_Blocks_get_const_data_begin( &beam_elements ),
        st_Blocks_get_const_data_end( &beam_elements ) );
    
    ASSERT_TRUE( copy_data_buffer.size() == 
                 st_Blocks_get_total_num_bytes( &beam_elements ) );
    
    st_Blocks copy_beam_elements;
    st_Blocks_preset( &copy_beam_elements );
    
    ret = st_Blocks_unserialize( 
        &copy_beam_elements, copy_data_buffer.data() );
    
    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &copy_beam_elements ) == 
                 NUM_BLOCKS );
    
    blocks_it  = st_Blocks_get_const_block_infos_begin( &ref_beam_elements );        
    blocks_end = st_Blocks_get_const_block_infos_end( &ref_beam_elements );
        
    ASSERT_TRUE( blocks_it != 
        st_Blocks_get_const_block_infos_begin( &copy_beam_elements ) );
    
    ASSERT_TRUE( blocks_end != 
        st_Blocks_get_const_block_infos_end( &copy_beam_elements ) );
    
    ASSERT_TRUE( std::distance( blocks_it, blocks_end ) == 
        static_cast< std::ptrdiff_t >( NUM_BLOCKS ) );
    
    /* --------------------------------------------------------------------- */
    
    for( ii = 0 ; blocks_it != blocks_end ; ++blocks_it, ++ii )
    {
        ASSERT_TRUE( st_BlockInfo_get_type_id( blocks_it ) == 
                     st_BLOCK_TYPE_DRIFT );
        
        ASSERT_TRUE( st_BlockInfo_get_const_ptr_begin( blocks_it ) != 
                     nullptr );  
        
        st_Drift const* drift = st_Blocks_get_const_drift( blocks_it );
        ASSERT_TRUE( drift != nullptr );
        ASSERT_TRUE( std::distance( drift, ptr_drifts[ ii ] ) == 0 );
        ASSERT_TRUE( DRIFT_LEN_CMP_EPS >= std::fabs( 
            st_Drift_get_length( drift ) - cmp_drift_lengths[ ii ] ) );
    }
    
    st_Blocks_free( &beam_elements );
    st_Blocks_free( &ref_beam_elements );
    st_Blocks_free( &copy_beam_elements );    
}

/* -------------------------------------------------------------------------*/

TEST( CommonTestsBeamElements, 
      CreateAndRandomInitDriftExactssUnserializeCompare )
{
    st_Blocks beam_elements;
    st_Blocks_preset( &beam_elements );
    
    SIXTRL_STATIC st_block_size_t const NUM_BLOCKS = 1000u;
    
    SIXTRL_STATIC st_block_size_t const BEAM_ELEMENTS_DATA_CAPACITY =
        st_Blocks_predict_data_capacity_for_num_blocks( 
            &beam_elements, NUM_BLOCKS ) +
        st_DriftExact_predict_blocks_data_capacity( 
            &beam_elements, NUM_BLOCKS );
        
    SIXTRL_STATIC SIXTRL_REAL_T const MIN_DRIFT_LEN = ( SIXTRL_REAL_T )0.05L;
    SIXTRL_STATIC SIXTRL_REAL_T const MAX_DRIFT_LEN = ( SIXTRL_REAL_T )1.00L;
    
    int ret = st_Blocks_init( 
        &beam_elements, NUM_BLOCKS, BEAM_ELEMENTS_DATA_CAPACITY );
    
    ASSERT_TRUE( ret == 0 );
    
    SIXTRL_REAL_T const DRIFT_LEN_RANGE = 
        MAX_DRIFT_LEN - MIN_DRIFT_LEN;
    
    SIXTRL_REAL_T const DRIFT_LEN_CMP_EPS = 
        std::numeric_limits< SIXTRL_REAL_T >::epsilon();
        
    /* --------------------------------------------------------------------- */
        
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
        
    std::vector< SIXTRL_REAL_T > cmp_drift_lengths( NUM_BLOCKS, 0.0 );
    std::vector< SIXTRL_GLOBAL_DEC st_DriftExact const* > 
        ptr_drifts( NUM_BLOCKS, nullptr );
    
    st_block_size_t ii = 0u;
    
    for( ; ii < NUM_BLOCKS ; ++ii )
    {
        cmp_drift_lengths[ ii ] = MIN_DRIFT_LEN + 
            DRIFT_LEN_RANGE * st_Random_genrand64_real1();
        
        ptr_drifts[ ii ] = st_Blocks_add_drift_exact( 
            &beam_elements, cmp_drift_lengths[ ii ] );
    }
    
    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &beam_elements ) == NUM_BLOCKS );
    ASSERT_TRUE( !st_Blocks_are_serialized( &beam_elements ) );
    
    ASSERT_TRUE( st_Blocks_serialize( &beam_elements ) == 0 );
    ASSERT_TRUE( st_Blocks_are_serialized( &beam_elements ) );
    
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &beam_elements ) == NUM_BLOCKS );
    
    SIXTRL_GLOBAL_DEC unsigned char* data_mem_begin =
        st_Blocks_get_data_begin( &beam_elements );
        
    ASSERT_TRUE( data_mem_begin != nullptr );
    
    st_Blocks ref_beam_elements;
    st_Blocks_preset( &ref_beam_elements );
    
    ret = st_Blocks_unserialize( &ref_beam_elements, data_mem_begin );
    
    ASSERT_TRUE( ret == 0 );    
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( 
        &ref_beam_elements ) == NUM_BLOCKS );
    
    SIXTRL_GLOBAL_DEC st_BlockInfo const* blocks_it  = 
        st_Blocks_get_const_block_infos_begin( &ref_beam_elements );
        
    SIXTRL_GLOBAL_DEC st_BlockInfo const* blocks_end =
        st_Blocks_get_const_block_infos_end( &ref_beam_elements );
        
    ASSERT_TRUE( std::distance( blocks_it, blocks_end ) == 
        static_cast< std::ptrdiff_t >( NUM_BLOCKS ) );
    
    /* --------------------------------------------------------------------- */
    
    for( ii = 0 ; blocks_it != blocks_end ; ++blocks_it, ++ii )
    {
        ASSERT_TRUE( st_BlockInfo_get_type_id( blocks_it ) == 
                     st_BLOCK_TYPE_DRIFT_EXACT );
        
        ASSERT_TRUE( st_BlockInfo_get_const_ptr_begin( blocks_it ) != 
                     nullptr );  
        
        st_DriftExact const* drift = st_Blocks_get_const_drift_exact( blocks_it );
        ASSERT_TRUE( drift != nullptr );
        ASSERT_TRUE( std::distance( drift, ptr_drifts[ ii ] ) == 0 );
        ASSERT_TRUE( DRIFT_LEN_CMP_EPS >= std::fabs( 
            st_DriftExact_get_length( drift ) - cmp_drift_lengths[ ii ] ) );
    }
    
    /* --------------------------------------------------------------------- */
    
    std::vector< unsigned char > copy_data_buffer(
        st_Blocks_get_const_data_begin( &beam_elements ),
        st_Blocks_get_const_data_end( &beam_elements ) );
    
    ASSERT_TRUE( copy_data_buffer.size() == 
                 st_Blocks_get_total_num_bytes( &beam_elements ) );
    
    st_Blocks copy_beam_elements;
    st_Blocks_preset( &copy_beam_elements );
    
    ret = st_Blocks_unserialize( 
        &copy_beam_elements, copy_data_buffer.data() );
    
    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &copy_beam_elements ) == 
                 NUM_BLOCKS );
    
    blocks_it  = st_Blocks_get_const_block_infos_begin( &ref_beam_elements );        
    blocks_end = st_Blocks_get_const_block_infos_end( &ref_beam_elements );
        
    ASSERT_TRUE( blocks_it != 
        st_Blocks_get_const_block_infos_begin( &copy_beam_elements ) );
    
    ASSERT_TRUE( blocks_end != 
        st_Blocks_get_const_block_infos_end( &copy_beam_elements ) );
    
    ASSERT_TRUE( std::distance( blocks_it, blocks_end ) == 
        static_cast< std::ptrdiff_t >( NUM_BLOCKS ) );
    
    /* --------------------------------------------------------------------- */
    
    for( ii = 0 ; blocks_it != blocks_end ; ++blocks_it, ++ii )
    {
        ASSERT_TRUE( st_BlockInfo_get_type_id( blocks_it ) == 
                     st_BLOCK_TYPE_DRIFT_EXACT );
        
        ASSERT_TRUE( st_BlockInfo_get_const_ptr_begin( blocks_it ) != 
                     nullptr );  
        
        st_DriftExact const* drift = st_Blocks_get_const_drift_exact( blocks_it );
        ASSERT_TRUE( drift != nullptr );
        ASSERT_TRUE( std::distance( drift, ptr_drifts[ ii ] ) == 0 );
        ASSERT_TRUE( DRIFT_LEN_CMP_EPS >= std::fabs( 
            st_DriftExact_get_length( drift ) - cmp_drift_lengths[ ii ] ) );
    }
    
    st_Blocks_free( &beam_elements );
    st_Blocks_free( &ref_beam_elements );
    st_Blocks_free( &copy_beam_elements );    
}

TEST( CommonTestsBeamElements, 
      CreateAndRandomInitMultiPolesUnserializeCompare )
{
    st_Blocks beam_elements;
    st_Blocks_preset( &beam_elements );
    
    SIXTRL_STATIC SIXTRL_INT64_T const MIN_ORDER   = 1u;
    SIXTRL_STATIC SIXTRL_INT64_T const MAX_ORDER   = 7u;
    SIXTRL_INT64_T const ORDER_RANGE = MAX_ORDER - MIN_ORDER;
    
    SIXTRL_STATIC st_block_size_t const NUM_BLOCKS = 1000u;
    
    SIXTRL_STATIC st_block_size_t const BEAM_ELEMENTS_DATA_CAPACITY =
        st_Blocks_predict_data_capacity_for_num_blocks( 
            &beam_elements, NUM_BLOCKS ) +
        st_MultiPole_predict_blocks_data_capacity( 
            &beam_elements, NUM_BLOCKS, MAX_ORDER );
        
    SIXTRL_STATIC SIXTRL_REAL_T const MIN_DRIFT_LEN = ( SIXTRL_REAL_T )0.05L;
    SIXTRL_STATIC SIXTRL_REAL_T const MAX_DRIFT_LEN = ( SIXTRL_REAL_T )1.00L;
    
    SIXTRL_REAL_T const DRIFT_LEN_RANGE = 
        MAX_DRIFT_LEN - MIN_DRIFT_LEN;
    
    SIXTRL_STATIC SIXTRL_REAL_T const MIN_HXYL = ( SIXTRL_REAL_T )0.05L;
    SIXTRL_STATIC SIXTRL_REAL_T const MAX_HXYL = ( SIXTRL_REAL_T )1.00L;
    SIXTRL_REAL_T const DELTA_HXYL = MAX_HXYL - MIN_HXYL;
        
    SIXTRL_STATIC SIXTRL_REAL_T const MIN_BAL_VALUE = ( SIXTRL_REAL_T )-1.0;
    SIXTRL_STATIC SIXTRL_REAL_T const MAX_BAL_VALUE = ( SIXTRL_REAL_T )+1.0;
    SIXTRL_REAL_T const DELTA_BAL_VALUE_RANGE = MAX_BAL_VALUE - MIN_BAL_VALUE;
    
    int ret = st_Blocks_init( 
        &beam_elements, NUM_BLOCKS, BEAM_ELEMENTS_DATA_CAPACITY );
    
    ASSERT_TRUE( ret == 0 );
    
    SIXTRL_REAL_T const REAL_CMP_EPS = 
        std::numeric_limits< SIXTRL_REAL_T >::epsilon();
        
    /* --------------------------------------------------------------------- */
        
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
        
    std::vector< SIXTRL_REAL_T > bal_values( 255, 0.0 );    
    std::vector< st_MultiPole > cmp_multi_poles( NUM_BLOCKS );
    std::vector< SIXTRL_GLOBAL_DEC st_MultiPole const* > 
        ptr_multi_poles( NUM_BLOCKS, nullptr );
    
    st_block_size_t ii = 0u;
    
    for( ; ii < NUM_BLOCKS ; ++ii )
    {
        SIXTRL_REAL_T const length = 
            MIN_DRIFT_LEN + DRIFT_LEN_RANGE * st_Random_genrand64_real1();
            
        SIXTRL_REAL_T const hxl = 
            MIN_HXYL + DELTA_HXYL * st_Random_genrand64_real1();
            
        SIXTRL_REAL_T const hyl = 
            MIN_HXYL + DELTA_HXYL * st_Random_genrand64_real1();
            
        SIXTRL_INT64_T const order = MIN_ORDER + static_cast< SIXTRL_INT64_T >( 
                ORDER_RANGE * st_Random_genrand64_real1() );
        
        SIXTRL_INT64_T const nn = 2 * order + 2;
        
        bal_values.clear();
        bal_values.reserve( nn );
        
        for( SIXTRL_INT64_T jj = 0 ; jj < nn ; ++jj )
        {
            bal_values.push_back( MIN_BAL_VALUE + 
                st_Random_genrand64_real1() * DELTA_BAL_VALUE_RANGE );
        }
        
        ptr_multi_poles[ ii ] = st_Blocks_add_multipole( 
            &beam_elements, length, hxl, hyl, order, bal_values.data() );
        
        ASSERT_TRUE( ptr_multi_poles[ ii ] != nullptr );
        
        cmp_multi_poles[ ii ] = *ptr_multi_poles[ ii ];
    }
    
    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &beam_elements ) == NUM_BLOCKS );
    ASSERT_TRUE( !st_Blocks_are_serialized( &beam_elements ) );
    
    ASSERT_TRUE( st_Blocks_serialize( &beam_elements ) == 0 );
    ASSERT_TRUE( st_Blocks_are_serialized( &beam_elements ) );
    
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &beam_elements ) == NUM_BLOCKS );
    
    SIXTRL_GLOBAL_DEC unsigned char* data_mem_begin =
        st_Blocks_get_data_begin( &beam_elements );
        
    ASSERT_TRUE( data_mem_begin != nullptr );
    
    st_Blocks ref_beam_elements;
    st_Blocks_preset( &ref_beam_elements );
    
    ret = st_Blocks_unserialize( &ref_beam_elements, data_mem_begin );
    
    ASSERT_TRUE( ret == 0 );    
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( 
        &ref_beam_elements ) == NUM_BLOCKS );
    
    SIXTRL_GLOBAL_DEC st_BlockInfo const* blocks_it  = 
        st_Blocks_get_const_block_infos_begin( &ref_beam_elements );
        
    SIXTRL_GLOBAL_DEC st_BlockInfo const* blocks_end =
        st_Blocks_get_const_block_infos_end( &ref_beam_elements );
        
    ASSERT_TRUE( std::distance( blocks_it, blocks_end ) == 
        static_cast< std::ptrdiff_t >( NUM_BLOCKS ) );
    
    /* --------------------------------------------------------------------- */
    
    for( ii = 0 ; blocks_it != blocks_end ; ++blocks_it, ++ii )
    {
        ASSERT_TRUE( st_BlockInfo_get_type_id( blocks_it ) == 
                     st_BLOCK_TYPE_MULTIPOLE );
        
        ASSERT_TRUE( st_BlockInfo_get_const_ptr_begin( blocks_it ) != 
                     nullptr );  
        
        st_MultiPole const* multipole = 
            st_Blocks_get_const_multipole( blocks_it );
            
        ASSERT_TRUE( multipole != nullptr );
        ASSERT_TRUE( std::distance( multipole, ptr_multi_poles[ ii ] ) == 0 );
        
        ASSERT_TRUE( REAL_CMP_EPS >= std::fabs( 
            st_MultiPole_get_length( multipole ) - 
            cmp_multi_poles[ ii ].length ) );
        
        ASSERT_TRUE( REAL_CMP_EPS >= std::fabs( 
            st_MultiPole_get_hxl( multipole ) - 
            cmp_multi_poles[ ii ].hxl ) );
        
        ASSERT_TRUE( REAL_CMP_EPS >= std::fabs( 
            st_MultiPole_get_hyl( multipole ) - 
            cmp_multi_poles[ ii ].hyl ) );
        
        ASSERT_TRUE( st_MultiPole_get_order( multipole ) ==
            cmp_multi_poles[ ii ].order );
        
        ASSERT_TRUE( 0 == memcmp( 
            st_MultiPole_get_const_bal( multipole ), 
            cmp_multi_poles[ ii ].bal, ( sizeof( SIXTRL_REAL_T ) * ( 
                cmp_multi_poles[ ii ].order * 2u + 2u ) ) ) );
    }
    
    /* --------------------------------------------------------------------- */
    
    std::vector< unsigned char > copy_data_buffer(
        st_Blocks_get_const_data_begin( &beam_elements ),
        st_Blocks_get_const_data_end( &beam_elements ) );
    
    ASSERT_TRUE( copy_data_buffer.size() == 
                 st_Blocks_get_total_num_bytes( &beam_elements ) );
    
    st_Blocks copy_beam_elements;
    st_Blocks_preset( &copy_beam_elements );
    
    ret = st_Blocks_unserialize( 
        &copy_beam_elements, copy_data_buffer.data() );
    
    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &copy_beam_elements ) == 
                 NUM_BLOCKS );
    
    blocks_it  = st_Blocks_get_const_block_infos_begin( &ref_beam_elements );        
    blocks_end = st_Blocks_get_const_block_infos_end( &ref_beam_elements );
        
    ASSERT_TRUE( blocks_it != 
        st_Blocks_get_const_block_infos_begin( &copy_beam_elements ) );
    
    ASSERT_TRUE( blocks_end != 
        st_Blocks_get_const_block_infos_end( &copy_beam_elements ) );
    
    ASSERT_TRUE( std::distance( blocks_it, blocks_end ) == 
        static_cast< std::ptrdiff_t >( NUM_BLOCKS ) );
    
    /* --------------------------------------------------------------------- */
    
    for( ii = 0 ; blocks_it != blocks_end ; ++blocks_it, ++ii )
    {
        ASSERT_TRUE( st_BlockInfo_get_type_id( blocks_it ) == 
                     st_BLOCK_TYPE_MULTIPOLE );
        
        ASSERT_TRUE( st_BlockInfo_get_const_ptr_begin( blocks_it ) != 
                     nullptr );  
        
        st_MultiPole const* multipole = 
            st_Blocks_get_const_multipole( blocks_it );
        
        ASSERT_TRUE( multipole != nullptr );
        
        ASSERT_TRUE( REAL_CMP_EPS >= std::fabs( 
            st_MultiPole_get_length( multipole ) - 
            cmp_multi_poles[ ii ].length ) );
        
        ASSERT_TRUE( REAL_CMP_EPS >= std::fabs( 
            st_MultiPole_get_hxl( multipole ) - 
            cmp_multi_poles[ ii ].hxl ) );
        
        ASSERT_TRUE( REAL_CMP_EPS >= std::fabs( 
            st_MultiPole_get_hyl( multipole ) - 
            cmp_multi_poles[ ii ].hyl ) );
        
        ASSERT_TRUE( st_MultiPole_get_order( multipole ) ==
            cmp_multi_poles[ ii ].order );
        
        ASSERT_TRUE( 0 == memcmp( 
            st_MultiPole_get_const_bal( multipole ), 
            cmp_multi_poles[ ii ].bal, sizeof( SIXTRL_REAL_T ) * ( 
                cmp_multi_poles[ ii ].order * 2u + 2u ) ) );
    }
    
    st_Blocks_free( &beam_elements );
    st_Blocks_free( &ref_beam_elements );
    st_Blocks_free( &copy_beam_elements );    
}

/* end: sixtracklib/common/tests/test_beam_elements.cpp */
