#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

/* ------------------------------------------------------------------------- */

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

/* ------------------------------------------------------------------------- */

TEST( CommonTestsBeamElements, 
      CreateAndRandomInitBeamBeamsUnserializeCompare )
{
    st_Blocks beam_elements;
    st_Blocks_preset( &beam_elements );
    
    SIXTRL_STATIC st_block_num_elements_t const MIN_NUM_SLICES = 2u;
    SIXTRL_STATIC st_block_num_elements_t const MAX_NUM_SLICES = 64u;
    SIXTRL_INT64_T const NUM_SLICES_RANGE = MAX_NUM_SLICES - MIN_NUM_SLICES;
    
    SIXTRL_STATIC st_block_size_t const NUM_BLOCKS = 1000u;
    
    SIXTRL_STATIC st_block_size_t const BEAM_ELEMENTS_DATA_CAPACITY =
        st_Blocks_predict_data_capacity_for_num_blocks( 
            &beam_elements, NUM_BLOCKS ) +
        st_BeamBeam_predict_blocks_data_capacity( 
            &beam_elements, NUM_BLOCKS, MAX_NUM_SLICES );
        
    int ret = st_Blocks_init( 
        &beam_elements, NUM_BLOCKS, BEAM_ELEMENTS_DATA_CAPACITY );
    
    ASSERT_TRUE( ret == 0 );
    
    SIXTRL_REAL_T const REAL_CMP_EPS = 
        std::numeric_limits< SIXTRL_REAL_T >::epsilon();
        
    /* --------------------------------------------------------------------- */
        
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
    
    st_BeamBeamBoostData boost_data;
    st_BeamBeamBoostData_preset( &boost_data );
    
    st_BeamBeamBoostData_set_sphi(   &boost_data, 1.0 );
    st_BeamBeamBoostData_set_cphi(   &boost_data, 2.0 );
    st_BeamBeamBoostData_set_tphi(   &boost_data, 3.0 );
    st_BeamBeamBoostData_set_salpha( &boost_data, 4.0 );
    st_BeamBeamBoostData_set_calpha( &boost_data, 5.0 );
    
    
    st_BeamBeamSigmas sigmas;
    st_BeamBeamSigmas_preset( &sigmas );
    
    st_BeamBeamSigmas_set_sigma11( &sigmas,  1.0 );
    st_BeamBeamSigmas_set_sigma12( &sigmas,  2.0 );
    st_BeamBeamSigmas_set_sigma13( &sigmas,  3.0 );
    st_BeamBeamSigmas_set_sigma14( &sigmas,  4.0 );
    
    st_BeamBeamSigmas_set_sigma22( &sigmas,  5.0 );
    st_BeamBeamSigmas_set_sigma23( &sigmas,  6.0 );
    st_BeamBeamSigmas_set_sigma24( &sigmas,  7.0 );
    
    st_BeamBeamSigmas_set_sigma33( &sigmas,  8.0 );
    st_BeamBeamSigmas_set_sigma34( &sigmas,  9.0 );
    
    st_BeamBeamSigmas_set_sigma44( &sigmas, 10.0 );
    
    SIXTRL_REAL_T const MIN_X = 0.01;
    SIXTRL_REAL_T const MAX_X = 1.00;
    SIXTRL_REAL_T const DELTA_X = MAX_X - MIN_X;
    
    SIXTRL_REAL_T const MIN_Y = 0.01;
    SIXTRL_REAL_T const MAX_Y = 1.00;
    SIXTRL_REAL_T const DELTA_Y = MAX_Y - MIN_Y;
    
    SIXTRL_REAL_T const MIN_SIGMA = 0.01;
    SIXTRL_REAL_T const MAX_SIGMA = 1.00;
    SIXTRL_REAL_T const DELTA_SIGMA = MAX_SIGMA - MIN_SIGMA;
        
    std::vector< SIXTRL_REAL_T > n_part_per_slice( MAX_NUM_SLICES, 0.0 );
    std::vector< SIXTRL_REAL_T > x_slices_star( MAX_NUM_SLICES, 0.0 );
    std::vector< SIXTRL_REAL_T > y_slices_star( MAX_NUM_SLICES, 0.0 );
    std::vector< SIXTRL_REAL_T > sigma_slices_star( MAX_NUM_SLICES, 0.0 );
    
    std::vector< st_BeamBeam > cmp_beam_beam( NUM_BLOCKS );
    std::vector< st_BeamBeam const* > beam_beam_ptrs( NUM_BLOCKS, nullptr );
    
    st_block_size_t ii = 0u;
    
    for( ; ii < NUM_BLOCKS ; ++ii )
    {
        st_block_num_elements_t const num_slices = 
            static_cast< st_block_num_elements_t >( MIN_NUM_SLICES + 
                NUM_SLICES_RANGE * st_Random_genrand64_real1() );
        
        SIXTRL_REAL_T const n_part  = ( 1.0 ) / static_cast< SIXTRL_REAL_T >( num_slices );
            
        SIXTRL_REAL_T const x_begin = MIN_X;
        SIXTRL_REAL_T const x_end   = x_begin + DELTA_X * st_Random_genrand64_real1();
        SIXTRL_REAL_T const dx      = ( x_end - x_begin ) / ( num_slices - 1.0 );
        
        SIXTRL_REAL_T const y_begin = MIN_Y;
        SIXTRL_REAL_T const y_end   = y_begin + DELTA_Y * st_Random_genrand64_real1();
        SIXTRL_REAL_T const dy      = ( y_end - y_begin ) / ( num_slices - 1.0 );
        
        SIXTRL_REAL_T const sigma_begin = MIN_SIGMA;
        SIXTRL_REAL_T const sigma_end   = sigma_begin + DELTA_SIGMA * st_Random_genrand64_real1();
        SIXTRL_REAL_T const dsigma      = ( sigma_end - sigma_begin ) / ( num_slices - 1.0 );
        
        n_part_per_slice.clear();
        n_part_per_slice.resize( num_slices, n_part );
        
        int n = 0;
        x_slices_star.clear();
        x_slices_star.resize( num_slices, 0.0 );
        std::generate( x_slices_star.begin(), x_slices_star.end(), 
                       [x_begin,dx,&n]() mutable 
                       { return x_begin + dx * n++; } );
        
        n = 0;
        y_slices_star.clear();
        y_slices_star.resize( num_slices, 0.0 );
        std::generate( y_slices_star.begin(), y_slices_star.end(), 
                       [y_begin,dy,&n]() mutable 
                       { return y_begin + dy * n++; } );
        
        n = 0;
        sigma_slices_star.clear();
        sigma_slices_star.resize( num_slices, 0.0 );
        std::generate( sigma_slices_star.begin(), sigma_slices_star.end(), 
                       [sigma_begin,dsigma,&n]() mutable 
                       { return sigma_begin + dsigma * n++; } );
        
        beam_beam_ptrs[ ii ] = st_Blocks_add_beam_beam( 
            &beam_elements, &boost_data, &sigmas, num_slices, 
            n_part_per_slice.data(), x_slices_star.data(),
            y_slices_star.data(), sigma_slices_star.data(),
            n_part, 0.0, 1.0 );
        
        ASSERT_TRUE( beam_beam_ptrs[ ii ] != nullptr );
        
        cmp_beam_beam[ ii ] = *beam_beam_ptrs[ ii ];
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
                     st_BLOCK_TYPE_BEAM_BEAM );
        
        ASSERT_TRUE( st_BlockInfo_get_const_ptr_begin( blocks_it ) != 
                     nullptr );  
        
        st_BeamBeam const* beam_beam = 
            st_Blocks_get_const_beam_beam( blocks_it );
            
        ASSERT_TRUE( beam_beam != nullptr );
        ASSERT_TRUE( std::distance( beam_beam, beam_beam_ptrs[ ii ] ) == 0 );
        
        ASSERT_TRUE( st_BeamBeam_get_num_of_slices( beam_beam ) ==
                     cmp_beam_beam[ ii ].num_of_slices );
        
        ASSERT_TRUE( REAL_CMP_EPS >= std::fabs( st_BeamBeam_get_q_part( 
            beam_beam ) - cmp_beam_beam[ ii ].q_part ) );
        
        ASSERT_TRUE( REAL_CMP_EPS >= std::fabs( st_BeamBeam_get_min_sigma_diff(
            beam_beam ) - cmp_beam_beam[ ii ].min_sigma_diff ) );
        
        ASSERT_TRUE( REAL_CMP_EPS >= std::fabs( 
            st_BeamBeam_get_treshold_singular( beam_beam ) -
                cmp_beam_beam[ ii ].treshold_sing ) );
        
        ASSERT_TRUE( 0 == std::memcmp( 
            cmp_beam_beam[ ii ].boost,
            st_BeamBeam_get_const_ptr_boost_data( beam_beam ), 
            sizeof( st_BeamBeamBoostData ) ) );
        
        ASSERT_TRUE( 0 == std::memcmp(
            cmp_beam_beam[ ii ].sigmas,
            st_BeamBeam_get_const_ptr_sigmas_matrix( beam_beam ),
            sizeof( st_BeamBeamSigmas ) ) );
        
        ASSERT_TRUE( 0 == std::memcmp(
            cmp_beam_beam[ ii ].n_part_per_slice, 
            st_BeamBeam_get_const_n_part_per_slice( beam_beam ),
            st_BeamBeam_get_num_of_slices( beam_beam ) * 
                sizeof( SIXTRL_REAL_T ) ) );
        
        ASSERT_TRUE( 0 == std::memcmp(
            cmp_beam_beam[ ii ].x_slices_star, 
            st_BeamBeam_get_const_x_slices_star( beam_beam ),
            st_BeamBeam_get_num_of_slices( beam_beam ) * 
                sizeof( SIXTRL_REAL_T ) ) );
        
        ASSERT_TRUE( 0 == std::memcmp(
            cmp_beam_beam[ ii ].y_slices_star, 
            st_BeamBeam_get_const_y_slices_star( beam_beam ),
            st_BeamBeam_get_num_of_slices( beam_beam ) * 
                sizeof( SIXTRL_REAL_T ) ) );
        
        ASSERT_TRUE( 0 == std::memcmp(
            cmp_beam_beam[ ii ].sigma_slices_star, 
            st_BeamBeam_get_const_sigma_slices_star( beam_beam ),
            st_BeamBeam_get_num_of_slices( beam_beam ) * 
                sizeof( SIXTRL_REAL_T ) ) );
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
                     st_BLOCK_TYPE_BEAM_BEAM );
        
        ASSERT_TRUE( st_BlockInfo_get_const_ptr_begin( blocks_it ) != 
                     nullptr );  
        
        st_BeamBeam const* beam_beam = st_Blocks_get_const_beam_beam( blocks_it );        
        ASSERT_TRUE( beam_beam != nullptr );
        
        ASSERT_TRUE( st_BeamBeam_get_num_of_slices( beam_beam ) ==
                     cmp_beam_beam[ ii ].num_of_slices );
        
        ASSERT_TRUE( REAL_CMP_EPS >= std::fabs( st_BeamBeam_get_q_part( 
            beam_beam ) - cmp_beam_beam[ ii ].q_part ) );
        
        ASSERT_TRUE( REAL_CMP_EPS >= std::fabs( st_BeamBeam_get_min_sigma_diff(
            beam_beam ) - cmp_beam_beam[ ii ].min_sigma_diff ) );
        
        ASSERT_TRUE( REAL_CMP_EPS >= std::fabs( 
            st_BeamBeam_get_treshold_singular( beam_beam ) -
                cmp_beam_beam[ ii ].treshold_sing ) );
        
        ASSERT_TRUE( 0 == std::memcmp( 
            cmp_beam_beam[ ii ].boost,
            st_BeamBeam_get_const_ptr_boost_data( beam_beam ), 
            sizeof( st_BeamBeamBoostData ) ) );
        
        ASSERT_TRUE( 0 == std::memcmp(
            cmp_beam_beam[ ii ].sigmas,
            st_BeamBeam_get_const_ptr_sigmas_matrix( beam_beam ),
            sizeof( st_BeamBeamSigmas ) ) );
        
        ASSERT_TRUE( 0 == std::memcmp(
            cmp_beam_beam[ ii ].n_part_per_slice, 
            st_BeamBeam_get_const_n_part_per_slice( beam_beam ),
            st_BeamBeam_get_num_of_slices( beam_beam ) * 
                sizeof( SIXTRL_REAL_T ) ) );
        
        ASSERT_TRUE( 0 == std::memcmp(
            cmp_beam_beam[ ii ].x_slices_star, 
            st_BeamBeam_get_const_x_slices_star( beam_beam ),
            st_BeamBeam_get_num_of_slices( beam_beam ) * 
                sizeof( SIXTRL_REAL_T ) ) );
        
        ASSERT_TRUE( 0 == std::memcmp(
            cmp_beam_beam[ ii ].y_slices_star, 
            st_BeamBeam_get_const_y_slices_star( beam_beam ),
            st_BeamBeam_get_num_of_slices( beam_beam ) * 
                sizeof( SIXTRL_REAL_T ) ) );
        
        ASSERT_TRUE( 0 == std::memcmp(
            cmp_beam_beam[ ii ].sigma_slices_star, 
            st_BeamBeam_get_const_sigma_slices_star( beam_beam ),
            st_BeamBeam_get_num_of_slices( beam_beam ) * 
                sizeof( SIXTRL_REAL_T ) ) );
    }
    
    st_Blocks_free( &beam_elements );
    st_Blocks_free( &ref_beam_elements );
    st_Blocks_free( &copy_beam_elements );    
}

/* end: sixtracklib/common/tests/test_beam_elements.cpp */
