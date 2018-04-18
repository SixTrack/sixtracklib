#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#if defined( __NAMESPACE )
    #define __SAVED_NAMESPACE __NAMESPACE
    #undef  __NAMESPACE     
#endif /* !defiend( __NAMESPACE ) */

#if !defined( __NAMESPACE )
    #define __NAMESPACE st_    
#endif /* !defiend( __NAMESPACE ) */

#include "sixtracklib/common/block.h"
#include "sixtracklib/common/impl/block_type.h"
#include "sixtracklib/common/block_drift.h"
#include "sixtracklib/common/impl/block_drift_type.h"
#include "sixtracklib/common/mem_pool.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

TEST( CommonBlocksTests, CreateBlockSaveNumDrfitsUnpackAndThenFree )
{
    typedef st_BeamElementInfo elem_info_t;
    typedef st_BeamElementType elem_type_t;
    typedef st_Drift           drift_t;
    typedef unsigned char      uchar_t;
    
    static std::size_t const ZERO = std::size_t{ 0 };
    static std::size_t const ONE  = std::size_t{ 1 };    
    static std::size_t const U64_SIZE = sizeof( SIXTRL_UINT64_T );
    
    static std::size_t const CAPACITY = 100;    
    st_Block* blocks = st_Block_new( CAPACITY );
    
    ASSERT_TRUE( blocks != 0 );
    ASSERT_TRUE( st_Block_get_capacity( blocks ) >= CAPACITY );
    ASSERT_TRUE( st_Block_get_size( blocks ) == std::size_t{ 0 } );
    
    ASSERT_TRUE( st_Block_get_const_elements_begin( blocks ) != nullptr );
    ASSERT_TRUE( st_Block_get_const_elements_begin( blocks ) ==
                 st_Block_get_const_elements_end( blocks ) );
    
    ASSERT_TRUE( st_Block_manages_own_memory( blocks ) );
    ASSERT_TRUE( st_Block_uses_mempool( blocks ) );
    
    SIXTRL_INT64_T prev_element_id = SIXTRL_INT64_T{ -1 };
    SIXTRL_SIZE_T prev_size = st_Block_get_size( blocks );
    
    std::vector< SIXTRL_REAL_T > lengths( CAPACITY, SIXTRL_REAL_T{ 0 } );
    lengths.clear();
    
    std::vector< SIXTRL_INT64_T > element_ids( 
        CAPACITY, st_PARTICLES_INVALID_BEAM_ELEMENT_ID );
    
    element_ids.clear();
    
    for( std::size_t ii = 0 ; ii < CAPACITY ; ++ii )
    {
        SIXTRL_REAL_T const length = static_cast< SIXTRL_REAL_T >( ii ) + 0.5;
        SIXTRL_INT64_T const new_element_id = st_Block_append_drift( 
            blocks, st_ELEMENT_TYPE_DRIFT, length );
        
        SIXTRL_SIZE_T const current_size = st_Block_get_size( blocks );
        
        ASSERT_TRUE( prev_element_id != new_element_id );
        ASSERT_TRUE( current_size <= CAPACITY );
        ASSERT_TRUE( prev_size + ONE == current_size ); 
        
        prev_size = current_size;
        prev_element_id = new_element_id;
        
        element_ids.push_back( new_element_id );
        lengths.push_back( length );
    }
    
    ASSERT_TRUE( element_ids.size() == CAPACITY );
    ASSERT_TRUE( lengths.size() == CAPACITY );    
    ASSERT_TRUE( st_Block_get_size( blocks ) == CAPACITY );
    
    elem_info_t* elem_it  = st_Block_get_elements_begin( blocks );
    elem_info_t* elem_end = st_Block_get_elements_end( blocks );
    
    ASSERT_TRUE( ( elem_it  != nullptr ) && ( elem_end != nullptr ) );
    ASSERT_TRUE( std::distance( elem_it, elem_end ) == 
        static_cast< std::ptrdiff_t >( CAPACITY ) );
    
    uchar_t* next_elem_data = nullptr;
    
    for( std::size_t ii = 0 ; elem_it != elem_end ; ++elem_it, ++ii )
    {
        ASSERT_TRUE( st_BeamElementInfo_is_available( elem_it ) );
        
        elem_type_t const elem_type_id = 
            st_BeamElementInfo_get_type_id( elem_it );
        
        ASSERT_TRUE( st_BeamElementInfo_get_element_id( elem_it ) ==
                     element_ids[ ii ] );
        
        uchar_t* curr_elem_data = static_cast< uchar_t* >(
            st_BeamElementInfo_get_ptr_mem_begin( elem_it ) );

        ASSERT_TRUE( curr_elem_data != nullptr );
        
        if( next_elem_data != nullptr )
        {
            ASSERT_TRUE( ( reinterpret_cast< std::uintptr_t >( 
                curr_elem_data ) % U64_SIZE ) == ZERO );
            
            ASSERT_TRUE( curr_elem_data == next_elem_data );
        }
        
        switch( elem_type_id )
        {
            case st_ELEMENT_TYPE_DRIFT:
            case st_ELEMENT_TYPE_DRIFT_EXACT:
            {
                drift_t drift;
                
                next_elem_data = st_Drift_unpack_from_flat_memory( &drift, curr_elem_data );
                
                ASSERT_TRUE( next_elem_data != nullptr );                
                ASSERT_TRUE( curr_elem_data != next_elem_data );
                ASSERT_TRUE( std::distance( curr_elem_data, next_elem_data ) > 0 );
                
                ASSERT_TRUE( st_Drift_get_element_id( &drift ) == element_ids[ ii ] );
                
                SIXTRL_REAL_T const delta_length= std::fabs(
                    lengths[ ii ] - st_Drift_get_length( &drift ) );
                
                ASSERT_TRUE( delta_length < 1e-16 ); 
                
                break;
            }
            
            default:
            {
                /* This should currently never happen -> 
                 * TODO: adapt this once this changes! */
                
                ASSERT_TRUE( false );
            }
        }
    }
    
    st_Block_free( blocks );
    free( blocks );
    blocks = nullptr;
}

TEST( CommonBlocksTests, CreateBlockSaveNumDrfitsAlignedUnpackAndThenFree )
{
    typedef st_BeamElementInfo elem_info_t;
    typedef st_BeamElementType elem_type_t;
    typedef st_Drift           drift_t;
    typedef unsigned char      uchar_t;
    
    static std::size_t const ZERO = std::size_t{ 0 };
    static std::size_t const ONE  = std::size_t{ 1 };    
    static std::size_t const U64_SIZE = sizeof( SIXTRL_UINT64_T );
    
    std::size_t const ALIGNMENT = std::size_t{ 32 };
    
    static std::size_t const CAPACITY = 100;    
    st_Block* blocks = st_Block_new( CAPACITY );
    
    ASSERT_TRUE( blocks != 0 );
    ASSERT_TRUE( st_Block_get_capacity( blocks ) >= CAPACITY );
    ASSERT_TRUE( st_Block_get_size( blocks ) == std::size_t{ 0 } );
    
    ASSERT_TRUE( st_Block_get_const_elements_begin( blocks ) != nullptr );
    ASSERT_TRUE( st_Block_get_const_elements_begin( blocks ) ==
                 st_Block_get_const_elements_end( blocks ) );
    
    ASSERT_TRUE( st_Block_manages_own_memory( blocks ) );
    ASSERT_TRUE( st_Block_uses_mempool( blocks ) );
    
    SIXTRL_INT64_T prev_element_id = SIXTRL_INT64_T{ -1 };
    SIXTRL_SIZE_T prev_size = st_Block_get_size( blocks );
    
    std::vector< SIXTRL_REAL_T > lengths( CAPACITY, SIXTRL_REAL_T{ 0 } );
    lengths.clear();
    
    std::vector< SIXTRL_INT64_T > element_ids( 
        CAPACITY, st_PARTICLES_INVALID_BEAM_ELEMENT_ID );
    
    element_ids.clear();
    
    for( std::size_t ii = 0 ; ii < CAPACITY ; ++ii )
    {
        SIXTRL_REAL_T const length = static_cast< SIXTRL_REAL_T >( ii ) + 0.5;
        SIXTRL_INT64_T const new_element_id = st_Block_append_drift_aligned( 
            blocks, st_ELEMENT_TYPE_DRIFT, length, ALIGNMENT );
        
        SIXTRL_SIZE_T const current_size = st_Block_get_size( blocks );
        
        ASSERT_TRUE( prev_element_id != new_element_id );
        ASSERT_TRUE( current_size <= CAPACITY );
        ASSERT_TRUE( prev_size + ONE == current_size ); 
        
        prev_size = current_size;
        prev_element_id = new_element_id;
        
        element_ids.push_back( new_element_id );
        lengths.push_back( length );
    }
    
    ASSERT_TRUE( element_ids.size() == CAPACITY );
    ASSERT_TRUE( lengths.size() == CAPACITY );    
    ASSERT_TRUE( st_Block_get_size( blocks ) == CAPACITY );
    
    elem_info_t* elem_it  = st_Block_get_elements_begin( blocks );
    elem_info_t* elem_end = st_Block_get_elements_end( blocks );
    
    ASSERT_TRUE( ( elem_it  != nullptr ) && ( elem_end != nullptr ) );
    ASSERT_TRUE( std::distance( elem_it, elem_end ) == 
        static_cast< std::ptrdiff_t >( CAPACITY ) );
    
    uchar_t* next_elem_data = nullptr;
    
    for( std::size_t ii = 0 ; elem_it != elem_end ; ++elem_it, ++ii )
    {
        ASSERT_TRUE( st_BeamElementInfo_is_available( elem_it ) );
        
        elem_type_t const elem_type_id = 
            st_BeamElementInfo_get_type_id( elem_it );
        
        ASSERT_TRUE( st_BeamElementInfo_get_element_id( elem_it ) ==
                     element_ids[ ii ] );
        
        uchar_t* curr_elem_data = static_cast< uchar_t* >(
            st_BeamElementInfo_get_ptr_mem_begin( elem_it ) );

        ASSERT_TRUE( curr_elem_data != nullptr );
        
        if( next_elem_data != nullptr )
        {
            ASSERT_TRUE( ( reinterpret_cast< std::uintptr_t >( 
                curr_elem_data ) % U64_SIZE ) == ZERO );
            
            ASSERT_TRUE( curr_elem_data == next_elem_data );
        }
        
        switch( elem_type_id )
        {
            case st_ELEMENT_TYPE_DRIFT:
            case st_ELEMENT_TYPE_DRIFT_EXACT:
            {
                drift_t drift;
                
                next_elem_data = st_Drift_unpack_from_flat_memory( &drift, curr_elem_data );
                
                ASSERT_TRUE( next_elem_data != nullptr );                
                ASSERT_TRUE( curr_elem_data != next_elem_data );
                ASSERT_TRUE( std::distance( curr_elem_data, next_elem_data ) > 0 );
                
                ASSERT_TRUE( st_Drift_get_element_id( &drift ) == element_ids[ ii ] );
                
                SIXTRL_REAL_T const delta_length= std::fabs(
                    lengths[ ii ] - st_Drift_get_length( &drift ) );
                
                ASSERT_TRUE( delta_length < 1e-16 ); 
                
                break;
            }
            
            default:
            {
                /* This should currently never happen -> 
                 * TODO: adapt this once this changes! */
                
                ASSERT_TRUE( false );
            }
        }
    }
    
    st_Block_free( blocks );
    free( blocks );
    blocks = nullptr;
}

/* end: sixtracklib/common/tests/test_blocks.cpp */
