#define _USE_MATH_DEFINES 

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#include <gtest/gtest.h>

#if defined( __NAMESPACE )
    #define __SAVED_NAMESPACE __NAMESPACE
    #undef  __NAMESPACE     
#endif /* !defiend( __NAMESPACE ) */

#if !defined( __NAMESPACE )
    #define __NAMESPACE st_    
#endif /* !defiend( __NAMESPACE ) */

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/path.h"
#include "sixtracklib/testdata/tracking_testfiles.h"

#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/track.h"
#include "sixtracklib/common/details/random.h"

#include "sixtracklib/common/tests/test_particles_tools.h"
#include "sixtracklib/common/tests/test_track_tools.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

/* ************************************************************************* */
/* *****                TESTS FOR BEAM_ELEMENT DRIFT                   ***** */
/* ************************************************************************* */

/* ========================================================================= */
/* ====   Test using the drift-only testdata file and call              ==== */
/* ====   the minimal st_Track_drift_particle() function from track.h   ==== */
/* ====   manually -> i.e. do everything by hand, incl. elem_by_elem io ==== */
/* ====                                                                 ==== */
/* ====   Since this has to be repeated for each and every BeamElement, ==== */
/* ====   and to avoid code duplication even in the unit-tests, cf.     ==== */
/* ====   the sixtracklib/common/tests/test_track_tools.h file for the  ==== */
/* ====   declaration and definition of the helper function which       ==== */
/* ====   contains the whole unit-test!                                 ==== */
/* ========================================================================= */

TEST( CommonTrackTests, TrackDriftParticle )
{
    bool const success = 
        ::st_TestData_test_tracking_single_particle_over_specific_be_type< 
            st_Drift >( st_PATH_TO_TEST_TRACKING_DRIFT_DATA, 
                        st_Track_drift_particle );
        
    ASSERT_TRUE( success );
}

/* ========================================================================= */
/* ====   Test using the drift-only testdata file and call              ==== */
/* ====   st_Track_drift() function from track_api.h. Expected are same ==== */
/* ====   results as from st_Track_drift()                              ==== */
/* ====                                                                 ==== */
/* ====   Since this has to be repeated for each and every BeamElement, ==== */
/* ====   and to avoid code duplication even in the unit-tests, cf.     ==== */
/* ====   the sixtracklib/common/tests/test_track_tools.h file for the  ==== */
/* ====   declaration and definition of the helper function which       ==== */
/* ====   contains the whole unit-test!                                 ==== */
/* ========================================================================= */

TEST( CommonTrackTests, TrackDrift )
{
    bool const success =
        ::st_TestData_test_tracking_particles_over_specific_be_type< 
            st_Drift >( st_PATH_TO_TEST_TRACKING_DRIFT_DATA, 
                        st_Track_drift );
        
    ASSERT_TRUE( success );
}

/* ========================================================================= */
/* ====   Test using the drift-only testdata file and call              ==== */
/* ====   st_Track_beam_elements_particle() function from track_api.h.  ==== */
/* ====   Expected are same   results as from st_Track_drift_particle() ==== */
/* ====                                                                 ==== */
/* ====   Since this has to be repeated for each and every BeamElement, ==== */
/* ====   and to avoid code duplication even in the unit-tests, cf.     ==== */
/* ====   the sixtracklib/common/tests/test_track_tools.h and           ==== */
/* ====   sixtracklib/common/tests/test_track_tools.c files for the     ==== */
/* ====   declaration and definition of the helper functionm which      ==== */
/* ====   contains the whole unit-test, respectively!                   ==== */
/* ========================================================================= */

TEST( CommonTrackTests, TrackBeamElementParticleForDrift )
{
    bool const success = ::st_TestData_test_tracking_single_particle(
        st_PATH_TO_TEST_TRACKING_DRIFT_DATA );
    
    ASSERT_TRUE( success );
}

/* ========================================================================= */
/* ====   Test using the drift-only testdata file and call              ==== */
/* ====   st_Track_beam_elements_particle() function from track_api.h.  ==== */
/* ====   Expected are same   results as from st_Track_drift_particle() ==== */
/* ====                                                                 ==== */
/* ====   Since this has to be repeated for each and every BeamElement, ==== */
/* ====   and to avoid code duplication even in the unit-tests, cf.     ==== */
/* ====   the sixtracklib/common/tests/test_track_tools.h and           ==== */
/* ====   sixtracklib/common/tests/test_track_tools.c files for the     ==== */
/* ====   declaration and definition of the helper functionm which      ==== */
/* ====   contains the whole unit-test, respectively!                   ==== */
/* ========================================================================= */

TEST( CommonTrackTests, TrackBeamElementForDrift )
{
    bool const success = ::st_TestData_test_tracking_particles(
        st_PATH_TO_TEST_TRACKING_DRIFT_DATA );
    
    ASSERT_TRUE( success );
}

/* ************************************************************************* */

TEST( CommonTrackTests, TrackDriftExacts )
{
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
    
    /* --------------------------------------------------------------------- */
    
    st_block_size_t const NUM_OF_TURNS         = st_block_size_t{ 100 };
    st_block_size_t const NUM_OF_PARTICLES     = st_block_size_t{ 100 };
    st_block_size_t const NUM_OF_BEAM_ELEMENTS = st_block_size_t{ 100 };
    
    st_Blocks beam_elements;
    st_Blocks_preset( &beam_elements );
    
    st_block_size_t const BEAM_ELEMENTS_DATA_CAPACITY =
        st_Blocks_predict_data_capacity_for_num_blocks( 
            &beam_elements, NUM_OF_BEAM_ELEMENTS ) +
        st_Drift_predict_blocks_data_capacity( 
            &beam_elements, NUM_OF_BEAM_ELEMENTS );
    
    int ret = st_Blocks_init( 
        &beam_elements, NUM_OF_BEAM_ELEMENTS, BEAM_ELEMENTS_DATA_CAPACITY );
    
    std::vector< st_DriftExact* > ptr_beam_elem( NUM_OF_BEAM_ELEMENTS, nullptr );
    
    SIXTRL_REAL_T const DRIFT_LEN = SIXTRL_REAL_T{ 0.5L };
    
    for( st_block_size_t ii = 0 ; ii < NUM_OF_BEAM_ELEMENTS ; ++ii )
    {
        ptr_beam_elem[ ii ] = st_Blocks_add_drift_exact( 
            &beam_elements, DRIFT_LEN );
        
        ASSERT_TRUE( ptr_beam_elem[ ii ] != nullptr );
    }
    
    ASSERT_TRUE( st_Blocks_serialize( &beam_elements ) == 0 );
    ASSERT_TRUE( st_Blocks_are_serialized( &beam_elements ) );
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &beam_elements ) == 
                 NUM_OF_BEAM_ELEMENTS );
    
    ASSERT_TRUE( ret == 0 );
    
    /* --------------------------------------------------------------------- */
    
    st_Blocks particles_buffer;
    st_Blocks_preset( &particles_buffer );
    
    st_Blocks copy_particles_buffer;
    st_Blocks_preset( &copy_particles_buffer );
    
    st_block_size_t const PARTICLES_DATA_CAPACITY = 
        st_Blocks_predict_data_capacity_for_num_blocks(
            &particles_buffer, 1u ) +
        st_Particles_predict_blocks_data_capacity(
            &particles_buffer, 1u, NUM_OF_PARTICLES );
        
    ret = st_Blocks_init( &particles_buffer, 1u, PARTICLES_DATA_CAPACITY );        
    ASSERT_TRUE( ret == 0 );
    
    ret = st_Blocks_init( &copy_particles_buffer, 1u, PARTICLES_DATA_CAPACITY );
    ASSERT_TRUE( ret == 0 );
    
    st_Particles* initial_particles = 
        st_Blocks_add_particles( &copy_particles_buffer, NUM_OF_PARTICLES );
    
    ASSERT_TRUE( initial_particles != nullptr );
    ASSERT_TRUE( st_Blocks_serialize( &copy_particles_buffer ) == 0 );
    ASSERT_TRUE( st_Blocks_are_serialized( &copy_particles_buffer ) );
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &copy_particles_buffer ) == 1u );
    
    st_Particles* particles = 
        st_Blocks_add_particles( &particles_buffer, NUM_OF_PARTICLES );
    
    ASSERT_TRUE( particles != nullptr );
    ASSERT_TRUE( st_Blocks_serialize( &particles_buffer ) == 0 );
    ASSERT_TRUE( st_Blocks_are_serialized( &particles_buffer ) );
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &particles_buffer ) == 1u );
    
    st_Particles_random_init( particles );
    st_Particles_copy_all_unchecked( initial_particles, particles );
    
    /* --------------------------------------------------------------------- */
    
    st_Blocks elem_by_elem;
    st_Blocks_preset( &elem_by_elem );
    
    st_block_size_t const NUM_OF_ELEM_BY_ELEM_BLOCKS =
        NUM_OF_TURNS * NUM_OF_BEAM_ELEMENTS;
    
    st_block_size_t const ELEM_BY_ELEM_DATA_CAPACITY =
        st_Blocks_predict_data_capacity_for_num_blocks(
            &elem_by_elem, NUM_OF_ELEM_BY_ELEM_BLOCKS ) +
        st_Particles_predict_blocks_data_capacity(
            &elem_by_elem, NUM_OF_ELEM_BY_ELEM_BLOCKS, NUM_OF_PARTICLES );

    ret = st_Blocks_init( &elem_by_elem, NUM_OF_ELEM_BY_ELEM_BLOCKS, 
                          ELEM_BY_ELEM_DATA_CAPACITY );
    
    ASSERT_TRUE( ret == 0 );
    
    std::vector< st_Particles* > elem_by_elem_particles( 
        NUM_OF_ELEM_BY_ELEM_BLOCKS, nullptr );
    
    for( st_block_size_t ii = 0 ; ii <  NUM_OF_ELEM_BY_ELEM_BLOCKS ; ++ii )
    {
        elem_by_elem_particles[ ii ] = st_Blocks_add_particles( 
            &elem_by_elem, NUM_OF_PARTICLES );
        
        ASSERT_TRUE( elem_by_elem_particles[ ii ] != nullptr );
        ASSERT_TRUE( elem_by_elem_particles[ ii ]->num_of_particles == 
                     NUM_OF_PARTICLES );
    }
    
    ASSERT_TRUE( st_Blocks_serialize( &elem_by_elem ) == 0 );
    ASSERT_TRUE( st_Blocks_are_serialized( &elem_by_elem ) );
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &elem_by_elem ) == 
        NUM_OF_ELEM_BY_ELEM_BLOCKS );
    
    /* --------------------------------------------------------------------- */
    
    st_BlockInfo* io_block_it = 
        st_Blocks_get_block_infos_begin( &elem_by_elem );
        
    for( st_block_size_t ii = 0 ; ii < NUM_OF_TURNS ; ++ii )
    {
        st_BlockInfo const* be_info_it = 
            st_Blocks_get_const_block_infos_begin( &beam_elements );
        
        for( st_block_size_t jj = 0 ; jj < NUM_OF_BEAM_ELEMENTS ; 
                ++jj, ++be_info_it, ++io_block_it )
        {
            st_DriftExact const* drift = 
                ( st_DriftExact const* )st_Blocks_get_const_drift_exact( 
                    be_info_it );
                    
            ret = st_Track_drift_exact( particles, 0u, NUM_OF_PARTICLES, 
                drift, st_Blocks_get_particles( io_block_it ) );
            
            ASSERT_TRUE( ret == 0 );
        }        
    }
    
    std::ostringstream a2str("");
    a2str << st_PATH_TO_BASE_DIR 
          << "sixtracklib/common/tests/testdata/test_track_drift_exact"
          << "__nturn"
          << std::setfill( '0' ) << std::setw( 6 ) << NUM_OF_TURNS
          << "__npart" 
          << std::setw( 6 ) << NUM_OF_PARTICLES 
          << "__nelem" 
          << std::setw( 4 ) << NUM_OF_BEAM_ELEMENTS
          << "__driftlen" << std::setw( 4 )
          << static_cast< int >( 1000.0 * DRIFT_LEN + 0.5 )
          << "mm.bin";
          
    std::ofstream test_data_out( 
        a2str.str().c_str(), std::ios::binary | std::ios::out );
    
    test_data_out << NUM_OF_TURNS;
    test_data_out << NUM_OF_PARTICLES;
    test_data_out << NUM_OF_BEAM_ELEMENTS;
    
    uint64_t const beam_elements_bytes = 
        st_Blocks_get_total_num_bytes( &beam_elements );
    
    test_data_out << beam_elements_bytes;
    test_data_out.write( 
        ( char* )st_Blocks_get_const_data_begin( &beam_elements ), 
        beam_elements_bytes );
    
    uint64_t const initial_particles_size =
        st_Blocks_get_total_num_bytes( &copy_particles_buffer );
        
    test_data_out << initial_particles_size;
    test_data_out.write(
        ( char* )st_Blocks_get_const_data_begin( &copy_particles_buffer ),
        initial_particles_size );
    
    uint64_t const elem_by_elem_size =
        st_Blocks_get_total_num_bytes( &elem_by_elem );
        
    test_data_out << elem_by_elem_size;
    test_data_out.write(
        ( char* )st_Blocks_get_const_data_begin( &elem_by_elem ),
        elem_by_elem_size );
    
    uint64_t const end_particles_size =
        st_Blocks_get_total_num_bytes( &particles_buffer );
        
    test_data_out << end_particles_size;
    test_data_out.write(
        ( char* )st_Blocks_get_const_data_begin( &particles_buffer ),
        end_particles_size );
    
    test_data_out.close();
    
    /* --------------------------------------------------------------------- */
    
    particles = nullptr;
    initial_particles = nullptr;
    
    st_Blocks_free( &elem_by_elem );
    st_Blocks_free( &copy_particles_buffer );
    st_Blocks_free( &beam_elements );
    st_Blocks_free( &particles_buffer );
}

/* ************************************************************************* */

TEST( CommonTrackTests, TrackMultiPoles )
{
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
    
    /* --------------------------------------------------------------------- */
    
    st_block_size_t const NUM_OF_TURNS         = st_block_size_t{ 100 };
    st_block_size_t const NUM_OF_PARTICLES     = st_block_size_t{ 100 };
    st_block_size_t const NUM_OF_BEAM_ELEMENTS = st_block_size_t{ 100 };
    
    st_Blocks beam_elements;
    st_Blocks_preset( &beam_elements );
    
    st_block_size_t const BEAM_ELEMENTS_DATA_CAPACITY =
        st_Blocks_predict_data_capacity_for_num_blocks( 
            &beam_elements, NUM_OF_BEAM_ELEMENTS ) +
        st_MultiPole_predict_blocks_data_capacity( 
            &beam_elements, NUM_OF_BEAM_ELEMENTS, 4 );
    
    int ret = st_Blocks_init( 
        &beam_elements, NUM_OF_BEAM_ELEMENTS, BEAM_ELEMENTS_DATA_CAPACITY );
    
    std::vector< st_MultiPole* > ptr_beam_elem( NUM_OF_BEAM_ELEMENTS, nullptr );
    
    SIXTRL_REAL_T  const MULTIPOLE_LENGTH = SIXTRL_REAL_T{ 0.5L };
    SIXTRL_REAL_T  const MULTIPOLE_HXL    = SIXTRL_REAL_T{ 0.5L };
    SIXTRL_REAL_T  const MULTIPOLE_HYL    = SIXTRL_REAL_T{ 0.5L };
    SIXTRL_INT64_T const MULTIPOLE_ORDER  = SIXTRL_INT64_T{ 2 };
    
    SIXTRL_REAL_T  const MULTIPOLE_BAL[]  =
    {
        SIXTRL_REAL_T{ 0.1 }, SIXTRL_REAL_T{ 0.1 }, 
        SIXTRL_REAL_T{ 0.3 }, SIXTRL_REAL_T{ 0.3 }, 
        SIXTRL_REAL_T{ 0.5 }, SIXTRL_REAL_T{ 0.5 }
    };
    
    for( st_block_size_t ii = 0 ; ii < NUM_OF_BEAM_ELEMENTS ; ++ii )
    {
        ptr_beam_elem[ ii ] = st_Blocks_add_multipole( 
            &beam_elements, MULTIPOLE_LENGTH, MULTIPOLE_HXL, MULTIPOLE_HYL,
                MULTIPOLE_ORDER, &MULTIPOLE_BAL[ 0 ] );
        
        ASSERT_TRUE( ptr_beam_elem[ ii ] != nullptr );
    }
    
    ASSERT_TRUE( st_Blocks_serialize( &beam_elements ) == 0 );
    ASSERT_TRUE( st_Blocks_are_serialized( &beam_elements ) );
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &beam_elements ) == 
                 NUM_OF_BEAM_ELEMENTS );
    
    ASSERT_TRUE( ret == 0 );
    
    /* --------------------------------------------------------------------- */
    
    st_Blocks particles_buffer;
    st_Blocks_preset( &particles_buffer );
    
    st_Blocks copy_particles_buffer;
    st_Blocks_preset( &copy_particles_buffer );
    
    st_block_size_t const PARTICLES_DATA_CAPACITY = 
        st_Blocks_predict_data_capacity_for_num_blocks(
            &particles_buffer, 1u ) +
        st_Particles_predict_blocks_data_capacity(
            &particles_buffer, 1u, NUM_OF_PARTICLES );
        
    ret = st_Blocks_init( &particles_buffer, 1u, PARTICLES_DATA_CAPACITY );        
    ASSERT_TRUE( ret == 0 );
    
    ret = st_Blocks_init( &copy_particles_buffer, 1u, PARTICLES_DATA_CAPACITY );
    ASSERT_TRUE( ret == 0 );
    
    st_Particles* initial_particles = 
        st_Blocks_add_particles( &copy_particles_buffer, NUM_OF_PARTICLES );
    
    ASSERT_TRUE( initial_particles != nullptr );
    ASSERT_TRUE( st_Blocks_serialize( &copy_particles_buffer ) == 0 );
    ASSERT_TRUE( st_Blocks_are_serialized( &copy_particles_buffer ) );
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &copy_particles_buffer ) == 1u );
    
    st_Particles* particles = 
        st_Blocks_add_particles( &particles_buffer, NUM_OF_PARTICLES );
    
    ASSERT_TRUE( particles != nullptr );
    ASSERT_TRUE( st_Blocks_serialize( &particles_buffer ) == 0 );
    ASSERT_TRUE( st_Blocks_are_serialized( &particles_buffer ) );
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &particles_buffer ) == 1u );
    
    st_Particles_random_init( particles );
    st_Particles_copy_all_unchecked( initial_particles, particles );
    
    /* --------------------------------------------------------------------- */
    
    st_Blocks elem_by_elem;
    st_Blocks_preset( &elem_by_elem );
    
    st_block_size_t const NUM_OF_ELEM_BY_ELEM_BLOCKS =
        NUM_OF_TURNS * NUM_OF_BEAM_ELEMENTS;
    
    st_block_size_t const ELEM_BY_ELEM_DATA_CAPACITY =
        st_Blocks_predict_data_capacity_for_num_blocks(
            &elem_by_elem, NUM_OF_ELEM_BY_ELEM_BLOCKS ) +
        st_Particles_predict_blocks_data_capacity(
            &elem_by_elem, NUM_OF_ELEM_BY_ELEM_BLOCKS, NUM_OF_PARTICLES );

    ret = st_Blocks_init( &elem_by_elem, NUM_OF_ELEM_BY_ELEM_BLOCKS, 
                          ELEM_BY_ELEM_DATA_CAPACITY );
    
    ASSERT_TRUE( ret == 0 );
    
    std::vector< st_Particles* > elem_by_elem_particles( 
        NUM_OF_ELEM_BY_ELEM_BLOCKS, nullptr );
    
    for( st_block_size_t ii = 0 ; ii <  NUM_OF_ELEM_BY_ELEM_BLOCKS ; ++ii )
    {
        elem_by_elem_particles[ ii ] = st_Blocks_add_particles( 
            &elem_by_elem, NUM_OF_PARTICLES );
        
        ASSERT_TRUE( elem_by_elem_particles[ ii ] != nullptr );
        ASSERT_TRUE( elem_by_elem_particles[ ii ]->num_of_particles == 
                     NUM_OF_PARTICLES );
    }
    
    ASSERT_TRUE( st_Blocks_serialize( &elem_by_elem ) == 0 );
    ASSERT_TRUE( st_Blocks_are_serialized( &elem_by_elem ) );
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &elem_by_elem ) == 
        NUM_OF_ELEM_BY_ELEM_BLOCKS );
    
    /* --------------------------------------------------------------------- */
    
    st_BlockInfo* io_block_it = 
        st_Blocks_get_block_infos_begin( &elem_by_elem );
        
    for( st_block_size_t ii = 0 ; ii < NUM_OF_TURNS ; ++ii )
    {
        st_BlockInfo const* be_info_it = 
            st_Blocks_get_const_block_infos_begin( &beam_elements );
        
        for( st_block_size_t jj = 0 ; jj < NUM_OF_BEAM_ELEMENTS ; 
                ++jj, ++be_info_it, ++io_block_it )
        {
            st_MultiPole const* multipole = 
                ( st_MultiPole const* )st_Blocks_get_const_multipole( 
                    be_info_it );
                    
            ret = st_Track_multipole( particles, 0u, NUM_OF_PARTICLES, 
                multipole, st_Blocks_get_particles( io_block_it ) );
            
            ASSERT_TRUE( ret == 0 );
        }        
    }
    
    std::ostringstream a2str("");
    a2str << st_PATH_TO_BASE_DIR 
          << "sixtracklib/common/tests/testdata/test_track_multipole"
          << "__nturn"
          << std::setfill( '0' ) << std::setw( 6 ) << NUM_OF_TURNS
          << "__npart" 
          << std::setw( 6 ) << NUM_OF_PARTICLES 
          << "__nelem" 
          << std::setw( 4 ) << NUM_OF_BEAM_ELEMENTS
          << "__mplen" << std::setw( 4 )
          << static_cast< int >( 1000.0 * MULTIPOLE_LENGTH + 0.5 )
          << "mm__mphxl" << std::setw( 4 )
          << static_cast< int >( 1000.0 * MULTIPOLE_HXL + 0.5 )
          << "mm__mphyl" << std::setw( 4 )
          << static_cast< int >( 1000.0 * MULTIPOLE_HYL + 0.5 )
          << "mm__order" << std::setw( 2 )
          << MULTIPOLE_ORDER
          << ".bin";
          
    std::ofstream test_data_out( 
        a2str.str().c_str(), std::ios::binary | std::ios::out );
    
    test_data_out << NUM_OF_TURNS;
    test_data_out << NUM_OF_PARTICLES;
    test_data_out << NUM_OF_BEAM_ELEMENTS;
    
    uint64_t const beam_elements_bytes = 
        st_Blocks_get_total_num_bytes( &beam_elements );
    
    test_data_out << beam_elements_bytes;
    test_data_out.write( 
        ( char* )st_Blocks_get_const_data_begin( &beam_elements ), 
        beam_elements_bytes );
    
    uint64_t const initial_particles_size =
        st_Blocks_get_total_num_bytes( &copy_particles_buffer );
        
    test_data_out << initial_particles_size;
    test_data_out.write(
        ( char* )st_Blocks_get_const_data_begin( &copy_particles_buffer ),
        initial_particles_size );
    
    uint64_t const elem_by_elem_size =
        st_Blocks_get_total_num_bytes( &elem_by_elem );
        
    test_data_out << elem_by_elem_size;
    test_data_out.write(
        ( char* )st_Blocks_get_const_data_begin( &elem_by_elem ),
        elem_by_elem_size );
    
    uint64_t const end_particles_size =
        st_Blocks_get_total_num_bytes( &particles_buffer );
        
    test_data_out << end_particles_size;
    test_data_out.write(
        ( char* )st_Blocks_get_const_data_begin( &particles_buffer ),
        end_particles_size );
    
    test_data_out.close();
    
    /* --------------------------------------------------------------------- */
    
    particles = nullptr;
    initial_particles = nullptr;
    
    st_Blocks_free( &elem_by_elem );
    st_Blocks_free( &copy_particles_buffer );
    st_Blocks_free( &beam_elements );
    st_Blocks_free( &particles_buffer );
}

namespace st
{
    void process_testdata_per_particle( const char *const PATH_TO_DATA_FILE );
    void process_testdata( const char *const PATH_TO_DATA_FILE );    
}

/* end: sixtracklib/common/tests/test_particles.cpp */
