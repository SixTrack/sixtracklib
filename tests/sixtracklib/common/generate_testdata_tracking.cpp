#define _USE_MATH_DEFINES

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <vector>

#include "sixtracklib/testlib.h"
#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/path.h"

#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/track.h"
#include "sixtracklib/common/impl/track_api.h"

namespace st
{
    static uint64_t const DEFAULT_PRNG_SEED = uint64_t{ 20180420 };

    bool create_testdata_drift(
        uint64_t const num_of_turns,
        uint64_t const num_of_particles,
        uint64_t const num_of_beam_elements,
        bool const generate_elem_by_elem_buffer,
        std::string const& path_to_bin_testdata_file,
        uint64_t const prng_seed = DEFAULT_PRNG_SEED );

    bool create_testdata_drift_exact(
        uint64_t const num_of_turns,
        uint64_t const num_of_particles,
        uint64_t const num_of_beam_elements,
        bool const generate_elem_by_elem_buffer,
        std::string const& path_to_bin_testdata_file,
        uint64_t const prng_seed = DEFAULT_PRNG_SEED );

    bool create_testdata_multipole(
        uint64_t const num_of_turns,
        uint64_t const num_of_particles,
        uint64_t const num_of_beam_elements,
        bool const generate_elem_by_elem_buffer,
        std::string const& path_to_bin_testdata_file,
        uint64_t const prng_seed = DEFAULT_PRNG_SEED );

    bool create_testdata_beam_beam(
        uint64_t const num_of_turns,
        uint64_t const num_of_particles,
        uint64_t const num_of_beam_elements,
        bool const generate_elem_by_elem_buffer,
        std::string const& path_to_bin_testdata_file,
        uint64_t const prng_seed = DEFAULT_PRNG_SEED );

    bool create_testdata_cavity(
        uint64_t const num_of_turns,
        uint64_t const num_of_particles,
        uint64_t const num_of_beam_elements,
        bool const generate_elem_by_elem_buffer,
        std::string const& path_to_bin_testdata_file,
        uint64_t const prng_seed = DEFAULT_PRNG_SEED );

    bool create_testdata_align(
        uint64_t const num_of_turns,
        uint64_t const num_of_particles,
        uint64_t const num_of_beam_elements,
        bool const generate_elem_by_elem_buffer,
        std::string const& path_to_bin_testdata_file,
        uint64_t const prng_seed = DEFAULT_PRNG_SEED );
}

int main( int argc, char* argv[] )
{
    uint64_t NUM_OF_TURNS               = uint64_t{   10 };
    uint64_t NUM_OF_PARTICLES           = uint64_t{ 1000 };
    uint64_t NUM_OF_BEAM_ELEMENTS       = uint64_t{  100 };
    uint64_t PRNG_SEED                  = st::DEFAULT_PRNG_SEED;
    bool     GENERATE_ELEM_BY_ELEM_DATA = false;

    if( argc > 1 )
    {
        NUM_OF_TURNS = static_cast< uint64_t >(
            std::atoi( argv[ 1 ] ) );
    }

    if( argc > 2 )
    {
        NUM_OF_BEAM_ELEMENTS = static_cast< uint64_t >(
            std::atoi( argv[ 2 ] ) );
    }

    if( argc > 3 )
    {
        NUM_OF_PARTICLES = static_cast< uint64_t >( std::atoi( argv[ 3 ] ) );
    }

    if( argc > 4 )
    {
        int const temp = std::atoi( argv[ 4 ] );
        GENERATE_ELEM_BY_ELEM_DATA = ( temp == 1 );
    }

    if( argc > 5 )
    {
        PRNG_SEED = static_cast< uint64_t >( std::atoi( argv[ 5 ] ) );
    }

    ::FILE* fp = std::fopen( NS(PATH_TO_TEST_TRACKING_STAMP_FILE), "rb" );

    if( fp != 0 )
    {
        std::fclose( fp );
        fp = nullptr;
        std::remove( NS(PATH_TO_TEST_TRACKING_STAMP_FILE) );
    }

    bool success = ( fp == nullptr );

    std::cout << "======================================================="
                 "=======================================================\r\n"
              << "generate data files with the following parameters: \r\n"
              << " num_of_turns         : "
              << std::setw( 10 ) << NUM_OF_TURNS << "\r\n"
              << " num_of_beam_elements : "
              << std::setw( 10 ) << NUM_OF_BEAM_ELEMENTS << "\r\n"
              << " num_of_particles     : "
              << std::setw( 10 ) << NUM_OF_PARTICLES << "\r\n"
              << " geneate elem_by_elem : "
              << std::boolalpha << GENERATE_ELEM_BY_ELEM_DATA << "\r\n"
              << " prng random seed     : "
              << std::setw( 10 ) << PRNG_SEED << "\r\n\r\n";

    std::cout << "-------------------------------------------------------"
                 "-------------------------------------------------------\r\n"
              << "Generating drift testdata ["
              << st_PATH_TO_TEST_TRACKING_DRIFT_DATA << "] .... ";

    if( st::create_testdata_drift(
            NUM_OF_TURNS, NUM_OF_PARTICLES, NUM_OF_BEAM_ELEMENTS,
            GENERATE_ELEM_BY_ELEM_DATA, st_PATH_TO_TEST_TRACKING_DRIFT_DATA,
            PRNG_SEED ) )
    {
        std::cout << "SUCCESS" << std::endl << std::endl;
    }
    else
    {
        success = false;
        std::cout << "FAILURE" << std::endl << std::endl;
    }

    std::cout << "-------------------------------------------------------"
                 "-------------------------------------------------------\r\n"
              << "Generating drift_exact testdata ["
              << st_PATH_TO_TEST_TRACKING_DRIFT_EXACT_DATA << "] .... ";

    if( st::create_testdata_drift_exact(
            NUM_OF_TURNS, NUM_OF_PARTICLES, NUM_OF_BEAM_ELEMENTS,
            GENERATE_ELEM_BY_ELEM_DATA,
            st_PATH_TO_TEST_TRACKING_DRIFT_EXACT_DATA,
            PRNG_SEED ) )
    {
        std::cout << "SUCCESS" << std::endl << std::endl;
    }
    else
    {
        success = false;
        std::cout << "FAILURE" << std::endl << std::endl;
    }

    std::cout << "-------------------------------------------------------"
                 "-------------------------------------------------------\r\n"
              << "Generating multipole testdata ["
              << st_PATH_TO_TEST_TRACKING_MULTIPOLE_DATA << "] .... ";

    if( st::create_testdata_multipole(
            NUM_OF_TURNS, NUM_OF_PARTICLES, NUM_OF_BEAM_ELEMENTS,
            GENERATE_ELEM_BY_ELEM_DATA,
            st_PATH_TO_TEST_TRACKING_MULTIPOLE_DATA,
            PRNG_SEED ) )
    {
        std::cout << "SUCCESS" << std::endl << std::endl;
    }
    else
    {
        success = false;
        std::cout << "FAILURE" << std::endl << std::endl;
    }

    /*

    std::cout << "-------------------------------------------------------"
                 "-------------------------------------------------------\r\n"
              << "Generating beam_beam testdata ["
              << st_PATH_TO_TEST_TRACKING_BEAM_BEAM_DATA << "] .... ";

    if( st::create_testdata_beam_beam(
            NUM_OF_TURNS, NUM_OF_PARTICLES, NUM_OF_BEAM_ELEMENTS,
            GENERATE_ELEM_BY_ELEM_DATA,
            st_PATH_TO_TEST_TRACKING_BEAM_BEAM_DATA,
            PRNG_SEED ) )
    {
        std::cout << "SUCCESS" << std::endl << std::endl;
    }
    else
    {
        success = false;
        std::cout << "FAILURE" << std::endl << std::endl;
    }

    */

    std::cout << "-------------------------------------------------------"
                 "-------------------------------------------------------\r\n"
              << "Generating cavity testdata ["
              << st_PATH_TO_TEST_TRACKING_CAVITY_DATA << "] .... ";

    if( st::create_testdata_cavity(
            NUM_OF_TURNS, NUM_OF_PARTICLES, NUM_OF_BEAM_ELEMENTS,
            GENERATE_ELEM_BY_ELEM_DATA,
            st_PATH_TO_TEST_TRACKING_CAVITY_DATA,
            PRNG_SEED ) )
    {
        std::cout << "SUCCESS" << std::endl << std::endl;
    }
    else
    {
        success = false;
        std::cout << "FAILURE" << std::endl << std::endl;
    }

    std::cout << "-------------------------------------------------------"
                 "-------------------------------------------------------\r\n"
              << "Generating align testdata ["
              << st_PATH_TO_TEST_TRACKING_ALIGN_DATA << "] .... ";

    if( st::create_testdata_align( NUM_OF_TURNS, NUM_OF_PARTICLES,
            NUM_OF_BEAM_ELEMENTS, GENERATE_ELEM_BY_ELEM_DATA,
            st_PATH_TO_TEST_TRACKING_ALIGN_DATA, PRNG_SEED ) )
    {
        std::cout << "SUCCESS" << std::endl << std::endl;
    }
    else
    {
        success = false;
        std::cout << "FAILURE" << std::endl << std::endl;
    }

    if( success )
    {
        assert( fp == nullptr );

        uint64_t flag = 0;
        fp = std::fopen( NS(PATH_TO_TEST_TRACKING_STAMP_FILE), "wb" );
        fwrite( &flag, sizeof( uint64_t ), 1, fp );
        std::fflush( fp );
        std::fclose( fp );
        fp = 0;
    }

    return 0;
}

/* ========================================================================= */

namespace st
{
    bool create_testdata_drift(
        uint64_t const num_of_turns,
        uint64_t const num_of_particles,
        uint64_t const num_of_beam_elements,
        bool const generate_elem_by_elem_buffer,
        std::string const& path_to_bin_testdata_file,
        uint64_t const prng_seed )
    {
        bool success = false;

        if( ( num_of_turns > uint64_t{ 0 } ) &&
            ( num_of_particles > uint64_t{ 0 } ) &&
            ( num_of_beam_elements > uint64_t{ 0 } ) &&
            ( !path_to_bin_testdata_file.empty() ) )
        {
            st_Random_init_genrand64( prng_seed );

            /* ------------------------------------------------------------ */

            uint64_t const num_of_particle_blocks = uint64_t{ 1 };

            st_Blocks initial_particles_buffer;
            st_Blocks_preset( &initial_particles_buffer );

            st_Blocks result_particles_buffer;
            st_Blocks_preset( &result_particles_buffer );

            st_block_size_t const initial_particles_data_capacity =
                st_Blocks_predict_data_capacity_for_num_blocks(
                    &initial_particles_buffer, num_of_particle_blocks ) +
                st_Particles_predict_blocks_data_capacity(
                    &initial_particles_buffer,
                        num_of_particle_blocks, num_of_particles );

            success = ( 0 == st_Blocks_init( &initial_particles_buffer,
                num_of_particle_blocks, initial_particles_data_capacity ) );

            st_Particles* initial_particles = st_Blocks_add_particles(
                &initial_particles_buffer, num_of_particles );

            success &= ( initial_particles != nullptr );
            st_Particles_random_init( initial_particles );

            success &= ( 0 == st_Blocks_init( &result_particles_buffer,
                num_of_particle_blocks, initial_particles_data_capacity ) );

            st_Particles* particles = st_Blocks_add_particles(
                &result_particles_buffer, num_of_particles );

            success &= ( particles != nullptr );

            st_Particles_copy_all_unchecked( particles, initial_particles );

            success &= ( 0 == st_Blocks_serialize(
                &initial_particles_buffer ) );

            success &= ( 0 == st_Blocks_serialize(
                &result_particles_buffer ) );

            /* ------------------------------------------------------------- */

            double const MIN_DRIFT_LENGTH = double{ 0.1 };
            double const MAX_DRIFT_LENGTH = double{ 1.0 };
            double const DRIFT_LENGTH_RANGE =
                MAX_DRIFT_LENGTH - MIN_DRIFT_LENGTH;

            st_Blocks beam_elements;
            st_Blocks_preset( &beam_elements );

            st_block_size_t const beam_elements_data_capacity =
                st_Blocks_predict_data_capacity_for_num_blocks(
                    &beam_elements, num_of_beam_elements ) +
                st_Drift_predict_blocks_data_capacity(
                    &beam_elements, num_of_beam_elements );

            success &= ( 0 == st_Blocks_init( &beam_elements,
                        num_of_beam_elements, beam_elements_data_capacity ) );

            for( st_block_size_t ii = 0 ; ii < num_of_beam_elements ; ++ii )
            {
                double const drift_length = MIN_DRIFT_LENGTH +
                    DRIFT_LENGTH_RANGE * st_Random_genrand64_real1();

                st_Drift* ptr_drift = st_Blocks_add_drift(
                    &beam_elements, drift_length );

                success &= ( ptr_drift != nullptr );
            }

            success &= ( st_Blocks_serialize( &beam_elements ) == 0 );

            /* ------------------------------------------------------------- */

            st_Blocks elem_by_elem_buffer;
            st_Blocks_preset( &elem_by_elem_buffer );

            if( ( success ) && ( generate_elem_by_elem_buffer ) )
            {
                uint64_t const elem_by_elem_blocks = num_of_turns *
                    num_of_beam_elements * num_of_particle_blocks;

                st_block_size_t const elem_by_elem_data_capacity =
                    st_Blocks_predict_data_capacity_for_num_blocks(
                        &elem_by_elem_buffer, elem_by_elem_blocks ) +
                    st_Particles_predict_blocks_data_capacity(
                        &elem_by_elem_buffer, elem_by_elem_blocks,
                        num_of_particles );

                success &= ( 0 == st_Blocks_init( &elem_by_elem_buffer,
                        num_of_beam_elements, elem_by_elem_data_capacity ) );

                for( uint64_t ii = 0 ; ii < elem_by_elem_blocks ; ++ii )
                {
                    st_Particles* e_by_e_particles = st_Blocks_add_particles(
                        &elem_by_elem_buffer, num_of_particles );

                    success &= ( e_by_e_particles != nullptr );
                }

                success &= ( st_Blocks_serialize(
                    &elem_by_elem_buffer ) == 0 );
            }

            /* ------------------------------------------------------------- */

            if( success )
            {
                st_BlockInfo* io_block =
                    st_Blocks_get_block_infos_begin( &elem_by_elem_buffer );

                st_block_size_t const io_blocks_per_turn =
                    num_of_beam_elements * num_of_particle_blocks;

                for( uint64_t ii = 0 ; ii < num_of_turns ; ++ii )
                {
                    success &= ( 0 == st_Track_beam_elements( particles, 0,
                        num_of_particles, &beam_elements, io_block ) );

                    if( io_block != 0 )
                    {
                        io_block = io_block + io_blocks_per_turn;
                    }
                }
            }

            /* ------------------------------------------------------------- */

            if( success )
            {
                st_Blocks* ptr_elem_by_elem_buffer =
                    ( generate_elem_by_elem_buffer )
                        ? &elem_by_elem_buffer : nullptr;



                success = st_Tracks_store_testdata_to_binary_file(
                    num_of_turns, &initial_particles_buffer,
                    &result_particles_buffer, &beam_elements,
                    ptr_elem_by_elem_buffer,
                    path_to_bin_testdata_file.c_str() );
            }
        }

        return success;
    }

    bool create_testdata_drift_exact(
        uint64_t const num_of_turns,
        uint64_t const num_of_particles,
        uint64_t const num_of_beam_elements,
        bool const generate_elem_by_elem_buffer,
        std::string const& path_to_bin_testdata_file,
        uint64_t const prng_seed )
    {
        bool success = false;

        if( ( num_of_turns > uint64_t{ 0 } ) &&
            ( num_of_particles > uint64_t{ 0 } ) &&
            ( num_of_beam_elements > uint64_t{ 0 } ) &&
            ( !path_to_bin_testdata_file.empty() ) )
        {
            st_Random_init_genrand64( prng_seed );

            /* ------------------------------------------------------------ */

            uint64_t const num_of_particle_blocks = uint64_t{ 1 };

            st_Blocks initial_particles_buffer;
            st_Blocks_preset( &initial_particles_buffer );

            st_Blocks result_particles_buffer;
            st_Blocks_preset( &result_particles_buffer );

            st_block_size_t const initial_particles_data_capacity =
                st_Blocks_predict_data_capacity_for_num_blocks(
                    &initial_particles_buffer, num_of_particle_blocks ) +
                st_Particles_predict_blocks_data_capacity(
                    &initial_particles_buffer,
                        num_of_particle_blocks, num_of_particles );

            success = ( 0 == st_Blocks_init( &initial_particles_buffer,
                num_of_particle_blocks, initial_particles_data_capacity ) );

            st_Particles* initial_particles = st_Blocks_add_particles(
                &initial_particles_buffer, num_of_particles );

            success &= ( initial_particles != nullptr );
            st_Particles_random_init( initial_particles );

            success &= ( 0 == st_Blocks_init( &result_particles_buffer,
                num_of_particle_blocks, initial_particles_data_capacity ) );

            st_Particles* particles = st_Blocks_add_particles(
                &result_particles_buffer, num_of_particles );

            success &= ( particles != nullptr );

            st_Particles_copy_all_unchecked( particles, initial_particles );

            success &= ( 0 == st_Blocks_serialize(
                &initial_particles_buffer ) );

            success &= ( 0 == st_Blocks_serialize(
                &result_particles_buffer ) );

            /* ------------------------------------------------------------- */

            double const MIN_DRIFT_LENGTH = double{ 0.1 };
            double const MAX_DRIFT_LENGTH = double{ 1.0 };
            double const DRIFT_LENGTH_RANGE =
                MAX_DRIFT_LENGTH - MIN_DRIFT_LENGTH;

            st_Blocks beam_elements;
            st_Blocks_preset( &beam_elements );

            st_block_size_t const beam_elements_data_capacity =
                st_Blocks_predict_data_capacity_for_num_blocks(
                    &beam_elements, num_of_beam_elements ) +
                st_DriftExact_predict_blocks_data_capacity(
                    &beam_elements, num_of_beam_elements );

            success &= ( 0 == st_Blocks_init( &beam_elements,
                        num_of_beam_elements, beam_elements_data_capacity ) );

            for( st_block_size_t ii = 0 ; ii < num_of_beam_elements ; ++ii )
            {
                double const drift_length = MIN_DRIFT_LENGTH +
                    DRIFT_LENGTH_RANGE * st_Random_genrand64_real1();

                st_DriftExact* ptr_drift = st_Blocks_add_drift_exact(
                    &beam_elements, drift_length );

                success &= ( ptr_drift != nullptr );
            }

            success &= ( st_Blocks_serialize( &beam_elements ) == 0 );

            /* ------------------------------------------------------------- */

            st_Blocks elem_by_elem_buffer;
            st_Blocks_preset( &elem_by_elem_buffer );

            if( ( success ) && ( generate_elem_by_elem_buffer ) )
            {
                uint64_t const elem_by_elem_blocks = num_of_turns *
                    num_of_beam_elements * num_of_particle_blocks;

                st_block_size_t const elem_by_elem_data_capacity =
                    st_Blocks_predict_data_capacity_for_num_blocks(
                        &elem_by_elem_buffer, elem_by_elem_blocks ) +
                    st_Particles_predict_blocks_data_capacity(
                        &elem_by_elem_buffer, elem_by_elem_blocks,
                        num_of_particles );

                success &= ( 0 == st_Blocks_init( &elem_by_elem_buffer,
                        num_of_beam_elements, elem_by_elem_data_capacity ) );

                for( uint64_t ii = 0 ; ii < elem_by_elem_blocks ; ++ii )
                {
                    st_Particles* e_by_e_particles = st_Blocks_add_particles(
                        &elem_by_elem_buffer, num_of_particles );

                    success &= ( e_by_e_particles != nullptr );
                }

                success &= ( st_Blocks_serialize(
                    &elem_by_elem_buffer ) == 0 );
            }

            /* ------------------------------------------------------------- */

            if( success )
            {
                st_BlockInfo* io_block =
                    st_Blocks_get_block_infos_begin( &elem_by_elem_buffer );

                st_block_size_t const io_blocks_per_turn =
                    num_of_beam_elements * num_of_particle_blocks;

                for( uint64_t ii = 0 ; ii < num_of_turns ; ++ii )
                {
                    success &= ( 0 == st_Track_beam_elements( particles, 0,
                        num_of_particles, &beam_elements, io_block ) );

                    if( io_block != 0 )
                    {
                        io_block = io_block + io_blocks_per_turn;
                    }
                }
            }

            /* ------------------------------------------------------------- */

            if( success )
            {
                st_Blocks* ptr_elem_by_elem_buffer =
                    ( generate_elem_by_elem_buffer )
                        ? &elem_by_elem_buffer : nullptr;



                success = st_Tracks_store_testdata_to_binary_file(
                    num_of_turns, &initial_particles_buffer,
                    &result_particles_buffer, &beam_elements,
                    ptr_elem_by_elem_buffer,
                    path_to_bin_testdata_file.c_str() );
            }
        }

        return success;
    }

    bool create_testdata_multipole(
        uint64_t const num_of_turns,
        uint64_t const num_of_particles,
        uint64_t const num_of_beam_elements,
        bool const generate_elem_by_elem_buffer,
        std::string const& path_to_bin_testdata_file,
        uint64_t const prng_seed )
    {
        bool success = false;

        SIXTRL_STATIC SIXTRL_INT64_T const MAX_ORDER   = 7u;

        if( ( num_of_turns > uint64_t{ 0 } ) &&
            ( num_of_particles > uint64_t{ 0 } ) &&
            ( num_of_beam_elements > uint64_t{ 0 } ) &&
            ( !path_to_bin_testdata_file.empty() ) )
        {
            st_Random_init_genrand64( prng_seed );

            /* ------------------------------------------------------------ */

            uint64_t const num_of_particle_blocks = uint64_t{ 1 };

            st_Blocks initial_particles_buffer;
            st_Blocks_preset( &initial_particles_buffer );

            st_Blocks result_particles_buffer;
            st_Blocks_preset( &result_particles_buffer );

            st_block_size_t const initial_particles_data_capacity =
                st_Blocks_predict_data_capacity_for_num_blocks(
                    &initial_particles_buffer, num_of_particle_blocks ) +
                st_Particles_predict_blocks_data_capacity(
                    &initial_particles_buffer,
                        num_of_particle_blocks, num_of_particles );

            success = ( 0 == st_Blocks_init( &initial_particles_buffer,
                num_of_particle_blocks, initial_particles_data_capacity ) );

            st_Particles* initial_particles = st_Blocks_add_particles(
                &initial_particles_buffer, num_of_particles );

            success &= ( initial_particles != nullptr );
            st_Particles_random_init( initial_particles );

            success &= ( 0 == st_Blocks_init( &result_particles_buffer,
                num_of_particle_blocks, initial_particles_data_capacity ) );

            st_Particles* particles = st_Blocks_add_particles(
                &result_particles_buffer, num_of_particles );

            success &= ( particles != nullptr );

            st_Particles_copy_all_unchecked( particles, initial_particles );

            success &= ( 0 == st_Blocks_serialize(
                &initial_particles_buffer ) );

            success &= ( 0 == st_Blocks_serialize(
                &result_particles_buffer ) );

            /* ------------------------------------------------------------- */

            double const MIN_MULTIPOLE_LENGTH = double{ 0.1 };
            double const MAX_MULTIPOLE_LENGTH = double{ 1.0 };
            double const MULTIPOLE_LENGTH_RANGE =
                MAX_MULTIPOLE_LENGTH - MIN_MULTIPOLE_LENGTH;

            double const MIN_HXYL = double{ 0.05L };
            double const MAX_HXYL = double{ 1.00L };
            double const DELTA_HXYL = MAX_HXYL - MIN_HXYL;

            double const MIN_BAL_VALUE = double{ -1.0 };
            double const MAX_BAL_VALUE = double{ +1.0 };
            double const DELTA_BAL_VALUE_RANGE = MAX_BAL_VALUE - MIN_BAL_VALUE;

            st_Blocks beam_elements;
            st_Blocks_preset( &beam_elements );

            int64_t const MIN_ORDER   = 1u;
            int64_t const ORDER_RANGE = MAX_ORDER - MIN_ORDER;

            std::vector< double > bal_values( 255, 0.0 );

            st_block_size_t const beam_elements_data_capacity =
                st_Blocks_predict_data_capacity_for_num_blocks(
                    &beam_elements, num_of_beam_elements ) +
                st_MultiPole_predict_blocks_data_capacity(
                    &beam_elements, num_of_beam_elements, MAX_ORDER );

            success &= ( 0 == st_Blocks_init( &beam_elements,
                        num_of_beam_elements, beam_elements_data_capacity ) );

            for( st_block_size_t ii = 0 ; ii < num_of_beam_elements ; ++ii )
            {
                double const mp_length = MIN_MULTIPOLE_LENGTH +
                    MULTIPOLE_LENGTH_RANGE * st_Random_genrand64_real1();

                double const mp_hxl    =
                    MIN_HXYL + DELTA_HXYL * st_Random_genrand64_real1();

                double const mp_hyl    =
                    MIN_HXYL + DELTA_HXYL * st_Random_genrand64_real1();

                int64_t const mp_order = MIN_ORDER + static_cast< int64_t >(
                    ORDER_RANGE * st_Random_genrand64_real1() );

                bal_values.clear();

                int64_t const nn = 2 * mp_order + 2;

                for( int64_t jj = 0 ; jj < nn ; ++jj )
                {
                    bal_values.push_back( MIN_BAL_VALUE +
                        DELTA_BAL_VALUE_RANGE * st_Random_genrand64_real1() );
                }

                st_MultiPole* ptr_multipole = st_Blocks_add_multipole(
                    &beam_elements, mp_length, mp_hxl, mp_hyl, mp_order,
                    bal_values.data() );

                success &= ( ptr_multipole != nullptr );
            }

            success &= ( st_Blocks_serialize( &beam_elements ) == 0 );

            /* ------------------------------------------------------------- */

            st_Blocks elem_by_elem_buffer;
            st_Blocks_preset( &elem_by_elem_buffer );

            if( ( success ) && ( generate_elem_by_elem_buffer ) )
            {
                uint64_t const elem_by_elem_blocks = num_of_turns *
                    num_of_beam_elements * num_of_particle_blocks;

                st_block_size_t const elem_by_elem_data_capacity =
                    st_Blocks_predict_data_capacity_for_num_blocks(
                        &elem_by_elem_buffer, elem_by_elem_blocks ) +
                    st_Particles_predict_blocks_data_capacity(
                        &elem_by_elem_buffer, elem_by_elem_blocks,
                        num_of_particles );

                success &= ( 0 == st_Blocks_init( &elem_by_elem_buffer,
                        num_of_beam_elements, elem_by_elem_data_capacity ) );

                for( uint64_t ii = 0 ; ii < elem_by_elem_blocks ; ++ii )
                {
                    st_Particles* e_by_e_particles = st_Blocks_add_particles(
                        &elem_by_elem_buffer, num_of_particles );

                    success &= ( e_by_e_particles != nullptr );
                }

                success &= ( st_Blocks_serialize(
                    &elem_by_elem_buffer ) == 0 );
            }

            /* ------------------------------------------------------------- */

            if( success )
            {
                st_BlockInfo* io_block =
                    st_Blocks_get_block_infos_begin( &elem_by_elem_buffer );

                st_block_size_t const io_blocks_per_turn =
                    num_of_beam_elements * num_of_particle_blocks;

                for( uint64_t ii = 0 ; ii < num_of_turns ; ++ii )
                {
                    success &= ( 0 == st_Track_beam_elements( particles, 0,
                        num_of_particles, &beam_elements, io_block ) );

                    if( io_block != 0 )
                    {
                        io_block = io_block + io_blocks_per_turn;
                    }
                }
            }

            /* ------------------------------------------------------------- */

            if( success )
            {
                st_Blocks* ptr_elem_by_elem_buffer =
                    ( generate_elem_by_elem_buffer )
                        ? &elem_by_elem_buffer : nullptr;



                success = st_Tracks_store_testdata_to_binary_file(
                    num_of_turns, &initial_particles_buffer,
                    &result_particles_buffer, &beam_elements,
                    ptr_elem_by_elem_buffer,
                    path_to_bin_testdata_file.c_str() );
            }
        }

        return success;
    }

    bool create_testdata_beam_beam(
        uint64_t const num_of_turns,
        uint64_t const num_of_particles,
        uint64_t const num_of_beam_elements,
        bool const generate_elem_by_elem_buffer,
        std::string const& path_to_bin_testdata_file,
        uint64_t const prng_seed )
    {
        bool success = false;

        if( ( num_of_turns > uint64_t{ 0 } ) &&
            ( num_of_particles > uint64_t{ 0 } ) &&
            ( num_of_beam_elements > uint64_t{ 0 } ) &&
            ( !path_to_bin_testdata_file.empty() ) )
        {
            st_Random_init_genrand64( prng_seed );

            /* ------------------------------------------------------------ */

            uint64_t const num_of_particle_blocks = uint64_t{ 1 };

            st_Blocks initial_particles_buffer;
            st_Blocks_preset( &initial_particles_buffer );

            st_Blocks result_particles_buffer;
            st_Blocks_preset( &result_particles_buffer );

            st_block_size_t const initial_particles_data_capacity =
                st_Blocks_predict_data_capacity_for_num_blocks(
                    &initial_particles_buffer, num_of_particle_blocks ) +
                st_Particles_predict_blocks_data_capacity(
                    &initial_particles_buffer,
                        num_of_particle_blocks, num_of_particles );

            success = ( 0 == st_Blocks_init( &initial_particles_buffer,
                num_of_particle_blocks, initial_particles_data_capacity ) );

            st_Particles* initial_particles = st_Blocks_add_particles(
                &initial_particles_buffer, num_of_particles );

            success &= ( initial_particles != nullptr );
            st_Particles_random_init( initial_particles );

            success &= ( 0 == st_Blocks_init( &result_particles_buffer,
                num_of_particle_blocks, initial_particles_data_capacity ) );

            st_Particles* particles = st_Blocks_add_particles(
                &result_particles_buffer, num_of_particles );

            success &= ( particles != nullptr );

            st_Particles_copy_all_unchecked( particles, initial_particles );

            success &= ( 0 == st_Blocks_serialize(
                &initial_particles_buffer ) );

            success &= ( 0 == st_Blocks_serialize(
                &result_particles_buffer ) );

            /* ------------------------------------------------------------- */

            st_block_num_elements_t const MIN_NUM_SLICES = 2u;
            st_block_num_elements_t const MAX_NUM_SLICES = 64u;
            st_block_num_elements_t const
                NUM_SLICES_RANGE = MAX_NUM_SLICES - MIN_NUM_SLICES;

            double const MIN_X          = 0.01;
            double const MAX_X          = 1.00;
            double const DELTA_X        = MAX_X - MIN_X;

            double const MIN_Y          = 0.01;
            double const MAX_Y          = 1.00;
            double const DELTA_Y        = MAX_Y - MIN_Y;

            double const MIN_SIGMA      = 0.01;
            double const MAX_SIGMA      = 1.00;
            double const DELTA_SIGMA    = MAX_SIGMA - MIN_SIGMA;

            double const MIN_PHI        = ( -10.0 * M_PI ) / 180.0;
            double const MAX_PHI        = ( -10.0 * M_PI ) / 180.0;
            double const DELTA_PHI      = MAX_PHI - MIN_PHI;

            double const MIN_ALPHA      = ( -45.0 * M_PI ) / 180.0;
            double const MAX_ALPHA      = ( +45.0 * M_PI ) / 180.0;
            double const DELTA_ALPHA    = MAX_ALPHA - MIN_ALPHA;

            double const MIN_BB_SIGMA   =   1.0;
            double const MAX_BB_SIGMA   =  20.0;
            double const DELTA_BB_SIGMA = MAX_BB_SIGMA - MIN_BB_SIGMA;

            std::vector< double > n_part_per_slice( MAX_NUM_SLICES, 0.0 );
            std::vector< double > x_slices_star( MAX_NUM_SLICES, 0.0 );
            std::vector< double > y_slices_star( MAX_NUM_SLICES, 0.0 );
            std::vector< double > sigma_slices_star( MAX_NUM_SLICES, 0.0 );

            st_Blocks beam_elements;
            st_Blocks_preset( &beam_elements );

            st_block_size_t const beam_elements_data_capacity =
                st_Blocks_predict_data_capacity_for_num_blocks(
                    &beam_elements, num_of_beam_elements ) +
                st_BeamBeam_predict_blocks_data_capacity(
                    &beam_elements, num_of_beam_elements, MAX_NUM_SLICES );

            success &= ( 0 == st_Blocks_init( &beam_elements,
                        num_of_beam_elements, beam_elements_data_capacity ) );

            for( st_block_size_t ii = 0 ; ii < num_of_beam_elements ; ++ii )
            {
                double const alpha =
                    MIN_ALPHA + DELTA_ALPHA * st_Random_genrand64_real1();

                double const phi   =
                    MIN_PHI + DELTA_PHI * st_Random_genrand64_real1();

                st_BeamBeamBoostData boost_data;
                st_BeamBeamBoostData_preset( &boost_data );

                st_BeamBeamBoostData_set_sphi(   &boost_data, sin( phi )   );
                st_BeamBeamBoostData_set_cphi(   &boost_data, cos( phi )   );
                st_BeamBeamBoostData_set_tphi(   &boost_data, tan( phi )   );
                st_BeamBeamBoostData_set_salpha( &boost_data, sin( alpha ) );
                st_BeamBeamBoostData_set_calpha( &boost_data, cos( alpha ) );

                st_BeamBeamSigmas sigmas;
                st_BeamBeamSigmas_preset( &sigmas );

                double const sigma_11 =
                    MIN_BB_SIGMA + DELTA_BB_SIGMA * st_Random_genrand64_real1();

                double const sigma_12 =
                    MIN_BB_SIGMA + DELTA_BB_SIGMA * st_Random_genrand64_real1();

                double const sigma_33 =
                    MIN_BB_SIGMA + DELTA_BB_SIGMA * st_Random_genrand64_real1();

                double const sigma_34 = ( sigma_33 / sigma_11 ) * sigma_12;

                double const sigma_22 =
                    MIN_BB_SIGMA + DELTA_BB_SIGMA * st_Random_genrand64_real1();

                double const sigma_44 =
                    MIN_BB_SIGMA + DELTA_BB_SIGMA * st_Random_genrand64_real1();



                double const sigma_13 = sqrt( 0.5 * sigma_11 * sigma_33 );
                double const sigma_24 = sqrt( 0.5 * sigma_22 * sigma_44 );
                double const sigma_14 = sqrt( 0.5 * sigma_11 * sigma_44 );

                double const sigma_23 =
                    0.25 * ( sigma_12 * sigma_33 + sigma_11 * sigma_34 -
                        sigma_13 * sigma_14 ) / sigma_13;

                st_BeamBeamSigmas_set_sigma11( &sigmas, sigma_11 );
                st_BeamBeamSigmas_set_sigma12( &sigmas, sigma_12 );
                st_BeamBeamSigmas_set_sigma13( &sigmas, sigma_13 );
                st_BeamBeamSigmas_set_sigma14( &sigmas, sigma_14 );

                st_BeamBeamSigmas_set_sigma22( &sigmas, sigma_22 );
                st_BeamBeamSigmas_set_sigma23( &sigmas, sigma_23 );
                st_BeamBeamSigmas_set_sigma24( &sigmas, sigma_24 );

                st_BeamBeamSigmas_set_sigma33( &sigmas, sigma_33 );
                st_BeamBeamSigmas_set_sigma34( &sigmas, sigma_34 );

                st_BeamBeamSigmas_set_sigma44( &sigmas, sigma_44 );

                st_block_num_elements_t const num_slices =
                    static_cast< st_block_num_elements_t >( MIN_NUM_SLICES +
                        NUM_SLICES_RANGE * st_Random_genrand64_real1() );

                double const n_part  = ( 1.0 ) / static_cast< double >( num_slices );

                double const x_begin = MIN_X;
                double const x_end   = x_begin + DELTA_X * st_Random_genrand64_real1();
                double const dx      = ( x_end - x_begin ) / ( num_slices - 1.0 );

                double const y_begin = MIN_Y;
                double const y_end   = y_begin + DELTA_Y * st_Random_genrand64_real1();
                double const dy      = ( y_end - y_begin ) / ( num_slices - 1.0 );

                double const sigma_begin = MIN_SIGMA;
                double const sigma_end   = sigma_begin + DELTA_SIGMA * st_Random_genrand64_real1();
                double const dsigma      = ( sigma_end - sigma_begin ) / ( num_slices - 1.0 );

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

                st_BeamBeam* beam_beam = st_Blocks_add_beam_beam(
                    &beam_elements, &boost_data, &sigmas, num_slices,
                    n_part_per_slice.data(), x_slices_star.data(),
                    y_slices_star.data(), sigma_slices_star.data(),
                    n_part, 0.0, 1.0 );

                success &= ( beam_beam != nullptr );
            }

            success &= ( st_Blocks_serialize( &beam_elements ) == 0 );

            /* ------------------------------------------------------------- */

            st_Blocks elem_by_elem_buffer;
            st_Blocks_preset( &elem_by_elem_buffer );

            if( ( success ) && ( generate_elem_by_elem_buffer ) )
            {
                uint64_t const elem_by_elem_blocks = num_of_turns *
                    num_of_beam_elements * num_of_particle_blocks;

                st_block_size_t const elem_by_elem_data_capacity =
                    st_Blocks_predict_data_capacity_for_num_blocks(
                        &elem_by_elem_buffer, elem_by_elem_blocks ) +
                    st_Particles_predict_blocks_data_capacity(
                        &elem_by_elem_buffer, elem_by_elem_blocks,
                        num_of_particles );

                success &= ( 0 == st_Blocks_init( &elem_by_elem_buffer,
                        num_of_beam_elements, elem_by_elem_data_capacity ) );

                for( uint64_t ii = 0 ; ii < elem_by_elem_blocks ; ++ii )
                {
                    st_Particles* e_by_e_particles = st_Blocks_add_particles(
                        &elem_by_elem_buffer, num_of_particles );

                    success &= ( e_by_e_particles != nullptr );
                }

                success &= ( st_Blocks_serialize(
                    &elem_by_elem_buffer ) == 0 );
            }

            /* ------------------------------------------------------------- */

            if( success )
            {
                st_BlockInfo* io_block =
                    st_Blocks_get_block_infos_begin( &elem_by_elem_buffer );

                st_block_size_t const io_blocks_per_turn =
                    num_of_beam_elements * num_of_particle_blocks;

                for( uint64_t ii = 0 ; ii < num_of_turns ; ++ii )
                {
                    success &= ( 0 == st_Track_beam_elements( particles, 0,
                        num_of_particles, &beam_elements, io_block ) );

                    if( io_block != 0 )
                    {
                        io_block = io_block + io_blocks_per_turn;
                    }
                }
            }

            /* ------------------------------------------------------------- */

            if( success )
            {
                st_Blocks* ptr_elem_by_elem_buffer =
                    ( generate_elem_by_elem_buffer )
                        ? &elem_by_elem_buffer : nullptr;

                success = st_Tracks_store_testdata_to_binary_file(
                    num_of_turns, &initial_particles_buffer,
                    &result_particles_buffer, &beam_elements,
                    ptr_elem_by_elem_buffer,
                    path_to_bin_testdata_file.c_str() );
            }
        }

        return success;
    }

    bool create_testdata_cavity(
        uint64_t const num_of_turns, uint64_t const num_of_particles,
        uint64_t const num_of_beam_elements,
        bool const generate_elem_by_elem_buffer,
        std::string const& path_to_bin_testdata_file,
        uint64_t const prng_seed )
    {
        bool success = false;

        if( ( num_of_turns > uint64_t{ 0 } ) &&
            ( num_of_particles > uint64_t{ 0 } ) &&
            ( num_of_beam_elements > uint64_t{ 0 } ) &&
            ( !path_to_bin_testdata_file.empty() ) )
        {
            st_Random_init_genrand64( prng_seed );

            /* ------------------------------------------------------------ */

            uint64_t const num_of_particle_blocks = uint64_t{ 1 };

            st_Blocks initial_particles_buffer;
            st_Blocks_preset( &initial_particles_buffer );

            st_Blocks result_particles_buffer;
            st_Blocks_preset( &result_particles_buffer );

            st_block_size_t const initial_particles_data_capacity =
                st_Blocks_predict_data_capacity_for_num_blocks(
                    &initial_particles_buffer, num_of_particle_blocks ) +
                st_Particles_predict_blocks_data_capacity(
                    &initial_particles_buffer,
                        num_of_particle_blocks, num_of_particles );

            success = ( 0 == st_Blocks_init( &initial_particles_buffer,
                num_of_particle_blocks, initial_particles_data_capacity ) );

            st_Particles* initial_particles = st_Blocks_add_particles(
                &initial_particles_buffer, num_of_particles );

            success &= ( initial_particles != nullptr );
            st_Particles_random_init( initial_particles );

            success &= ( 0 == st_Blocks_init( &result_particles_buffer,
                num_of_particle_blocks, initial_particles_data_capacity ) );

            st_Particles* particles = st_Blocks_add_particles(
                &result_particles_buffer, num_of_particles );

            success &= ( particles != nullptr );

            st_Particles_copy_all_unchecked( particles, initial_particles );

            success &= ( 0 == st_Blocks_serialize(
                &initial_particles_buffer ) );

            success &= ( 0 == st_Blocks_serialize(
                &result_particles_buffer ) );

            /* ------------------------------------------------------------- */

            double const MIN_VOLTAGE     =  0.0;
            double const MAX_VOLTAGE     =  1e5;
            double const DELTA_VOLTAGE   = MAX_VOLTAGE - MIN_VOLTAGE;

            double const MIN_FREQUENCY   = 0.0;
            double const MAX_FREQUENCY   = 400.0;
            double const DELTA_FREQUENCY = MAX_FREQUENCY - MIN_FREQUENCY;

            double const MIN_LAG         = 0.0;
            double const MAX_LAG         = 0.1;
            double const DELTA_LAG       = MAX_LAG - MIN_LAG;

            st_Blocks beam_elements;
            st_Blocks_preset( &beam_elements );

            st_block_size_t const beam_elements_data_capacity =
                st_Blocks_predict_data_capacity_for_num_blocks(
                    &beam_elements, num_of_beam_elements ) +
                st_Cavity_predict_blocks_data_capacity(
                    &beam_elements, num_of_beam_elements );

            success &= ( 0 == st_Blocks_init( &beam_elements,
                        num_of_beam_elements, beam_elements_data_capacity ) );

            for( st_block_size_t ii = 0 ; ii < num_of_beam_elements ; ++ii )
            {
                double const voltage =
                    MIN_VOLTAGE + DELTA_VOLTAGE * st_Random_genrand64_real1();

                double const frequency =
                    MIN_FREQUENCY + DELTA_FREQUENCY *
                        st_Random_genrand64_real1();

                double const lag =
                    MIN_LAG + DELTA_LAG * st_Random_genrand64_real1();

                st_Cavity* ptr_cavity = st_Blocks_add_cavity(
                    &beam_elements, voltage, frequency, lag );

                success &= ( ptr_cavity != nullptr );
            }

            success &= ( st_Blocks_serialize( &beam_elements ) == 0 );

            /* ------------------------------------------------------------- */

            st_Blocks elem_by_elem_buffer;
            st_Blocks_preset( &elem_by_elem_buffer );

            if( ( success ) && ( generate_elem_by_elem_buffer ) )
            {
                uint64_t const elem_by_elem_blocks = num_of_turns *
                    num_of_beam_elements * num_of_particle_blocks;

                st_block_size_t const elem_by_elem_data_capacity =
                    st_Blocks_predict_data_capacity_for_num_blocks(
                        &elem_by_elem_buffer, elem_by_elem_blocks ) +
                    st_Particles_predict_blocks_data_capacity(
                        &elem_by_elem_buffer, elem_by_elem_blocks,
                        num_of_particles );

                success &= ( 0 == st_Blocks_init( &elem_by_elem_buffer,
                        num_of_beam_elements, elem_by_elem_data_capacity ) );

                for( uint64_t ii = 0 ; ii < elem_by_elem_blocks ; ++ii )
                {
                    st_Particles* e_by_e_particles = st_Blocks_add_particles(
                        &elem_by_elem_buffer, num_of_particles );

                    success &= ( e_by_e_particles != nullptr );
                }

                success &= ( st_Blocks_serialize(
                    &elem_by_elem_buffer ) == 0 );
            }

            /* ------------------------------------------------------------- */

            if( success )
            {
                st_BlockInfo* io_block =
                    st_Blocks_get_block_infos_begin( &elem_by_elem_buffer );

                st_block_size_t const io_blocks_per_turn =
                    num_of_beam_elements * num_of_particle_blocks;

                for( uint64_t ii = 0 ; ii < num_of_turns ; ++ii )
                {
                    success &= ( 0 == st_Track_beam_elements( particles, 0,
                        num_of_particles, &beam_elements, io_block ) );

                    if( io_block != 0 )
                    {
                        io_block = io_block + io_blocks_per_turn;
                    }
                }
            }

            /* ------------------------------------------------------------- */

            if( success )
            {
                st_Blocks* ptr_elem_by_elem_buffer =
                    ( generate_elem_by_elem_buffer )
                        ? &elem_by_elem_buffer : nullptr;

                success = st_Tracks_store_testdata_to_binary_file(
                    num_of_turns, &initial_particles_buffer,
                    &result_particles_buffer, &beam_elements,
                    ptr_elem_by_elem_buffer,
                    path_to_bin_testdata_file.c_str() );
            }
        }

        return success;
    }

    bool create_testdata_align(
        uint64_t const num_of_turns,
        uint64_t const num_of_particles,
        uint64_t const num_of_beam_elements,
        bool const generate_elem_by_elem_buffer,
        std::string const& path_to_bin_testdata_file,
        uint64_t const prng_seed )
    {
        bool success = false;

        if( ( num_of_turns > uint64_t{ 0 } ) &&
            ( num_of_particles > uint64_t{ 0 } ) &&
            ( num_of_beam_elements > uint64_t{ 0 } ) &&
            ( !path_to_bin_testdata_file.empty() ) )
        {
            st_Random_init_genrand64( prng_seed );

            /* ------------------------------------------------------------ */

            uint64_t const num_of_particle_blocks = uint64_t{ 1 };

            st_Blocks initial_particles_buffer;
            st_Blocks_preset( &initial_particles_buffer );

            st_Blocks result_particles_buffer;
            st_Blocks_preset( &result_particles_buffer );

            st_block_size_t const initial_particles_data_capacity =
                st_Blocks_predict_data_capacity_for_num_blocks(
                    &initial_particles_buffer, num_of_particle_blocks ) +
                st_Particles_predict_blocks_data_capacity(
                    &initial_particles_buffer,
                        num_of_particle_blocks, num_of_particles );

            success = ( 0 == st_Blocks_init( &initial_particles_buffer,
                num_of_particle_blocks, initial_particles_data_capacity ) );

            st_Particles* initial_particles = st_Blocks_add_particles(
                &initial_particles_buffer, num_of_particles );

            success &= ( initial_particles != nullptr );
            st_Particles_random_init( initial_particles );

            success &= ( 0 == st_Blocks_init( &result_particles_buffer,
                num_of_particle_blocks, initial_particles_data_capacity ) );

            st_Particles* particles = st_Blocks_add_particles(
                &result_particles_buffer, num_of_particles );

            success &= ( particles != nullptr );

            st_Particles_copy_all_unchecked( particles, initial_particles );

            success &= ( 0 == st_Blocks_serialize(
                &initial_particles_buffer ) );

            success &= ( 0 == st_Blocks_serialize(
                &result_particles_buffer ) );

            /* ------------------------------------------------------------- */

            double const MIN_TILT       =  0.0;
            double const MAX_TILT       =  1.0;
            double const DELTA_TILT     = MAX_TILT - MIN_TILT;

            double const MIN_Z          = ( -5.0 * M_PI ) / 180.0;
            double const MAX_Z          = (  5.0 * M_PI ) / 180.0;
            double const DELTA_Z        = MAX_Z - MIN_Z;

            double const MIN_DX         = -0.5;
            double const MAX_DX         =  0.5;
            double const DELTA_DX       = MAX_DX - MIN_DX;

            double const MIN_DY         = -0.5;
            double const MAX_DY         =  0.5;
            double const DELTA_DY       = MAX_DY - MIN_DY;

            st_Blocks beam_elements;
            st_Blocks_preset( &beam_elements );

            st_block_size_t const beam_elements_data_capacity =
                st_Blocks_predict_data_capacity_for_num_blocks(
                    &beam_elements, num_of_beam_elements ) +
                st_Align_predict_blocks_data_capacity(
                    &beam_elements, num_of_beam_elements );

            success &= ( 0 == st_Blocks_init( &beam_elements,
                        num_of_beam_elements, beam_elements_data_capacity ) );

            for( st_block_size_t ii = 0 ; ii < num_of_beam_elements ; ++ii )
            {
                double const z = MIN_Z + DELTA_Z * st_Random_genrand64_real1();
                double const tilt =
                    MIN_TILT + DELTA_TILT * st_Random_genrand64_real1();

                double const dx =
                    MIN_DX + DELTA_DX * st_Random_genrand64_real1();

                double const dy =
                    MIN_DY + DELTA_DY * st_Random_genrand64_real1();

                st_Align* ptr_align = st_Blocks_add_align(
                    &beam_elements, tilt, cos( z ), sin( z ), dx, dy );

                success &= ( ptr_align != nullptr );
            }

            success &= ( st_Blocks_serialize( &beam_elements ) == 0 );

            /* ------------------------------------------------------------- */

            st_Blocks elem_by_elem_buffer;
            st_Blocks_preset( &elem_by_elem_buffer );

            if( ( success ) && ( generate_elem_by_elem_buffer ) )
            {
                uint64_t const elem_by_elem_blocks = num_of_turns *
                    num_of_beam_elements * num_of_particle_blocks;

                st_block_size_t const elem_by_elem_data_capacity =
                    st_Blocks_predict_data_capacity_for_num_blocks(
                        &elem_by_elem_buffer, elem_by_elem_blocks ) +
                    st_Particles_predict_blocks_data_capacity(
                        &elem_by_elem_buffer, elem_by_elem_blocks,
                        num_of_particles );

                success &= ( 0 == st_Blocks_init( &elem_by_elem_buffer,
                        num_of_beam_elements, elem_by_elem_data_capacity ) );

                for( uint64_t ii = 0 ; ii < elem_by_elem_blocks ; ++ii )
                {
                    st_Particles* e_by_e_particles = st_Blocks_add_particles(
                        &elem_by_elem_buffer, num_of_particles );

                    success &= ( e_by_e_particles != nullptr );
                }

                success &= ( st_Blocks_serialize(
                    &elem_by_elem_buffer ) == 0 );
            }

            /* ------------------------------------------------------------- */

            if( success )
            {
                st_BlockInfo* io_block =
                    st_Blocks_get_block_infos_begin( &elem_by_elem_buffer );

                st_block_size_t const io_blocks_per_turn =
                    num_of_beam_elements * num_of_particle_blocks;

                for( uint64_t ii = 0 ; ii < num_of_turns ; ++ii )
                {
                    success &= ( 0 == st_Track_beam_elements( particles, 0,
                        num_of_particles, &beam_elements, io_block ) );

                    if( io_block != 0 )
                    {
                        io_block = io_block + io_blocks_per_turn;
                    }
                }
            }

            /* ------------------------------------------------------------- */

            if( success )
            {
                st_Blocks* ptr_elem_by_elem_buffer =
                    ( generate_elem_by_elem_buffer )
                        ? &elem_by_elem_buffer : nullptr;

                success = st_Tracks_store_testdata_to_binary_file(
                    num_of_turns, &initial_particles_buffer,
                    &result_particles_buffer, &beam_elements,
                    ptr_elem_by_elem_buffer,
                    path_to_bin_testdata_file.c_str() );
            }
        }

        return success;
    }
}

/* end: sixtracklib/testdata/generate_testdata_tracking.cpp */
