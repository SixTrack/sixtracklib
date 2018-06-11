#define _USE_MATH_DEFINES 

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>

#if defined( __NAMESPACE )
    #define __SAVED_NAMESPACE __NAMESPACE
    #undef  __NAMESPACE     
#endif /* !defiend( __NAMESPACE ) */

#if !defined( __NAMESPACE )
    #define __NAMESPACE st_    
#endif /* !defiend( __NAMESPACE ) */

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/path.h"

#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/track.h"
#include "sixtracklib/common/impl/track_api.h"

#include "sixtracklib/common/details/random.h"
#include "sixtracklib/common/tests/test_particles_tools.h"
#include "sixtracklib/common/tests/test_track_tools.h"

#include "sixtracklib/testdata/tracking_testfiles.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

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
        std::cout << "FAILURE" << std::endl << std::endl;
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
            
            std::vector< SIXTRL_REAL_T > bal_values( 255, 0.0 );    
            
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
}

/* end: sixtracklib/testdata/generate_testdata_tracking.cpp */
