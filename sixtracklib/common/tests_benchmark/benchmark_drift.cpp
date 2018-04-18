#define _USE_MATH_DEFINES

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

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
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/impl/particles_type.h"
#include "sixtracklib/common/particles_sequence.h"
#include "sixtracklib/common/track.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

int main()
{
    using prng_t            = std::mt19937_64;
    using prng_seed_t       = std::mt19937_64::result_type;
    using elem_type_dist_t  = std::uniform_int_distribution<>;
    using drift_len_dist_t  = std::uniform_real_distribution<>;
    
    /* to keep it reproducible for the moment, seed the rng with knwon number */
    static prng_seed_t const PRNG_SEED = prng_seed_t{ 20180416 }; 
    
    /* Global flag to enable/disable saving of elem_by_elem and turn_by_turn: */
    static bool const ENABLE_ELEM_BY_ELEM = false;
    static bool const ENABLE_TURN_BY_TURN = false;
    
    /* Half a gigabyte of memory for elem_by_elem or turn_by_turn seems about right */
    static SIXTRL_SIZE_T const ELEM_BY_ELEM_MEM_LIMIT = SIXTRL_SIZE_T{ 0x20000000 };
    static SIXTRL_SIZE_T const TURN_BY_TURN_MEM_LIMIT = SIXTRL_SIZE_T{ 0x20000000 };
    
    static SIXTRL_REAL_T const MIN_DRIFT_LENGTH = SIXTRL_REAL_T{ 0.1 };
    static SIXTRL_REAL_T const MAX_DRIFT_LENGTH = SIXTRL_REAL_T{ 1.0 };
    
    /* These values are not supposed to make physical sense (now) .... */
    static SIXTRL_REAL_T const BETA0 = SIXTRL_REAL_T{ 1.0 };
    static SIXTRL_REAL_T const RPP   = SIXTRL_REAL_T{ 0.5 };
    static SIXTRL_REAL_T const RVV   = SIXTRL_REAL_T{ 0.5 };
    static SIXTRL_REAL_T const DELTA = SIXTRL_REAL_T{ 2.0 };
    
    prng_t prng( PRNG_SEED );
        
    std::vector< st_BeamElementType > const 
        ELEMENT_TYPES{ st_ELEMENT_TYPE_DRIFT, st_ELEMENT_TYPE_DRIFT_EXACT };
    
    std::vector< SIXTRL_SIZE_T > const 
        NUM_OF_ELEMENTS{ 1, 10, 100, 1000 }; 
    
    std::vector< SIXTRL_SIZE_T > const 
        NUM_OF_PARTICLES{ 1, 10, 100, 1000, 10000, 100000 };
        
    std::vector< SIXTRL_SIZE_T > const 
        NUM_OF_TURNS{ 1, 1000, 10000 };
        
    std::cout << std::setw( 20 ) << "NELEMS"
              << std::setw( 20 ) << "NPARTS"
              << std::setw( 20 ) << "NTURNS"
              << std::setw( 20 ) << "TURN_BY_TURN"
              << std::setw( 20 ) << "ELEM_BY_ELEM"
              << std::setw( 20 ) << "time [s]"
              << std::endl;
        
    for( SIXTRL_SIZE_T const NELEMS : NUM_OF_ELEMENTS )
    {
        /* ----------------------------------------------------------------- */
        /* setup the blocks of beam elements: */
        
        st_Block* blocks = st_Block_new( NELEMS );
        
        std::size_t const MAX_ELEM_TYPE_INDEX = ( !ELEMENT_TYPES.empty() )
            ? ELEMENT_TYPES.size() - std::size_t{ 1 } : std::size_t{ 0 };
            
        elem_type_dist_t elem_type_dist( std::size_t{ 0 }, MAX_ELEM_TYPE_INDEX );
        drift_len_dist_t drift_len_dist( MIN_DRIFT_LENGTH, MAX_DRIFT_LENGTH );
        
        for( std::size_t ii = 0 ; ii < NELEMS ; ++ii )
        {
            SIXTRL_REAL_T const drift_length = drift_len_dist( prng );
            
            st_BeamElementType const elem_type = 
                ELEMENT_TYPES[ elem_type_dist( prng ) ];
                
            SIXTRL_INT64_T const element_id = st_Block_append_drift(
                blocks, elem_type, drift_length );
            
            SIXTRL_ASSERT( element_id >= 0 );
            ( void )element_id;
        }
        
        SIXTRL_ASSERT( st_Block_get_size( blocks ) == NELEMS );
        
        for( SIXTRL_SIZE_T const NTURNS : NUM_OF_TURNS )
        {
            for( std::size_t const NPARTS : NUM_OF_PARTICLES )
            {
                SIXTRL_SIZE_T particles_chunk_size = 
                    st_PARTICLES_DEFAULT_MEMPOOL_CHUNK_SIZE;
                    
                SIXTRL_SIZE_T particles_alignment  = 
                    st_PARTICLES_DEFAULT_MEMPOOL_ALIGNMENT;
                
                auto part_sequ_del = 
                    []( st_ParticlesSequence* p )
                    { 
                        if( p != nullptr )
                        {
                            st_ParticlesSequence_free( p ); 
                            delete p; 
                        }
                    };
                    
                /* --------------------------------------------------------- */
                /* setup the element by element structure: */
                
                using ptr_elem_by_elem_t = std::unique_ptr< 
                    st_ParticlesSequence, decltype( part_sequ_del ) >;
                
                ptr_elem_by_elem_t ptr_elem_by_elem( nullptr, part_sequ_del );
                                
                if( ( ENABLE_ELEM_BY_ELEM ) && ( ELEM_BY_ELEM_MEM_LIMIT >= 
                        ( sizeof( st_Particles ) * NELEMS * NPARTS ) ) )
                {
                    ptr_elem_by_elem.reset( new st_ParticlesSequence );
                    st_ParticlesSequence_preset( ptr_elem_by_elem.get() );
                    
                    if( !st_ParticlesSequence_init( ptr_elem_by_elem.get(),
                        NELEMS, NPARTS, &particles_chunk_size, 
                            &particles_alignment, true ) )
                    {
                        ptr_elem_by_elem.reset();                        
                    }
                }
                
                /* --------------------------------------------------------- */
                /* setup the turn by turn structure: */
                
                using ptr_turn_by_turn_t = std::unique_ptr< 
                    st_ParticlesSequence, decltype( part_sequ_del ) >;
                
                ptr_turn_by_turn_t ptr_turn_by_turn( nullptr, part_sequ_del );
                                
                if( ( ENABLE_TURN_BY_TURN ) && ( TURN_BY_TURN_MEM_LIMIT >= 
                        ( sizeof( st_Particles ) * NTURNS * NPARTS ) ) )
                {
                    ptr_turn_by_turn.reset( new st_ParticlesSequence );
                    st_ParticlesSequence_preset( ptr_turn_by_turn.get() );
                    
                    if( !st_ParticlesSequence_init( ptr_turn_by_turn.get(),
                        NTURNS, NPARTS, &particles_chunk_size, 
                            &particles_alignment, true ) )
                    {
                        ptr_turn_by_turn.reset();                        
                    }
                }
                
                /* --------------------------------------------------------- */
                /* setup of the particles structure itself: */
                
                st_Particles* particles = st_Particles_new( NPARTS );

                /* -------------------------------------------------------- */
                /* init values for the particles -> take care to avoid 
                 * producing NaN's by getting pathological combinations of 
                 * values in the particle states : */
                
                std::vector< SIXTRL_INT64_T > temp_i64( NPARTS, SIXTRL_INT64_T{ -1 } );
                    
                st_Particles_set_lost_at_element_id( particles, temp_i64.data() );
                st_Particles_set_lost_at_turn( particles, temp_i64.data() );
            
                std::fill( temp_i64.begin(), temp_i64.end(), SIXTRL_INT64_T{ 0 } );
                st_Particles_set_state( particles, temp_i64.data() );
            
                    
                SIXTRL_INT64_T part_id = SIXTRL_INT64_T{ 0 };
                std::generate( temp_i64.begin(), temp_i64.end(), 
                               [&part_id](){ return part_id++; } );
                
                st_Particles_set_particle_id( particles, temp_i64.data() );
                
                /* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */
                
                std::vector< SIXTRL_REAL_T > temp( NPARTS, SIXTRL_REAL_T{ 0 } );
                st_Particles_set_s( particles, temp.data() );                
                st_Particles_set_y( particles, temp.data() );
                
                SIXTRL_REAL_T const COORD_X_STRIDE = SIXTRL_REAL_T{ 0.1 };
                SIXTRL_REAL_T x_coord = SIXTRL_REAL_T{ 0 };
                
                std::generate( temp.begin(), temp.end(), 
                               [&x_coord,COORD_X_STRIDE]()
                               { SIXTRL_REAL_T const x = x_coord + COORD_X_STRIDE; 
                                x_coord = x; return x; } );
                
                st_Particles_set_x( particles, temp.data() );
                
                std::fill( temp.begin(), temp.end(), BETA0 );
                st_Particles_set_sigma( particles, temp.data() );
                
                std::fill( temp.begin(), temp.end(), RPP );
                st_Particles_set_rpp( particles, temp.data() );
                
                std::fill( temp.begin(), temp.end(), RVV );
                st_Particles_set_rvv( particles, temp.data() );
                
                std::fill( temp.begin(), temp.end(), DELTA );
                st_Particles_set_delta( particles, temp.data() );
                
                using px_py_angle_dist_t = 
                    std::uniform_real_distribution< SIXTRL_REAL_T >;
                
                px_py_angle_dist_t px_py_angle_dist( 0.0, 2.0 * M_PI );
                std::vector< SIXTRL_REAL_T > temp_x( NPARTS, SIXTRL_REAL_T{ 0 } );
                std::vector< SIXTRL_REAL_T > temp_y( NPARTS, SIXTRL_REAL_T{ 0 } );
                
                for( std::size_t ii = 0 ; ii < NPARTS ; ++ii )
                {
                    static SIXTRL_REAL_T const PX_PY = SIXTRL_REAL_T{ 2.0 };
                    SIXTRL_REAL_T const angle = px_py_angle_dist( prng );
                    SIXTRL_REAL_T const py = PX_PY * std::sin( angle );
                    SIXTRL_REAL_T const px = std::sqrt( PX_PY - py * py );
                    temp_x[ ii ] = px;
                    temp_y[ ii ] = py;
                }
                
                st_Particles_set_px( particles, temp_x.data() );
                st_Particles_set_py( particles, temp_y.data() );
                
                bool const elem_by_elem_flag = ( ptr_elem_by_elem.get() != nullptr );
                bool const turn_by_turn_flag = ( ptr_turn_by_turn.get() != nullptr );
                
                auto start = std::chrono::steady_clock::now();
                
                /* ********************************************************* */
                /* START CODE TO BENCHMARK HERE:                             */
                /* ********************************************************* */
                
                st_BeamElementInfo const* elements_begin  = 
                    st_Block_get_const_elements_begin( blocks );
                    
                for( SIXTRL_SIZE_T ii = 0 ; ii < NTURNS ; ++ii )
                {
                    for( SIXTRL_SIZE_T jj = 0 ; jj < NPARTS ; ++jj )
                    {
                        st_Track_single_particle( 
                            elements_begin, NELEMS, particles, jj, 
                                ptr_elem_by_elem.get() );
                    }
                    
                    if( turn_by_turn_flag )
                    {
                        st_Particles_copy_all_unchecked( 
                            st_ParticlesSequence_get_particles_by_index( 
                                ptr_turn_by_turn.get(), ii ), particles );
                    }
                }
                
                /* ********************************************************* */
                /* END CODE TO BENCHMARK HERE:                             */
                /* ********************************************************* */
                
                auto end = std::chrono::steady_clock::now();
                auto const diff = end - start;
                
                std::cout.precision( 9 );
                std::cout << std::setw( 20 ) << NELEMS
                          << std::setw( 20 ) << NPARTS
                          << std::setw( 20 ) << NTURNS
                          << std::setw( 20 ) << std::boolalpha << turn_by_turn_flag
                          << std::setw( 20 ) << std::boolalpha << elem_by_elem_flag                          
                          << std::setw( 20 ) 
                          << std::chrono::duration< double, std::ratio< 1 > >( diff ).count()
                          << std::endl;
                
                /* --------------------------------------------------------- */
                /* Cleaning up: */
                
                st_Particles_free( particles );
                free( particles );
                particles = nullptr;
            }
        }
        
        st_Block_free( blocks );
        free( blocks );
        blocks = nullptr;
    }
    
    return 0;
}

/* end: */
