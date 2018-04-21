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

#include "sixtracklib/common/single_particle.h"
#include "sixtracklib/common/impl/track_single.h"
#include "sixtracklib/common/impl/block_drift_type.h"

#if defined( _OPENMP )
#include <omp.h>
#endif /* defined( _OPENMP ) */

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

int main()
{
    using prng_t             = std::mt19937_64;
    using prng_seed_t        = std::mt19937_64::result_type;
    using elem_type_dist_t   = std::uniform_int_distribution<>;
    using drift_len_dist_t   = std::uniform_real_distribution<>;
    using px_py_angle_dist_t = std::uniform_real_distribution< SIXTRL_REAL_T >;
    
    /* to keep it reproducible for the moment, seed the rng with knwon number */
    static prng_seed_t const PRNG_SEED = prng_seed_t{ 20180416 }; 
    
    static SIXTRL_REAL_T const MIN_DRIFT_LENGTH = SIXTRL_REAL_T{ 0.1 };
    static SIXTRL_REAL_T const MAX_DRIFT_LENGTH = SIXTRL_REAL_T{ 1.0 };
    
    /* These values are not supposed to make physical sense (now) .... */
    static SIXTRL_REAL_T const ZERO   = SIXTRL_REAL_T{ 0.0 };
    static SIXTRL_REAL_T const BETA0  = SIXTRL_REAL_T{ 1.0 };
    static SIXTRL_REAL_T const RPP    = SIXTRL_REAL_T{ 0.5 };
    static SIXTRL_REAL_T const RVV    = SIXTRL_REAL_T{ 0.5 };
    static SIXTRL_REAL_T const DELTA  = SIXTRL_REAL_T{ 2.0 };
    static SIXTRL_REAL_T const PSIGMA = SIXTRL_REAL_T{ 1.0 };
    static SIXTRL_REAL_T const PX_PY  = SIXTRL_REAL_T{ 2.0 };
    
    prng_t prng( PRNG_SEED );
        
    std::vector< st_BeamElementType > const 
        ELEMENT_TYPES{ st_ELEMENT_TYPE_DRIFT, st_ELEMENT_TYPE_DRIFT_EXACT };
    
    std::vector< SIXTRL_SIZE_T > const 
        NUM_OF_ELEMENTS{ 10000000 }; 
    
    std::vector< SIXTRL_SIZE_T > const 
        NUM_OF_PARTICLES{ 32 };
        
    std::vector< SIXTRL_SIZE_T > const 
        NUM_OF_TURNS{ 1 };
        
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
        
        std::size_t const MAX_ELEM_TYPE_INDEX = ( !ELEMENT_TYPES.empty() )
            ? ELEMENT_TYPES.size() - std::size_t{ 1 } : std::size_t{ 0 };
            
        elem_type_dist_t elem_type_dist( std::size_t{ 0 }, MAX_ELEM_TYPE_INDEX );
        drift_len_dist_t drift_len_dist( MIN_DRIFT_LENGTH, MAX_DRIFT_LENGTH );        
        px_py_angle_dist_t px_py_angle_dist( 0.0, 2.0 * M_PI );
        
        std::vector< st_DriftSingle > blocks( NELEMS );
        std::vector< st_BeamElementInfo > beam_elements( NELEMS );
        
        SIXTRL_ASSERT( blocks.size() == NELEMS );
        SIXTRL_ASSERT( beam_elements.size() == NELEMS );
        
        SIXTRL_INT64_T element_id = SIXTRL_INT64_T{ 0 };
        
        for( std::size_t ii = 0 ; ii < NELEMS ; ++ii, ++element_id )
        {
            st_BeamElementType const type_id = 
                ELEMENT_TYPES[ elem_type_dist( prng ) ];
            
            blocks[ ii ].type_id = static_cast< uint64_t >( type_id );
            blocks[ ii ].length = drift_len_dist( prng );
            blocks[ ii ].element_id = element_id++;
            
            beam_elements[ ii ].element_id = element_id;
            beam_elements[ ii ].type_id = type_id;
            beam_elements[ ii ].ptr_mem_begin = static_cast< void* >( &blocks[ ii ] );
        }
        
        for( SIXTRL_SIZE_T const NTURNS : NUM_OF_TURNS )
        {
            for( std::size_t const NPARTS : NUM_OF_PARTICLES )
            {
                std::vector< st_SingleParticle > particles( NPARTS );
                SIXTRL_REAL_T x_coord = ZERO;
                SIXTRL_REAL_T const delta_x = SIXTRL_REAL_T{ 0.1 };
                SIXTRL_INT64_T particle_id  = SIXTRL_INT64_T{ 0 };
                
                for( std::size_t ii = 0 ; ii < NPARTS ; 
                        ++ii, ++particle_id, x_coord += delta_x  )
                {
                    SIXTRL_REAL_T const angle = px_py_angle_dist( prng );
                    SIXTRL_REAL_T const px = PX_PY * std::cos( angle );
                    SIXTRL_REAL_T const py = std::sqrt( PX_PY * PX_PY - px * px );
                    
                    st_SingleParticle single;
                    
                    single.q0      = ZERO;
                    single.mass0   = ZERO;
                    single.beta0   = BETA0;
                    single.gamma0  = ZERO;
                    single.p0c     = ZERO;
                    
                    single.partid  = particle_id;
                    single.elemid  = -1;
                    single.turn    = -1;
                    single.state   = SIXTRL_INT64_T{ 0 };
                    
                    single.s       = ZERO;
                    single.x       = x_coord;
                    single.y       = ZERO;
                    single.px      = px;
                    single.py      = py;
                    single.sigma   = ZERO;
                    
                    single.psigma  = PSIGMA;
                    single.delta   = DELTA;
                    single.rpp     = RPP;
                    single.rvv     = RVV;
                    single.chi     = ZERO;
                    
                    particles[ ii ] = single;
                }
                
                auto start = std::chrono::steady_clock::now();
                
                /* ********************************************************* */
                /* START CODE TO BENCHMARK HERE:                             */
                /* ********************************************************* */
                
                st_BeamElementInfo const* elements_begin = beam_elements.data();
                    
                #if defined( _OPENMP )
                std::size_t const MAX_NUM_THREADS = omp_get_max_threads();
                
                if( ( NPARTS >= MAX_NUM_THREADS ) && ( MAX_NUM_THREADS > 0 ) )
                {
                    #pragma omp parallel num_threads( MAX_NUM_THREADS )
                    {
                        
                #endif /* defined( OPENMP ) */                    
                    
                        for( SIXTRL_SIZE_T ii = 0 ; ii < NTURNS ; ++ii )
                        {
                            int const nparts = NPARTS;
                            
                            #if defined( _OPENMP )
                            #pragma omp for nowait
                            #endif /* defined( _OPENMP ) */
                            for( int jj = 0 ; jj < nparts ; ++jj )
                            {
                                st_TrackSingle_single_particle( 
                                    elements_begin, NELEMS, &particles[ jj ] );
                            }
                        }                    
                    
                #if defined( _OPENMP )
                    }
                }
                #endif /* defined( _OPENMP ) */
                
                /* ********************************************************* */
                /* END CODE TO BENCHMARK HERE:                             */
                /* ********************************************************* */
                
                auto end = std::chrono::steady_clock::now();
                auto const diff = end - start;
                
                std::cout.precision( 9 );
                std::cout << std::setw( 20 ) << NELEMS
                          << std::setw( 20 ) << NPARTS
                          << std::setw( 20 ) << NTURNS
                          << std::setw( 20 ) << std::boolalpha << false
                          << std::setw( 20 ) << std::boolalpha << false
                          << std::setw( 20 ) 
                          << std::chrono::duration< double, std::ratio< 1 > >( diff ).count()
                          << std::endl;
            }
        }
    }
    
    return 0;
}

/* end: */
