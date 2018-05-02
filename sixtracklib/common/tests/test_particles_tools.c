#include "sixtracklib/common/tests/test_particles_tools.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/details/random.h"
#include "sixtracklib/common/impl/particles_impl.h"

extern void NS(Particles_random_init)( NS(Particles)* SIXTRL_RESTRICT p );

void NS(Particles_random_init)( NS(Particles)* SIXTRL_RESTRICT p )
{
    SIXTRL_SIZE_T const NUM_PARTICLES = NS(Particles_get_num_particles)( p );
    
    if( ( p != 0 ) && ( NUM_PARTICLES > ( SIXTRL_SIZE_T )0u ) )
    {
        SIXTRL_REAL_T const TWO_PI  = 2.0 * M_PI;
        SIXTRL_REAL_T const Q0      = 1.0;
        SIXTRL_REAL_T const MASS0   = 1.0;
        SIXTRL_REAL_T const BETA0   = 1.0;
        SIXTRL_REAL_T const GAMMA0  = 1.0;
        SIXTRL_REAL_T const P0C     = 0.0;
            
        SIXTRL_REAL_T const S       = 0.0;
        SIXTRL_REAL_T const MIN_X   = 0.0;
        SIXTRL_REAL_T const MAX_X   = 0.2;
        SIXTRL_REAL_T const DELTA_X = ( MAX_X - MIN_X );
        
        SIXTRL_REAL_T const MIN_Y   = 0.0;
        SIXTRL_REAL_T const MAX_Y   = 0.3;    
        SIXTRL_REAL_T const DELTA_Y = ( MAX_Y - MIN_Y );
                
        SIXTRL_REAL_T const P       = 0.1;
        SIXTRL_REAL_T const SIGMA   = 0.0;
        
        SIXTRL_REAL_T const PSIGMA  = 0.0;    
        SIXTRL_REAL_T const RPP     = 1.0;
        SIXTRL_REAL_T const RVV     = 1.0;
        SIXTRL_REAL_T const DELTA   = 0.5;
        SIXTRL_REAL_T const CHI     = 0.0;
        
        int64_t const STATE  = INT64_C( 0 );
        
        size_t ii; 
        int64_t particle_id = 0;
            
        for( ii = 0 ; ii < NUM_PARTICLES ; ++ii, ++particle_id )
        {
            SIXTRL_REAL_T const ANGLE = NS(Random_genrand64_real1)() * TWO_PI;
            SIXTRL_REAL_T const PX = P * cos( ANGLE );
            SIXTRL_REAL_T const PY = sqrt( P * P - PX * PX );
            SIXTRL_REAL_T const X  = MIN_X + NS(Random_genrand64_real1)() * DELTA_X;
            SIXTRL_REAL_T const Y  = MIN_Y + NS(Random_genrand64_real1)() * DELTA_Y;
            
            NS(Particles_set_q0_value)(                 p, ii, Q0 );
            NS(Particles_set_mass0_value)(              p, ii, MASS0 );
            NS(Particles_set_beta0_value)(              p, ii, BETA0 );
            NS(Particles_set_gamma0_value)(             p, ii, GAMMA0 );
            NS(Particles_set_p0c_value)(                p, ii, P0C );
            
            NS(Particles_set_particle_id_value)(        p, ii, particle_id );
            NS(Particles_set_lost_at_element_id_value)( p, ii, -1 );
            NS(Particles_set_lost_at_turn_value)(       p, ii, -1 );
            NS(Particles_set_state_value)(              p, ii, STATE );
            
            NS(Particles_set_s_value)(                  p, ii, S );
            NS(Particles_set_x_value)(                  p, ii, X );
            NS(Particles_set_y_value)(                  p, ii, Y );
            NS(Particles_set_px_value)(                 p, ii, PX );
            NS(Particles_set_py_value)(                 p, ii, PY );
            NS(Particles_set_sigma_value)(              p, ii, SIGMA );
                                                    
            NS(Particles_set_psigma_value)(             p, ii, PSIGMA );
            NS(Particles_set_delta_value)(              p, ii, DELTA );
            NS(Particles_set_rpp_value)(                p, ii, RPP );
            NS(Particles_set_rvv_value)(                p, ii, RVV );
            NS(Particles_set_chi_value)(                p, ii, CHI );
        }
    }
    
    return;
}

/* end: sixtracklib/common/tests/test_particles_tools.c */
