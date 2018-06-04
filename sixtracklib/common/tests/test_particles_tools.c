#include "sixtracklib/common/tests/test_particles_tools.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/details/random.h"
#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/particles.h"

extern void NS(Particles_random_init)( NS(Particles)* SIXTRL_RESTRICT p );

extern int NS(Particles_have_same_structure)(
    const st_Particles *const SIXTRL_RESTRICT lhs, 
    const st_Particles *const SIXTRL_RESTRICT rhs );

extern int NS(Particles_map_to_same_memory)(
    const st_Particles *const SIXTRL_RESTRICT lhs, 
    const st_Particles *const SIXTRL_RESTRICT rhs );

extern int NS(Particles_compare_values)(
    const st_Particles *const SIXTRL_RESTRICT lhs, 
    const st_Particles *const SIXTRL_RESTRICT rhs );


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


int NS(Particles_have_same_structure)(
    const st_Particles *const SIXTRL_RESTRICT lhs, 
    const st_Particles *const SIXTRL_RESTRICT rhs )
{
    return ( ( lhs != 0 ) && ( rhs != 0 ) && 
             ( NS(Particles_get_num_particles)( lhs ) ==
               NS(Particles_get_num_particles)( rhs ) ) )
        ? 1 : 0;
}


int NS(Particles_map_to_same_memory)(
    const st_Particles *const SIXTRL_RESTRICT lhs, 
    const st_Particles *const SIXTRL_RESTRICT rhs )
{
    int result = 0;
    
    if( NS(Particles_have_same_structure)( lhs, rhs ) )
    {
        result = (
            ( NS(Particles_get_const_q0)( lhs ) ==
              NS(Particles_get_const_q0)( rhs ) ) &&
            ( NS(Particles_get_const_mass0)( lhs ) ==
              NS(Particles_get_const_mass0)( rhs ) ) &&
            ( NS(Particles_get_const_beta0)( lhs ) ==
              NS(Particles_get_const_beta0)( rhs ) ) &&
            ( NS(Particles_get_const_gamma0)( lhs ) ==
              NS(Particles_get_const_gamma0)( rhs ) ) &&
            ( NS(Particles_get_const_p0c)( lhs ) ==
              NS(Particles_get_const_p0c)( rhs ) ) &&
            ( NS(Particles_get_const_s)( lhs ) ==
              NS(Particles_get_const_s)( rhs ) ) &&
            ( NS(Particles_get_const_x)( lhs ) ==
              NS(Particles_get_const_x)( rhs ) ) &&
            ( NS(Particles_get_const_y)( lhs ) ==
              NS(Particles_get_const_y)( rhs ) ) &&
            ( NS(Particles_get_const_px)( lhs ) ==
              NS(Particles_get_const_px)( rhs ) ) &&
            ( NS(Particles_get_const_py)( lhs ) ==
              NS(Particles_get_const_py)( rhs ) ) &&
            ( NS(Particles_get_const_sigma)( lhs ) ==
              NS(Particles_get_const_sigma)( rhs ) ) &&  
            ( NS(Particles_get_const_psigma)( lhs ) ==
              NS(Particles_get_const_psigma)( rhs ) ) &&
            ( NS(Particles_get_const_delta)( lhs ) ==
              NS(Particles_get_const_delta)( rhs ) ) &&
            ( NS(Particles_get_const_rpp)( lhs ) ==
              NS(Particles_get_const_rpp)( rhs ) ) &&
            ( NS(Particles_get_const_rvv)( lhs ) ==
              NS(Particles_get_const_rvv)( rhs ) ) &&
            ( NS(Particles_get_const_chi)( lhs ) ==
              NS(Particles_get_const_chi)( rhs ) ) &&
            ( NS(Particles_get_const_particle_id)( lhs ) ==
              NS(Particles_get_const_particle_id)( rhs ) ) &&
            ( NS(Particles_get_const_lost_at_element_id)( lhs ) ==
              NS(Particles_get_const_lost_at_element_id)( rhs ) ) &&  
            ( NS(Particles_get_const_lost_at_turn)( lhs ) ==
              NS(Particles_get_const_lost_at_turn)( rhs ) ) &&  
            ( NS(Particles_get_const_state)( lhs ) ==
              NS(Particles_get_const_state)( rhs ) ) ) 
        ? 1 : 0;
    }
    
    return result;
}


int NS(Particles_compare_values)(
    const st_Particles *const SIXTRL_RESTRICT lhs, 
    const st_Particles *const SIXTRL_RESTRICT rhs )
{
    int cmp_result = -1;
    
    if( NS(Particles_have_same_structure)( lhs, rhs ) )
    {
        NS(block_size_t) const NUM_PARTICLES =
            NS(Particles_get_num_particles)( lhs );
            
        NS(block_size_t) const REAL_ATTRIBUTE_SIZE =
            NUM_PARTICLES * sizeof( SIXTRL_REAL_T );
            
        NS(block_size_t) const I64_ATTRIBUTE_SIZE =
            NUM_PARTICLES * sizeof( SIXTRL_INT64_T );
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
            
        cmp_result = memcmp( 
            st_Particles_get_const_q0( lhs ), 
            st_Particles_get_const_q0( rhs ), 
            REAL_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_mass0( lhs ), 
            st_Particles_get_const_mass0( rhs ), 
            REAL_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_beta0( lhs ), 
            st_Particles_get_const_beta0( rhs ), 
            REAL_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_gamma0( lhs ), 
            st_Particles_get_const_gamma0( rhs ), 
            REAL_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
                
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_p0c( lhs ), 
            st_Particles_get_const_p0c( rhs ), 
            REAL_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_s( lhs ), 
            st_Particles_get_const_s( rhs ), 
            REAL_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_x( lhs ), 
            st_Particles_get_const_x( rhs ), 
            REAL_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_y( lhs ), 
            st_Particles_get_const_y( rhs ), 
            REAL_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_px( lhs ), 
            st_Particles_get_const_px( rhs ), 
            REAL_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_py( lhs ), 
            st_Particles_get_const_py( rhs ), 
            REAL_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_sigma( lhs ), 
            st_Particles_get_const_sigma( rhs ), 
            REAL_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_psigma( lhs ), 
            st_Particles_get_const_psigma( rhs ), 
            REAL_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_delta( lhs ), 
            st_Particles_get_const_delta( rhs ), 
            REAL_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_rpp( lhs ), 
            st_Particles_get_const_rpp( rhs ), 
            REAL_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_rvv( lhs ), 
            st_Particles_get_const_rvv( rhs ), 
            REAL_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_chi( lhs ), 
            st_Particles_get_const_chi( rhs ), 
            REAL_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_particle_id( lhs ), 
            st_Particles_get_const_particle_id( rhs ), 
            I64_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        cmp_result = memcmp( 
            st_Particles_get_const_lost_at_element_id( lhs ), 
            st_Particles_get_const_lost_at_element_id( rhs ), 
            I64_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_lost_at_turn( lhs ), 
            st_Particles_get_const_lost_at_turn( rhs ), 
            I64_ATTRIBUTE_SIZE );
        
        if( cmp_result != 0 ) return cmp_result;
        
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
        cmp_result = memcmp( 
            st_Particles_get_const_state( lhs ),
            st_Particles_get_const_state( rhs ), I64_ATTRIBUTE_SIZE );
    }
    else if( ( lhs != 0 ) && ( rhs == 0 ) )
    {
        cmp_result = 1;
    }
    else
    {
        cmp_result = -1;
    }
    
    return cmp_result;
}

/* end: sixtracklib/common/tests/test_particles_tools.c */
