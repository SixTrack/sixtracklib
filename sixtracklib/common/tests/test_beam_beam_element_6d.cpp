#define _USE_MATH_DEFINES 

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>

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
#include "sixtracklib/_impl/testdata_files.h"

#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/impl/beam_beam_element_6d.h"
#include "sixtracklib/common/tests/test_particles_tools.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

TEST( CommonBeamBeamElement6dTests, TestLorentzBoostAndInverseLorentzBoost )
{
    st_Blocks particles_buffer;
    st_Blocks copied_particles_buffer;
    st_Blocks diff_particles_buffer;
        
    st_Blocks_preset( &particles_buffer );
    st_Blocks_init( &particles_buffer, 1u, ( 1 << 20 ) );
    
    st_Blocks_preset( &copied_particles_buffer );
    st_Blocks_init( &copied_particles_buffer, 1u, ( 1 << 20 )  );
    
    st_Blocks_preset( &diff_particles_buffer );
    st_Blocks_init( &diff_particles_buffer, 1u, ( 1 << 20 ) );
    
    st_Particles* particles = 
        st_Blocks_add_particles( &particles_buffer, 2u );
    
    st_Particles* initial_particles = 
        st_Blocks_add_particles( &copied_particles_buffer, 2u );
    
    st_Particles* diff_particles = 
        st_Blocks_add_particles( &diff_particles_buffer, 2u );
    
    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE( initial_particles != nullptr );
    
    st_Particles_set_q0_value( initial_particles,     0, 1.0 );
    st_Particles_set_q0_value( initial_particles,     1, 1.0 );
    
    st_Particles_set_mass0_value( initial_particles,  0, 938272013.0 );
    st_Particles_set_mass0_value( initial_particles,  1, 938272013.0 );
    
    st_Particles_set_beta0_value( initial_particles,  0, 0.9999999895816052 );
    st_Particles_set_beta0_value( initial_particles,  1, 0.9999999895816052 );
                                                     
    st_Particles_set_gamma0_value( initial_particles, 0, 6927.628566067013 );
    st_Particles_set_gamma0_value( initial_particles, 1, 6927.628566067013 );
                                                     
    st_Particles_set_p0c_value( initial_particles,    0, 6499999932280.434 );
    st_Particles_set_p0c_value( initial_particles,    1, 6499999932280.434 );
                                                     
    st_Particles_set_s_value( initial_particles,      0, 0.0 );
    st_Particles_set_s_value( initial_particles,      1, 0.0 );
    
    st_Particles_set_x_value( initial_particles,      0, 0.0007145746958428162 );
    st_Particles_set_x_value( initial_particles,      1, 0.0007145736958428162 );
    
    st_Particles_set_y_value( initial_particles,      0, 0.0006720936794100408 );
    st_Particles_set_y_value( initial_particles,      1, 0.0006720936794100408 );
    
    st_Particles_set_px_value( initial_particles,     0, -1.7993573732749124e-05 );
    st_Particles_set_px_value( initial_particles,     1, -1.7993573732749124e-05 );
    
    st_Particles_set_py_value( initial_particles,     0, 7.738322282156584e-06 );
    st_Particles_set_py_value( initial_particles,     1, 7.738322282156584e-06 );
    
    st_Particles_set_sigma_value( initial_particles,  0, 1.700260986257983e-05 );
    st_Particles_set_sigma_value( initial_particles,  1, 1.700260986257983e-05 );
    
    st_Particles_set_psigma_value( initial_particles, 0, 0.00027626172996313825 );
    st_Particles_set_psigma_value( initial_particles, 1, 0.00027626172996313825 );
    
    st_Particles_set_delta_value( initial_particles,  0, 0.00027626172996228097 );
    st_Particles_set_delta_value( initial_particles,  1, 0.00027626172996228097 );
    
    st_Particles_set_rpp_value( initial_particles,    0, 0.9997238145695025 );
    st_Particles_set_rpp_value( initial_particles,    1, 0.9997238145695025 );
    
    st_Particles_set_rvv_value( initial_particles,    0, 0.999999999994246 );
    st_Particles_set_rvv_value( initial_particles,    1, 0.999999999994246 );
    
    st_Particles_set_chi_value( initial_particles,    0, 1.0 );
    st_Particles_set_chi_value( initial_particles,    0, 1.0 );
    
    st_Particles_set_state_value( initial_particles,  0, 0 );
    st_Particles_set_state_value( initial_particles,  1, 0 );
    
    st_Particles_set_particle_id_value( initial_particles,        0, 1 );
    st_Particles_set_particle_id_value( initial_particles,        1, 2 );
    
    st_Particles_set_lost_at_element_id_value( initial_particles, 0, -1 );
    st_Particles_set_lost_at_element_id_value( initial_particles, 1, -1 );
    
    st_Particles_set_lost_at_turn_value( initial_particles,       0, -1 );
    st_Particles_set_lost_at_turn_value( initial_particles,       1, -1 );
    
    /* --------------------------------------------------------------------- */
    
    double const MIN_PHI        = ( -10.0 * M_PI ) / 180.0;
    double const MAX_PHI        = ( -10.0 * M_PI ) / 180.0;
                                    
    double const MIN_ALPHA      = ( -45.0 * M_PI ) / 180.0;
    double const MAX_ALPHA      = ( +45.0 * M_PI ) / 180.0;
    
    std::mt19937_64 prng( 20180626 );
    
    std::uniform_real_distribution<> phi_distr( MIN_PHI, MAX_PHI );    
    std::uniform_real_distribution<> alpha_distr( MIN_ALPHA, MAX_ALPHA );
    
    std::size_t const NUM_BOOSTS = 1000;
    
    for( std::size_t ii = std::size_t{ 0 } ; ii < NUM_BOOSTS ; ++ii )
    {
        double const alpha = alpha_distr( prng );
        double const phi   = phi_distr( prng );
        
        st_BeamBeamBoostData boost_data;
        st_BeamBeamBoostData_preset( &boost_data );
                        
        st_BeamBeamBoostData_set_sphi(   &boost_data, std::sin( phi )   );
        st_BeamBeamBoostData_set_cphi(   &boost_data, std::cos( phi )   );
        st_BeamBeamBoostData_set_tphi(   &boost_data, std::tan( phi )   );
        st_BeamBeamBoostData_set_salpha( &boost_data, std::sin( alpha ) );
        st_BeamBeamBoostData_set_calpha( &boost_data, std::cos( alpha ) );
        
        st_Particles_copy_all_unchecked( particles, initial_particles );
        
        int ret = st_BeamBeam_boost_particle( particles, 0u, &boost_data );
        ret    |= st_BeamBeam_boost_particle( particles, 1u, &boost_data );
        
        ASSERT_TRUE( ret == 0 );
        
        ret  = st_BeamBeam_inv_boost_particle( particles, 0u, &boost_data );
        ret |= st_BeamBeam_inv_boost_particle( particles, 1u, &boost_data );
        
        ASSERT_TRUE( ret == 0 );
        
        if( !st_Particles_buffer_compare_values( 
                &particles_buffer, &copied_particles_buffer ) )
        {
            st_Particles_calculate_difference(
                initial_particles, particles, diff_particles );
            
            printf( "ii = %6u\r\n", static_cast< unsigned >( ii ) );
            st_Particles_print( stdout, diff_particles );
            printf( "\r\n" );
        }
    }
    
    /* --------------------------------------------------------------------- */
    
    st_Blocks_free( &particles_buffer );
    st_Blocks_free( &copied_particles_buffer );
    st_Blocks_free( &diff_particles_buffer );
}

/* end: sixtracklib/common/tests/test_beam_beam_element_6d.cpp */
