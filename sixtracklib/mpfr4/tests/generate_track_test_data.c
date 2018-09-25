#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#if !defined( MPFR_USE_INTMAX_T )
#define define( MPFR_USE_INTMAX_T )
#endif /* !defined( MPFR_USE_INTMAX_T ) */

#include <mpfr.h>

#if defined( __NAMESPACE )
    #define __SAVED_NAMESPACE __NAMESPACE
    #undef  __NAMESPACE     
#endif /* !defiend( __NAMESPACE ) */

#if !defined( __NAMESPACE )
    #define __NAMESPACE st_    
#endif /* !defined( __NAMESPACE ) */


#include "sixtracklib/mpfr4/impl/particles_impl.h"
#include "sixtracklib/mpfr4/beam_elements.h"
#include "sixtracklib/mpfr4/track.h"

#include "sixtracklib/common/impl/particles_impl.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/details/random.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

void st_Particles_init_random_mpfr4(
    st_Particles* SIXTRL_RESTRICT particles, mpfr_rnd_t const rnd );

int st_TurnByTurnContainer_init_mpfr4(
    st_ParticlesContainer* SIXTRL_RESTRICT elem_by_elem_buffer, 
    st_block_size_t const NUM_TURNS, 
    st_block_num_elements_t const NUM_PARTICLES );

int st_ElementsByElementsContainer_init_mpfr4(
    st_ParticlesContainer* SIXTRL_RESTRICT elem_by_elem_buffer, 
    st_block_size_t const NUM_TURNS, 
    st_block_num_elements_t const NUM_ELEMENTS, 
    st_block_num_elements_t const NUM_PARTICLES );

int st_Drifts_init_random_mpfr4(
    st_Drift* SIXTRL_RESTRICT drift, mpfr_rnd_t const rnd );

void st_Drifts_clear_mpfr4(
    st_BeamElements* SIXTRL_RESTRICT beam_elements );



int main( int argc, char* argv[] )
{
    int ret = 0; 
    int use_turn_by_turn_buffer = 0;
    
    static st_block_size_t const REAL_SIZE = sizeof( SIXTRL_REAL_T );
    static st_block_size_t const I64_SIZE  = sizeof( SIXTRL_INT64_T );
    
    st_block_num_elements_t ii = 0;
    st_block_num_elements_t NUM_PARTICLES = 1000;
    st_block_num_elements_t NUM_ELEMENTS  = 100;
    
    mpfr_prec_t prec = ( mpfr_prec_t )128u;
    uint64_t prng_seed = ( uint64_t )20180501u;
    
    st_block_size_t jj = 0;
    st_block_size_t BLOCK_CAPACITY = 1;
    st_block_size_t DATA_CAPACITY  = 0;
    st_block_size_t NUM_TURNS      = 1;
        
    mpfr_rnd_t  const rnd  = mpfr_get_default_rounding_mode();
    
    st_Particles particles;    
    st_ParticlesContainer particle_buffer;
    st_ParticlesContainer elem_by_elem_buffer;
    st_ParticlesContainer turn_by_turn_buffer;
    
    st_BeamElements beam_elements;
    
    printf( "\r\n\r\nUsage: %s [NUM_PARTICLES=1000] [NUM_ELEMENTS=100] "
            "[NUM_TURNS=1] [precision=128] [seed=20180501] \r\n", argv[ 0 ] );
    
    if( argc >= 6 )
    {
        uint64_t const temp_seed = ( uint64_t )atoi( argv[ 5 ] );
        if( temp_seed > ( uint64_t )0u ) prng_seed = temp_seed;
    }
    
    if( argc >= 5 )
    {
        uint64_t const  temp_prec = ( uint64_t )atoi( argv[ 4 ] );
        if( temp_prec > ( uint64_t )0u ) prec = ( mpfr_prec_t )temp_prec;
    }

    if( argc >= 4 )
    {
        st_block_size_t const temp_num_turns = atoi( argv[ 3 ] );
        if( temp_num_turns > 0 ) NUM_TURNS = temp_num_turns;        
    }
    
    if( argc >= 3 )
    {
        st_block_num_elements_t const temp_num_elements = atoi( argv[ 2 ] );        
        if( temp_num_elements > 0 ) NUM_ELEMENTS = temp_num_elements;
    }
    
    if( argc > 1 )
    {
        st_block_num_elements_t const temp_num_particles = atoi( argv[ 1 ] );
        if( temp_num_particles > 0 ) NUM_PARTICLES = temp_num_particles;
        
        printf( " --> Use values: \r\n" 
                "   - NUM_PARTICLES: %12ld\r\n"
                "   - NUM_ELEMENTS : %12ld\r\n"
                "   - NUM_TURNS    : %12lu\r\n"
                "   - seed         : %12lu\r\n\r\n",
                NUM_PARTICLES, NUM_ELEMENTS, NUM_TURNS, prng_seed );
    }    
    else
    {
        printf( " --> Use default values\r\n" );        
    }
    
    /* -------------------------------------------------------------------- */
    
    st_Random_init_genrand64( prng_seed );
    
    /* -------------------------------------------------------------------- */
    
    st_BeamElements_preset( &beam_elements );
    
    st_BeamElements_set_data_begin_alignment( &beam_elements, REAL_SIZE );
    st_BeamElements_set_data_alignment( &beam_elements, REAL_SIZE );
    
    BLOCK_CAPACITY = NUM_ELEMENTS;
    DATA_CAPACITY  = BLOCK_CAPACITY * ( REAL_SIZE + I64_SIZE );
    
    st_BeamElements_reserve_num_blocks( &beam_elements, BLOCK_CAPACITY );
    st_BeamElements_reserve_for_data( &beam_elements, DATA_CAPACITY );
    
    for( ii = 0 ; ii < NUM_ELEMENTS ; ++ii )
    {
        st_Drift drift;
        st_Drift_preset( &drift );
        
        ret |= st_BeamElements_create_beam_element_mpfr4(
            &drift, &beam_elements, st_BLOCK_TYPE_DRIFT, prec );
        
        if( ret == 0 )
        {
            ret = st_Drifts_init_random_mpfr4( &drift, rnd );
        }
        
        if( ret != 0 )
        {
            break;
        }
    }
    
    /* -------------------------------------------------------------------- */
    
    st_Particles_preset( &particles );
    st_ParticlesContainer_preset( &particle_buffer );
    
    if( ret == 0 )
    {
        ret  = st_ParticlesContainer_init_num_of_blocks_mpfr4( 
            &particle_buffer, 1, NUM_PARTICLES, prec );
    
        ret |= st_ParticlesContainer_get_particles( 
            &particles, &particle_buffer, 0 );
    }
    
    if( ret == 0 )
    {
        st_Particles_init_random_mpfr4( &particles, rnd );
    }
    
    /* --------------------------------------------------------------------- */
    
    st_ParticlesContainer_preset( &elem_by_elem_buffer );
    
    if( ret == 0 )
    {
        ret = st_ParticlesContainer_init_num_of_blocks_mpfr4( 
            &elem_by_elem_buffer, NUM_TURNS * NUM_ELEMENTS, 
                NUM_PARTICLES, prec );
    }
    
    /* --------------------------------------------------------------------- */
    
    st_ParticlesContainer_preset( &turn_by_turn_buffer );
    
    if( ret == 0 )
    {
        ret = st_ParticlesContainer_init_num_of_blocks_mpfr4( 
            &turn_by_turn_buffer, NUM_TURNS, NUM_PARTICLES, prec );
    }
    
    /* --------------------------------------------------------------------- */
    
    if( ret == 0 )
    {
        st_block_num_elements_t kk = 0;
        
        for( jj = 0 ; jj < NUM_TURNS ; ++jj, kk += NUM_ELEMENTS )
        {
            /* ============================================================= */
            
            ret |= st_Track_beam_elements_mpfr4( &particles, 
                    &beam_elements, prec, rnd, kk, &elem_by_elem_buffer );
            
            /* ============================================================= */
            
            if( use_turn_by_turn_buffer )
            {
                st_Particles particles_after_turn;
                
                ret |= st_ParticlesContainer_get_particles( 
                    &particles_after_turn, &turn_by_turn_buffer, jj );
                
                st_Particles_copy_all_unchecked_mpfr4(
                    &particles_after_turn, &particles, rnd );
            }
            
            if( ret != 0 )
            {
                break;
            }
        }
    }
    
    /* -------------------------------------------------------------------- */
    
    st_BeamElements_free_mpfr4( &beam_elements );
    st_ParticlesContainer_free_mpfr4( &particle_buffer );
    st_ParticlesContainer_free_mpfr4( &turn_by_turn_buffer );
    st_ParticlesContainer_free_mpfr4( &elem_by_elem_buffer );
    
    /* -------------------------------------------------------------------- */
    
    mpfr_free_cache();
    
    return 0;
}

void st_Particles_init_random_mpfr4(
    st_Particles* SIXTRL_RESTRICT particles, mpfr_rnd_t const rnd )
{
    char const Q0_STR[]     = "1";
    char const MASS0_STR[]  = "1";
    char const BETA0_STR[]  = "1";
    char const GAMMA0_STR[] = "1";
    char const P0C_STR[]    = "0";
    
    char const S_STR[]      = "0";
    char const SIGMA_STR[]  = "0";
    char const PSIGMA_STR[] = "0";
    char const RPP_STR[]    = "1";
    char const RVV_STR[]    = "1";
    char const DELTA_STR[]  = "0.5";
    char const CHI_STR[]    = "0";
    
    double const TWO_PI  = ( double )2.0L * M_PI;
    double const MIN_X   = ( double )0.0L;
    double const MAX_X   = ( double )0.2L;
    double const DELTA_X = ( MAX_X - MIN_X );
    
    double const MIN_Y   = ( double )0.0L;
    double const MAX_Y   = ( double )0.3L;    
    double const DELTA_Y = ( MAX_Y - MIN_Y );
    double const P       = ( double )0.1L;
    
    st_block_num_elements_t ii = 0;
    st_block_num_elements_t const NUM_PARTICLES = 
        st_Particles_get_num_particles( particles );
    
    for( ii = 0 ; ii < NUM_PARTICLES ; ++ii )
    {
        double const ANGLE = st_Random_genrand64_real1() * TWO_PI;
        double const PX = P * cos( ANGLE );
        double const PY = sqrt( P * P - PX * PX );
        double const X  = MIN_X + st_Random_genrand64_real1() * DELTA_X;
        double const Y  = MIN_Y + st_Random_genrand64_real1() * DELTA_Y;
        
        mpfr_set_str( st_Particles_get_q0( 
            particles )[ ii ].value, Q0_STR, 10, rnd );
        
        mpfr_set_str( st_Particles_get_mass0( particles )[ ii ].value, 
                      MASS0_STR, 10, rnd );
        
        mpfr_set_str( st_Particles_get_beta0( particles )[ ii ].value, 
                      BETA0_STR, 10, rnd );
        
        mpfr_set_str( st_Particles_get_gamma0( particles )[ ii ].value, 
                      GAMMA0_STR, 10, rnd );
        
        mpfr_set_str( st_Particles_get_p0c( particles )[ ii ].value, 
                      P0C_STR, 10, rnd );
        
        mpfr_set_str( st_Particles_get_s( particles )[ ii ].value, 
                      S_STR, 10, rnd );
        
        mpfr_set_d( st_Particles_get_x( particles )[ ii ].value, X, rnd ); 
        mpfr_set_d( st_Particles_get_y( particles )[ ii ].value, Y, rnd );
        mpfr_set_d( st_Particles_get_px( particles )[ ii ].value, PX, rnd );
        mpfr_set_d( st_Particles_get_py( particles )[ ii ].value, PY, rnd );
        
        mpfr_set_str( st_Particles_get_sigma( particles )[ ii ].value, 
                      SIGMA_STR, 10, rnd );
        
        mpfr_set_str( st_Particles_get_psigma( particles )[ ii ].value, 
                      PSIGMA_STR, 10, rnd );
        
        mpfr_set_str( st_Particles_get_delta( particles )[ ii ].value, 
                      DELTA_STR, 10, rnd );
        
        mpfr_set_str( st_Particles_get_rpp( particles )[ ii ].value, 
                      RPP_STR, 10, rnd );
        
        mpfr_set_str( st_Particles_get_rvv( particles )[ ii ].value, 
                      RVV_STR, 10, rnd );
        
        mpfr_set_str( st_Particles_get_chi( particles )[ ii ].value, 
                      CHI_STR, 10, rnd );
    }
    
    SIXTRL_ASSERT( st_Particles_is_consistent( particles ) );
    
    return;
}

int st_Drifts_init_random_mpfr4(
    st_Drift* SIXTRL_RESTRICT drift, mpfr_rnd_t const rnd )
{
    int success = -1;
    
    st_BlockType const type_id = st_Drift_get_type_id( drift );
            
    if( ( type_id == st_BLOCK_TYPE_DRIFT ) || 
        ( type_id == st_BLOCK_TYPE_DRIFT_EXACT ) )
    {
        double const MIN_DRIFT_LENGTH  = ( double )0.05L;
        double const MAX_DRIFT_LENGTH  = ( double )1.00L;
        double const DRIFT_LENGTH_SPAN = MAX_DRIFT_LENGTH - MIN_DRIFT_LENGTH;
        
        double const drift_length = MIN_DRIFT_LENGTH + 
            DRIFT_LENGTH_SPAN * st_Random_genrand64_real1();
        
        SIXTRL_REAL_T* ptr_length = st_Drift_get_length( drift );
        SIXTRL_ASSERT( ptr_length != 0 );
        
        mpfr_set_d( ptr_length->value, drift_length, rnd );
        
        success = 0;
    }
    
    return success;
}

/* end: sixtracklib/mpfr4/tests/generate_test_data.c */
