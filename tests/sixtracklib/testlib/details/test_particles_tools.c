#include "sixtracklib/testlib/test_particles_tools.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/particles.h"

#include "sixtracklib/testlib/random.h"

extern void NS(Particles_random_init)( NS(Particles)* SIXTRL_RESTRICT p );
extern void NS(Particles_realistic_init)( NS(Particles)* SIXTRL_RESTRICT p );

/* ------------------------------------------------------------------------- */

extern int NS(Particles_have_same_structure)(
    const st_Particles *const SIXTRL_RESTRICT lhs,
    const st_Particles *const SIXTRL_RESTRICT rhs );

extern int NS(Particles_map_to_same_memory)(
    const st_Particles *const SIXTRL_RESTRICT lhs,
    const st_Particles *const SIXTRL_RESTRICT rhs );

extern int NS(Particles_compare_values)(
    const st_Particles *const SIXTRL_RESTRICT lhs,
    const st_Particles *const SIXTRL_RESTRICT rhs );

extern int NS(Particles_compare_values_with_treshold)(
    const NS(Particles) *const SIXTRL_RESTRICT lhs,
    const NS(Particles) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

static int NS(compare_sequences_exact)(
    void const* SIXTRL_RESTRICT lhs_values,
    void const* SIXTRL_RESTRICT rhs_values,
    NS(block_size_t) const num_values,
    NS(block_size_t) const element_size );

static int NS(compare_real_sequences_with_treshold)(
    SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_values,
    SIXTRL_REAL_T const* SIXTRL_RESTRICT rhs_values,
    NS(block_size_t)* SIXTRL_RESTRICT ptr_first_out_of_bounds_index,
    SIXTRL_UINT64_T const num_values, SIXTRL_REAL_T const treshold );

static void NS(compare_real_sequences_and_get_max_difference)(
    SIXTRL_REAL_T*    SIXTRL_RESTRICT ptr_max_diff,
    NS(block_size_t)* SIXTRL_RESTRICT ptr_max_diff_index,
    SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_values,
    SIXTRL_REAL_T const* SIXTRL_RESTRICT rhs_values,
    NS(block_size_t) const num_values );

static void NS(compare_int64_sequences_and_get_max_difference)(
    SIXTRL_INT64_T*    SIXTRL_RESTRICT ptr_max_diff,
    NS(block_size_t)* SIXTRL_RESTRICT ptr_max_diff_index,
    SIXTRL_INT64_T const* SIXTRL_RESTRICT lhs_values,
    SIXTRL_INT64_T const* SIXTRL_RESTRICT rhs_values,
    NS(block_size_t) const num_values );

extern void NS(Particles_get_max_difference)(
    NS(Particles)* SIXTRL_RESTRICT max_diff,
    NS(block_size_t)* SIXTRL_RESTRICT max_diff_indices,
    const NS(Particles) *const SIXTRL_RESTRICT lhs,
    const NS(Particles) *const SIXTRL_RESTRICT rhs );



extern void NS(Particles_print)(
    FILE* SIXTRL_RESTRICT fp,
    const NS(Particles) *const SIXTRL_RESTRICT particles );

extern void NS(Particles_print_max_diff)(
    FILE* SIXTRL_RESTRICT fp,
    const NS(Particles) *const SIXTRL_RESTRICT max_diff,
    NS(block_size_t) const* max_diff_indices );

/* ------------------------------------------------------------------------- */

extern int NS(Particles_buffers_have_same_structure)(
    const NS(Blocks) *const SIXTRL_RESTRICT lhs_buffer,
    const NS(Blocks) *const SIXTRL_RESTRICT rhs_buffer );

extern int NS(Particles_buffers_map_to_same_memory)(
    const NS(Blocks) *const SIXTRL_RESTRICT lhs_buffer,
    const NS(Blocks) *const SIXTRL_RESTRICT rhs_buffer );

extern int NS(Particles_buffer_compare_values)(
    const NS(Blocks) *const SIXTRL_RESTRICT lhs_buffer,
    const NS(Blocks) *const SIXTRL_RESTRICT rhs_buffer );

extern int NS(Particles_buffer_compare_values_with_treshold)(
    const NS(Blocks) *const SIXTRL_RESTRICT lhs_buffer,
    const NS(Blocks) *const SIXTRL_RESTRICT rhs_buffer,
    SIXTRL_REAL_T const treshold );

extern void NS(Particles_buffer_get_max_difference)(
    NS(Blocks)* SIXTRL_RESTRICT max_diff,
    NS(block_size_t)* SIXTRL_RESTRICT max_diff_indices,
    const NS(Blocks) *const SIXTRL_RESTRICT lhs,
    const NS(Blocks) *const SIXTRL_RESTRICT rhs );

extern void NS(Particles_buffer_print)(
    FILE* SIXTRL_RESTRICT fp,
    const NS(Blocks) *const SIXTRL_RESTRICT particles );

extern void NS(Particles_buffer_print_max_diff)(
    FILE* SIXTRL_RESTRICT fp,
    const NS(Blocks) *const SIXTRL_RESTRICT max_diff,
    NS(block_size_t) const* max_diff_indices );


/* ------------------------------------------------------------------------- */

int NS(compare_sequences_exact)(
    void const* SIXTRL_RESTRICT lhs_values,
    void const* SIXTRL_RESTRICT rhs_values,
    NS(block_size_t) const num_values,
    NS(block_size_t) const element_size )
{
    int cmp_result = -1;

    NS(block_size_t) const ATTR_LENGTH = num_values * element_size;

    if( ATTR_LENGTH > ( NS(block_size_t) )0u )
    {
        if( ( lhs_values != 0 ) && ( rhs_values != 0 ) )
        {
            cmp_result = memcmp( lhs_values, rhs_values, ATTR_LENGTH );
        }
        else if( lhs_values != 0 )
        {
            cmp_result = +1;
        }
        else if( rhs_values != 0 )
        {
            cmp_result = -1;
        }
    }

    return cmp_result;
}

int NS(compare_real_sequences_with_treshold)(
    SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_values,
    SIXTRL_REAL_T const* SIXTRL_RESTRICT rhs_values,
    NS(block_size_t)* SIXTRL_RESTRICT ptr_first_out_of_bounds_index,
    SIXTRL_UINT64_T const num_values, SIXTRL_REAL_T const treshold )
{
    int cmp_result = -1;

    if( ( lhs_values != 0 )  && ( rhs_values != 0 ) )
    {
        static SIXTRL_REAL_T const ZERO = ( SIXTRL_REAL_T )0.0L;
        NS(block_size_t) ii = 0;

        cmp_result = 0;

        for( ; ii < num_values ; ++ii )
        {
            SIXTRL_REAL_T const diff = lhs_values[ ii ] - rhs_values[ ii ];

            if( ( ( diff >= ZERO ) && (  diff > treshold ) ) ||
                ( ( diff <  ZERO ) && ( -diff > treshold ) ) )
            {
                if( ptr_first_out_of_bounds_index != 0 )
                {
                    *ptr_first_out_of_bounds_index = ii;
                }

                cmp_result = ( lhs_values[ ii ] >= rhs_values[ ii ] ) ? +1 : -1;
                break;
            }
        }
    }
    else if( lhs_values != 0 )
    {
        cmp_result = +1;
    }
    else if( rhs_values != 0 )
    {
        cmp_result = -1;
    }

    return cmp_result;
}

void NS(compare_real_sequences_and_get_max_difference)(
    SIXTRL_REAL_T*    SIXTRL_RESTRICT ptr_max_diff,
    NS(block_size_t)* SIXTRL_RESTRICT ptr_max_diff_index,
    SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_values,
    SIXTRL_REAL_T const* SIXTRL_RESTRICT rhs_values,
    NS(block_size_t) const num_values )
{
    static SIXTRL_REAL_T const ZERO = ( SIXTRL_REAL_T )0.0L;

    NS(block_size_t) max_diff_index = 0u;

    SIXTRL_REAL_T cmp_max_diff = ZERO;
    SIXTRL_REAL_T max_diff     = ZERO;

    if( ( lhs_values != 0  ) && ( rhs_values   != 0 ) &&
        ( num_values >  0u ) && ( ptr_max_diff != 0 ) )
    {
        NS(block_size_t) ii = 0u;

        for( ; ii < num_values ; ++ii )
        {
            SIXTRL_REAL_T const diff = lhs_values[ ii ] - rhs_values[ ii ];
            SIXTRL_REAL_T const cmp_diff = ( diff > ZERO ) ? diff : -diff;

            if( cmp_diff > cmp_max_diff )
            {
                cmp_max_diff = cmp_diff;
                max_diff = diff;
                max_diff_index = ii;
            }
        }
    }
    else if( ( ( lhs_values != 0 ) || ( rhs_values != 0 ) ) &&
               ( num_values > 0u ) && ( ptr_max_diff != 0 ) )
    {
        SIXTRL_REAL_T const* values =
            ( lhs_values != 0 ) ? lhs_values : rhs_values;

        NS(block_size_t) ii = 0u;

        for( ; ii < num_values ; ++ii )
        {
            SIXTRL_REAL_T const diff = values[ ii ];
            SIXTRL_REAL_T const cmp_diff = ( diff > ZERO ) ? diff : -diff;

            if( cmp_diff > cmp_max_diff )
            {
                cmp_max_diff = cmp_diff;
                max_diff = diff;
                max_diff_index = ii;
            }
        }
    }

    if( ptr_max_diff != 0 )         *ptr_max_diff = max_diff;
    if( ptr_max_diff_index != 0 )   *ptr_max_diff_index = max_diff_index;

    return;
}

void NS(compare_int64_sequences_and_get_max_difference)(
    SIXTRL_INT64_T*    SIXTRL_RESTRICT ptr_max_diff,
    NS(block_size_t)* SIXTRL_RESTRICT ptr_max_diff_index,
    SIXTRL_INT64_T const* SIXTRL_RESTRICT lhs_values,
    SIXTRL_INT64_T const* SIXTRL_RESTRICT rhs_values,
    NS(block_size_t) const num_values )
{
    static SIXTRL_INT64_T const ZERO = ( SIXTRL_INT64_T )0u;

    NS(block_size_t) max_diff_index = 0u;
    SIXTRL_INT64_T max_diff = ZERO;
    SIXTRL_INT64_T cmp_max_diff = ZERO;

    if( ( lhs_values != 0  ) && ( rhs_values   != 0 ) &&
        ( num_values >  0u ) && ( ptr_max_diff != 0 ) )
    {
        NS(block_size_t) ii = 0u;

        for( ; ii < num_values ; ++ii )
        {
            SIXTRL_INT64_T const diff = lhs_values[ ii ] - rhs_values[ ii ];
            SIXTRL_INT64_T const cmp_diff = ( diff > ZERO ) ? diff : -diff;

            if( cmp_diff > cmp_max_diff )
            {
                cmp_max_diff = cmp_diff;
                max_diff = diff;
                max_diff_index = ii;
            }
        }
    }
    else if( ( ( lhs_values != 0 ) || ( rhs_values != 0 ) ) &&
               ( num_values > 0u ) && ( ptr_max_diff != 0 ) )
    {
        SIXTRL_INT64_T const* values =
            ( lhs_values != 0 ) ? lhs_values : rhs_values;

        NS(block_size_t) ii = 0u;

        for( ; ii < num_values ; ++ii )
        {
            SIXTRL_INT64_T const diff = values[ ii ];
            SIXTRL_INT64_T const cmp_diff = ( diff > ZERO ) ? diff : -diff;

            if( cmp_diff > cmp_max_diff )
            {
                cmp_max_diff = cmp_diff;
                max_diff = diff;
                max_diff_index = ii;
            }
        }
    }

    if( ptr_max_diff != 0 )         *ptr_max_diff = max_diff;
    if( ptr_max_diff_index != 0 )   *ptr_max_diff_index = max_diff_index;

    return;
}

/* ------------------------------------------------------------------------- */

void NS(Particles_realistic_init)( NS(Particles)* SIXTRL_RESTRICT p )
{
    SIXTRL_SIZE_T const NUM_PARTICLES = NS(Particles_get_num_particles)( p );

    if( ( p != 0 ) && ( NUM_PARTICLES > ( SIXTRL_SIZE_T )0u ) )
    {
        SIXTRL_SIZE_T ii;
        SIXTRL_INT64_T particle_id = 0;

        SIXTRL_REAL_T const Q0     = ( SIXTRL_REAL_T )1.0;
        SIXTRL_REAL_T const MASS0  = ( SIXTRL_REAL_T )938272013.0;
        SIXTRL_REAL_T const BETA0  = ( SIXTRL_REAL_T )0.9999999895816052;
        SIXTRL_REAL_T const GAMMA0 = ( SIXTRL_REAL_T )6927.628566067013;
        SIXTRL_REAL_T const P0C    = ( SIXTRL_REAL_T )6499999932280.434;

        SIXTRL_REAL_T const S      = ( SIXTRL_REAL_T )0.0;
        SIXTRL_REAL_T const X0     = ( SIXTRL_REAL_T )0.0007145746958428162;
        SIXTRL_REAL_T const Y      = ( SIXTRL_REAL_T )0.0006720936794100408;
        SIXTRL_REAL_T const PX     = ( SIXTRL_REAL_T )-1.7993573732749124e-05;
        SIXTRL_REAL_T const PY     = ( SIXTRL_REAL_T )7.738322282156584e-06;
        SIXTRL_REAL_T const SIGMA  = ( SIXTRL_REAL_T )1.700260986257983e-05;
        SIXTRL_REAL_T const PSIGMA = ( SIXTRL_REAL_T )0.00027626172996313825;
        SIXTRL_REAL_T const DELTA  = ( SIXTRL_REAL_T )0.00027626172996228097;
        SIXTRL_REAL_T const RPP    = ( SIXTRL_REAL_T )0.9997238145695025;
        SIXTRL_REAL_T const RVV    = ( SIXTRL_REAL_T )0.999999999994246;
        SIXTRL_REAL_T const CHI    = ( SIXTRL_REAL_T )1.0;

        SIXTRL_REAL_T const DELTA_X = ( SIXTRL_REAL_T )1e-9;

        for( ii = 0 ; ii < NUM_PARTICLES ; ++ii, ++particle_id )
        {
            SIXTRL_REAL_T const X = X0 + DELTA_X * ii;

            NS(Particles_set_q0_value)(                 p, ii, Q0 );
            NS(Particles_set_mass0_value)(              p, ii, MASS0 );
            NS(Particles_set_beta0_value)(              p, ii, BETA0 );
            NS(Particles_set_gamma0_value)(             p, ii, GAMMA0 );
            NS(Particles_set_p0c_value)(                p, ii, P0C );

            NS(Particles_set_particle_id_value)(        p, ii, particle_id );
            NS(Particles_set_lost_at_element_id_value)( p, ii, -1 );
            NS(Particles_set_lost_at_turn_value)(       p, ii, -1 );
            NS(Particles_set_state_value)(              p, ii,  0 );

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
        SIXTRL_REAL_T const P0C     = 1.0;

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

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_q0)( lhs ),
            NS(Particles_get_const_q0)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_REAL_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_mass0)( lhs ),
            NS(Particles_get_const_mass0)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_REAL_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_beta0)( lhs ),
            NS(Particles_get_const_beta0)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_REAL_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_gamma0)( lhs ),
            NS(Particles_get_const_gamma0)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_REAL_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_p0c)( lhs ),
            NS(Particles_get_const_p0c)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_REAL_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_s)( lhs ),
            NS(Particles_get_const_s)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_REAL_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_x)( lhs ),
            NS(Particles_get_const_x)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_REAL_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_y)( lhs ),
            NS(Particles_get_const_y)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_REAL_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_px)( lhs ),
            NS(Particles_get_const_px)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_REAL_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_py)( lhs ),
            NS(Particles_get_const_py)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_REAL_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_sigma)( lhs ),
            NS(Particles_get_const_sigma)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_REAL_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_psigma)( lhs ),
            NS(Particles_get_const_psigma)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_REAL_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_delta)( lhs ),
            NS(Particles_get_const_delta)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_REAL_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_rpp)( lhs ),
            NS(Particles_get_const_rpp)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_REAL_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_rvv)( lhs ),
            NS(Particles_get_const_rvv)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_REAL_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_chi)( lhs ),
            NS(Particles_get_const_chi)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_REAL_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_particle_id)( lhs ),
            NS(Particles_get_const_particle_id)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_INT64_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_lost_at_element_id)( lhs ),
            NS(Particles_get_const_lost_at_element_id)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_INT64_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_lost_at_turn)( lhs ),
            NS(Particles_get_const_lost_at_turn)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_INT64_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_state)( lhs ),
            NS(Particles_get_const_state)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_INT64_T ) );
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

int NS(Particles_compare_values_with_treshold)(
    const NS(Particles) *const SIXTRL_RESTRICT lhs,
    const NS(Particles) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    int cmp_result = -1;

    if( NS(Particles_have_same_structure)( lhs, rhs ) )
    {
        NS(block_size_t) const NUM_PARTICLES =
            NS(Particles_get_num_particles)( lhs );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_real_sequences_with_treshold)(
            NS(Particles_get_const_q0)( lhs ),
            NS(Particles_get_const_q0)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_real_sequences_with_treshold)(
            NS(Particles_get_const_mass0)( lhs ),
            NS(Particles_get_const_mass0)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_real_sequences_with_treshold)(
            NS(Particles_get_const_beta0)( lhs ),
            NS(Particles_get_const_beta0)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_real_sequences_with_treshold)(
            NS(Particles_get_const_gamma0)( lhs ),
            NS(Particles_get_const_gamma0)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_real_sequences_with_treshold)(
            NS(Particles_get_const_p0c)( lhs ),
            NS(Particles_get_const_p0c)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_real_sequences_with_treshold)(
            NS(Particles_get_const_s)( lhs ),
            NS(Particles_get_const_s)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_real_sequences_with_treshold)(
            NS(Particles_get_const_x)( lhs ),
            NS(Particles_get_const_x)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_real_sequences_with_treshold)(
            NS(Particles_get_const_y)( lhs ),
            NS(Particles_get_const_y)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_real_sequences_with_treshold)(
            NS(Particles_get_const_px)( lhs ),
            NS(Particles_get_const_px)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_real_sequences_with_treshold)(
            NS(Particles_get_const_py)( lhs ),
            NS(Particles_get_const_py)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_real_sequences_with_treshold)(
            NS(Particles_get_const_sigma)( lhs ),
            NS(Particles_get_const_sigma)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_real_sequences_with_treshold)(
            NS(Particles_get_const_psigma)( lhs ),
            NS(Particles_get_const_psigma)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_real_sequences_with_treshold)(
            NS(Particles_get_const_delta)( lhs ),
            NS(Particles_get_const_delta)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_real_sequences_with_treshold)(
            NS(Particles_get_const_rpp)( lhs ),
            NS(Particles_get_const_rpp)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_real_sequences_with_treshold)(
            NS(Particles_get_const_rvv)( lhs ),
            NS(Particles_get_const_rvv)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_real_sequences_with_treshold)(
            NS(Particles_get_const_chi)( lhs ),
            NS(Particles_get_const_chi)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_particle_id)( lhs ),
            NS(Particles_get_const_particle_id)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_INT64_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_lost_at_element_id)( lhs ),
            NS(Particles_get_const_lost_at_element_id)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_INT64_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_lost_at_turn)( lhs ),
            NS(Particles_get_const_lost_at_turn)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_INT64_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_state)( lhs ),
            NS(Particles_get_const_state)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_INT64_T ) );
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

void NS(Particles_get_max_difference)(
    NS(Particles)* SIXTRL_RESTRICT max_diff,
    NS(block_size_t)* SIXTRL_RESTRICT max_diff_indices,
    const NS(Particles) *const SIXTRL_RESTRICT lhs,
    const NS(Particles) *const SIXTRL_RESTRICT rhs )
{
    if( ( max_diff != 0 ) && ( max_diff_indices != 0 ) &&
        ( NS(Particles_have_same_structure)( lhs, rhs ) ) &&
        ( NS(Particles_get_num_particles)( lhs ) > 0u ) &&
        ( NS(Particles_get_num_particles)( max_diff ) >= 1u ) )
    {
        NS(block_size_t) dummy_max_diff_indices[ 20 ] =
        {
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
        };

        NS(block_size_t) const NUM_PARTICLES =
            NS(Particles_get_num_particles)( lhs );

        if( max_diff_indices == 0 )
        {
            max_diff_indices = &dummy_max_diff_indices[ 0 ];
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_real_sequences_and_get_max_difference)(
            NS(Particles_get_q0)( max_diff ), &max_diff_indices[ 0 ],
            NS(Particles_get_const_q0)( lhs ),
            NS(Particles_get_const_q0)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_real_sequences_and_get_max_difference)(
            NS(Particles_get_mass0)( max_diff ), &max_diff_indices[ 1 ],
            NS(Particles_get_const_mass0)( lhs ),
            NS(Particles_get_const_mass0)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_real_sequences_and_get_max_difference)(
            NS(Particles_get_beta0)( max_diff ), &max_diff_indices[ 2 ],
            NS(Particles_get_const_beta0)( lhs ),
            NS(Particles_get_const_beta0)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_real_sequences_and_get_max_difference)(
            NS(Particles_get_gamma0)( max_diff ), &max_diff_indices[ 3 ],
            NS(Particles_get_const_gamma0)( lhs ),
            NS(Particles_get_const_gamma0)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_real_sequences_and_get_max_difference)(
            NS(Particles_get_p0c)( max_diff ), &max_diff_indices[ 4 ],
            NS(Particles_get_const_p0c)( lhs ),
            NS(Particles_get_const_p0c)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_real_sequences_and_get_max_difference)(
            NS(Particles_get_s)( max_diff ), &max_diff_indices[ 5 ],
            NS(Particles_get_const_s)( lhs ),
            NS(Particles_get_const_s)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_real_sequences_and_get_max_difference)(
            NS(Particles_get_x)( max_diff ), &max_diff_indices[ 6 ],
            NS(Particles_get_const_x)( lhs ),
            NS(Particles_get_const_x)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_real_sequences_and_get_max_difference)(
            NS(Particles_get_y)( max_diff ), &max_diff_indices[ 7 ],
            NS(Particles_get_const_y)( lhs ),
            NS(Particles_get_const_y)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_real_sequences_and_get_max_difference)(
            NS(Particles_get_px)( max_diff ), &max_diff_indices[ 8 ],
            NS(Particles_get_const_px)( lhs ),
            NS(Particles_get_const_px)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_real_sequences_and_get_max_difference)(
            NS(Particles_get_py)( max_diff ), &max_diff_indices[ 9 ],
            NS(Particles_get_const_py)( lhs ),
            NS(Particles_get_const_py)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_real_sequences_and_get_max_difference)(
            NS(Particles_get_sigma)( max_diff ), &max_diff_indices[ 10 ],
            NS(Particles_get_const_sigma)( lhs ),
            NS(Particles_get_const_sigma)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_real_sequences_and_get_max_difference)(
            NS(Particles_get_psigma)( max_diff ), &max_diff_indices[ 11 ],
            NS(Particles_get_const_psigma)( lhs ),
            NS(Particles_get_const_psigma)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_real_sequences_and_get_max_difference)(
            NS(Particles_get_delta)( max_diff ), &max_diff_indices[ 12 ],
            NS(Particles_get_const_delta)( lhs ),
            NS(Particles_get_const_delta)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_real_sequences_and_get_max_difference)(
            NS(Particles_get_rpp)( max_diff ), &max_diff_indices[ 13 ],
            NS(Particles_get_const_rpp)( lhs ),
            NS(Particles_get_const_rpp)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_real_sequences_and_get_max_difference)(
            NS(Particles_get_rvv)( max_diff ), &max_diff_indices[ 14 ],
            NS(Particles_get_const_rvv)( lhs ),
            NS(Particles_get_const_rvv)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_real_sequences_and_get_max_difference)(
            NS(Particles_get_chi)( max_diff ), &max_diff_indices[ 15 ],
            NS(Particles_get_const_chi)( lhs ),
            NS(Particles_get_const_chi)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_int64_sequences_and_get_max_difference)(
            NS(Particles_get_particle_id)( max_diff ), &max_diff_indices[ 16 ],
            NS(Particles_get_const_particle_id)( lhs ),
            NS(Particles_get_const_particle_id)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        NS(compare_int64_sequences_and_get_max_difference)(
            NS(Particles_get_lost_at_element_id)( max_diff ),
            &max_diff_indices[ 17 ],
            NS(Particles_get_const_lost_at_element_id)( lhs ),
            NS(Particles_get_const_lost_at_element_id)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_int64_sequences_and_get_max_difference)(
            NS(Particles_get_lost_at_turn)( max_diff ),
            &max_diff_indices[ 18 ],
            NS(Particles_get_const_lost_at_turn)( lhs ),
            NS(Particles_get_const_lost_at_turn)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_int64_sequences_and_get_max_difference)(
            NS(Particles_get_state)( max_diff ), &max_diff_indices[ 19 ],
            NS(Particles_get_const_state)( lhs ),
            NS(Particles_get_const_state)( rhs ), NUM_PARTICLES );
    }

    return;
}

void NS(Particles_print)(
    FILE* SIXTRL_RESTRICT fp,
    const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    NS(block_size_t) const num_particles =
        NS(Particles_get_num_particles)( particles );

    if( ( fp != 0 ) && ( particles != 0 ) && ( num_particles > 0u ) )
    {
        NS(block_size_t) ii = 0u;

        for( ; ii < num_particles ; ++ii )
        {
            if( num_particles > 1u )
            {
                fprintf( fp, "particle id    = %8lu\r\n", ii );
            }

            fprintf( fp, "q0             = %.16f\r\n",
                     NS(Particles_get_q0_value)( particles, ii ) );

            fprintf( fp, "mass0          = %.16f\r\n",
                     NS(Particles_get_mass0_value)( particles, ii ) );

            fprintf( fp, "beta0          = %.16f\r\n",
                     NS(Particles_get_beta0_value)( particles, ii ) );

            fprintf( fp, "gamma0         = %.16f\r\n",
                     NS(Particles_get_gamma0_value)( particles, ii ) );

            fprintf( fp, "p0c            = %.16f\r\n",
                     NS(Particles_get_p0c_value)( particles, ii ) );

            fprintf( fp, "s              = %.16f\r\n",
                     NS(Particles_get_s_value)( particles, ii ) );

            fprintf( fp, "x              = %.16f\r\n",
                     NS(Particles_get_x_value)( particles, ii ) );

            fprintf( fp, "y              = %.16f\r\n",
                     NS(Particles_get_y_value)( particles, ii ) );

            fprintf( fp, "px             = %.16f\r\n",
                     NS(Particles_get_px_value)( particles, ii ) );

            fprintf( fp, "py             = %.16f\r\n",
                     NS(Particles_get_py_value)( particles, ii ) );

            fprintf( fp, "sigma          = %.16f\r\n",
                     NS(Particles_get_sigma_value)( particles, ii ) );

            fprintf( fp, "psigma         = %.16f\r\n",
                     NS(Particles_get_psigma_value)( particles, ii ) );

            fprintf( fp, "delta          = %.16f\r\n",
                     NS(Particles_get_delta_value)( particles, ii ) );

            fprintf( fp, "rpp            = %.16f\r\n",
                     NS(Particles_get_rpp_value)( particles, ii ) );

            fprintf( fp, "rvv            = %.16f\r\n",
                     NS(Particles_get_rvv_value)( particles, ii ) );

            fprintf( fp, "chi            = %.16f\r\n",
                     NS(Particles_get_chi_value)( particles, ii ) );

            fprintf( fp, "particle_id    = %18ld\r\n",
                     NS(Particles_get_particle_id_value)( particles, ii ) );

            fprintf( fp, "lost @ elem_id = %18ld\r\n",
                     NS(Particles_get_lost_at_element_id_value)(
                        particles, ii ) );

            fprintf( fp, "lost @ turn    = %18ld\r\n",
                     NS(Particles_get_lost_at_turn_value)( particles, ii ) );

            fprintf( fp, "state          = %18ld\r\n\r\n",
                     NS(Particles_get_state_value)( particles, ii ) );
        }
    }

    return;
}

void NS(Particles_print_max_diff)(
    FILE* SIXTRL_RESTRICT fp,
    const NS(Particles) *const SIXTRL_RESTRICT max_diff,
    NS(block_size_t) const* max_diff_indices )
{
    NS(block_size_t) const num_particles =
        NS(Particles_get_num_particles)( max_diff );

    NS(block_size_t) const dummy_max_diff_indices[ 20 ] =
    {
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
    };

    if( max_diff_indices == 0 )
    {
        max_diff_indices = &dummy_max_diff_indices[ 0 ];
    }

    if( ( fp != 0 ) && ( max_diff != 0 ) && ( num_particles >= 1u ) )
    {
        fprintf( fp, "Delta |q0|             = %.16e  "
					 "max diff at index = %8lu\r\n",
                    NS(Particles_get_q0_value)( max_diff, 0 ),
                    max_diff_indices[  0 ] );

        fprintf( fp, "Delta |mass0|          = %.16e  "
					 "max diff at index = %8lu\r\n",
                    NS(Particles_get_mass0_value)( max_diff, 0 ),
                    max_diff_indices[  1 ]  );

        fprintf( fp, "Delta |beta0|          = %.16e  "
					 "max diff at index = %8lu\r\n",
                    NS(Particles_get_beta0_value)( max_diff, 0 ),
                    max_diff_indices[  2 ]  );

        fprintf( fp, "Delta |gamma0|         = %.16e  "
					 "max diff at index = %8lu\r\n",
                    NS(Particles_get_gamma0_value)( max_diff, 0 ),
                    max_diff_indices[  3 ]  );

        fprintf( fp, "Delta |p0c|            = %.16e  "
					 "max diff at index = %8lu\r\n",
                    NS(Particles_get_p0c_value)( max_diff, 0 ),
                    max_diff_indices[  4 ]  );

        fprintf( fp, "Delta |s|              = %.16e  "
					 "max diff at index = %8lu\r\n",
                    NS(Particles_get_s_value)( max_diff, 0 ),
                    max_diff_indices[  5 ]  );

        fprintf( fp, "Delta |x|              = %.16e  "
					 "max diff at index = %8lu\r\n",
                    NS(Particles_get_x_value)( max_diff, 0 ),
                    max_diff_indices[  6 ]  );

        fprintf( fp, "Delta |y|              = %.16e  "
					 "max diff at index = %8lu\r\n",
                    NS(Particles_get_y_value)( max_diff, 0 ),
                    max_diff_indices[  7 ]  );

        fprintf( fp, "Delta |px|             = %.16e  "
					 "max diff at index = %8lu\r\n",
                    NS(Particles_get_px_value)( max_diff, 0 ),
                    max_diff_indices[  8 ]  );

        fprintf( fp, "Delta |py|             = %.16e  "
					 "max diff at index = %8lu\r\n",
                    NS(Particles_get_py_value)( max_diff, 0 ),
                    max_diff_indices[  9 ]  );

        fprintf( fp, "Delta |sigma|          = %.16e  "
					 "max diff at index = %8lu\r\n",
                    NS(Particles_get_sigma_value)( max_diff, 0 ),
                    max_diff_indices[ 10 ]  );

        fprintf( fp, "Delta |psigma|         = %.16e  "
					 "max diff at index = %8lu\r\n",
                    NS(Particles_get_psigma_value)( max_diff, 0 ),
                    max_diff_indices[ 11 ]  );

        fprintf( fp, "Delta |delta|          = %.16e  "
					 "max diff at index = %8lu\r\n",
                    NS(Particles_get_delta_value)( max_diff, 0 ),
                    max_diff_indices[ 12 ]  );

        fprintf( fp, "Delta |rpp|            = %.16e  "
					 "max diff at index = %8lu\r\n",
                    NS(Particles_get_rpp_value)( max_diff, 0 ),
                    max_diff_indices[ 13 ]  );

        fprintf( fp, "Delta |rvv|            = %.16e  "
					 "max diff at index = %8lu\r\n",
                    NS(Particles_get_rvv_value)( max_diff, 0 ),
                    max_diff_indices[ 14 ]  );

        fprintf( fp, "Delta |chi|            = %.16e  "
                     "max diff at index = %8lu\r\n",
                    NS(Particles_get_chi_value)( max_diff, 0 ),
                    max_diff_indices[ 15 ]  );

        fprintf( fp, "Delta |particle_id|    = %22ld  "
                     "max diff at index = %8lu\r\n",
                    NS(Particles_get_particle_id_value)( max_diff, 0 ),
                    max_diff_indices[ 16 ]  );

        fprintf( fp, "Delta |lost| @ elem_id = %22ld  "
                     "max diff at index = %8lu\r\n",
                    NS(Particles_get_lost_at_element_id_value)( max_diff, 0 ),
                    max_diff_indices[ 17 ]  );

        fprintf( fp, "Delta |lost| @ turn    = %22ld  "
                     "max diff at index = %8lu\r\n",
                    NS(Particles_get_lost_at_turn_value)( max_diff, 0 ),
                    max_diff_indices[ 18 ]  );

        fprintf( fp, "Delta |state|          = %22ld  "
                     "max diff at index = %8lu\r\n\r\n",
                    NS(Particles_get_state_value)( max_diff, 0 ),
                    max_diff_indices[ 19 ]  );
    }

    return;
}

/* ------------------------------------------------------------------------- */

int NS(Particles_buffers_have_same_structure)(
    const NS(Blocks) *const SIXTRL_RESTRICT lhs_buffer,
    const NS(Blocks) *const SIXTRL_RESTRICT rhs_buffer )
{
    int has_same_structure = 0;

    if( ( lhs_buffer != 0 ) &&
        ( rhs_buffer != 0 ) &&
        ( NS(Blocks_are_serialized)( lhs_buffer ) ) &&
        ( NS(Blocks_are_serialized)( rhs_buffer ) ) &&
        ( NS(Blocks_get_num_of_blocks)( lhs_buffer ) ==
          NS(Blocks_get_num_of_blocks)( rhs_buffer ) ) )
    {
        NS(BlockInfo) const* lhs_it  =
            NS(Blocks_get_const_block_infos_begin)( lhs_buffer );

        NS(BlockInfo) const* lhs_end =
            NS(Blocks_get_const_block_infos_end)( lhs_buffer );

        NS(BlockInfo) const* rhs_it  =
            NS(Blocks_get_const_block_infos_begin)( rhs_buffer );

        if( ( ( lhs_it != 0 ) && ( lhs_end != 0 ) && ( rhs_it != 0 ) ) ||
            ( ( lhs_it == 0 ) && ( lhs_end == 0 ) && ( rhs_it == 0 ) ) )
        {
            has_same_structure = 1;

            for( ; lhs_it != lhs_end ; ++lhs_it, ++rhs_it )
            {
                NS(Particles) const* lhs_particles =
                    NS(Blocks_get_const_particles)( lhs_it );

                NS(Particles) const* rhs_particles =
                    NS(Blocks_get_const_particles)( rhs_it );

                if( ( lhs_particles == 0 ) || ( rhs_particles == 0 ) ||
                    ( !NS(Particles_have_same_structure)(
                        lhs_particles, rhs_particles ) ) )
                {
                    has_same_structure = 0;
                    break;
                }
            }
        }
    }

    return has_same_structure;
}

int NS(Particles_buffers_map_to_same_memory)(
    const NS(Blocks) *const SIXTRL_RESTRICT lhs_buffer,
    const NS(Blocks) *const SIXTRL_RESTRICT rhs_buffer )
{
    int maps_to_same_memory = 0;

    if( ( lhs_buffer != 0 ) &&
        ( rhs_buffer != 0 ) &&
        ( NS(Blocks_are_serialized)( lhs_buffer ) ) &&
        ( NS(Blocks_are_serialized)( rhs_buffer ) ) &&
        ( NS(Blocks_get_num_of_blocks)( lhs_buffer ) ==
          NS(Blocks_get_num_of_blocks)( rhs_buffer ) ) )
    {
        NS(BlockInfo) const* lhs_it  =
            NS(Blocks_get_const_block_infos_begin)( lhs_buffer );

        NS(BlockInfo) const* lhs_end =
            NS(Blocks_get_const_block_infos_end)( lhs_buffer );

        NS(BlockInfo) const* rhs_it  =
            NS(Blocks_get_const_block_infos_begin)( rhs_buffer );

        if( ( lhs_it != 0 ) && ( lhs_end != 0 ) && ( rhs_it != 0 ) )
        {
            maps_to_same_memory = 1;

            for( ; lhs_it != lhs_end ; ++lhs_it, ++rhs_it )
            {
                NS(Particles) const* lhs_particles =
                    NS(Blocks_get_const_particles)( lhs_it );

                NS(Particles) const* rhs_particles =
                    NS(Blocks_get_const_particles)( rhs_it );

                if( ( lhs_particles == 0 ) || ( rhs_particles == 0 ) ||
                    ( !NS(Particles_map_to_same_memory)(
                        lhs_particles, rhs_particles ) ) )
                {
                    maps_to_same_memory = 0;
                    break;
                }
            }
        }
    }

    return maps_to_same_memory;
}

int NS(Particles_buffer_compare_values)(
    const NS(Blocks) *const SIXTRL_RESTRICT lhs_buffer,
    const NS(Blocks) *const SIXTRL_RESTRICT rhs_buffer )
{
    int cmp_result = -1;

    if( ( lhs_buffer != 0 ) &&
        ( rhs_buffer != 0 ) &&
        ( NS(Blocks_are_serialized)( lhs_buffer ) ) &&
        ( NS(Blocks_are_serialized)( rhs_buffer ) ) &&
        ( NS(Blocks_get_num_of_blocks)( lhs_buffer ) ==
          NS(Blocks_get_num_of_blocks)( rhs_buffer ) ) )
    {
        NS(BlockInfo) const* lhs_it  =
            NS(Blocks_get_const_block_infos_begin)( lhs_buffer );

        NS(BlockInfo) const* lhs_end =
            NS(Blocks_get_const_block_infos_end)( lhs_buffer );

        NS(BlockInfo) const* rhs_it  =
            NS(Blocks_get_const_block_infos_begin)( rhs_buffer );

        if( ( lhs_it != 0 ) && ( lhs_end != 0 ) && ( rhs_it != 0 ) )
        {
            cmp_result = 0;

            for( ; lhs_it != lhs_end ; ++lhs_it, ++rhs_it )
            {
                NS(Particles) const* lhs_particles =
                    NS(Blocks_get_const_particles)( lhs_it );

                NS(Particles) const* rhs_particles =
                    NS(Blocks_get_const_particles)( rhs_it );

                cmp_result = NS(Particles_compare_values)(
                    lhs_particles, rhs_particles );

                if( cmp_result != 0 ) break;
            }
        }
    }

    return cmp_result;
}

int NS(Particles_buffer_compare_values_with_treshold)(
    const NS(Blocks) *const SIXTRL_RESTRICT lhs_buffer,
    const NS(Blocks) *const SIXTRL_RESTRICT rhs_buffer,
    SIXTRL_REAL_T const treshold )
{
    int cmp_result = -1;

    if( ( lhs_buffer != 0 ) &&
        ( rhs_buffer != 0 ) &&
        ( NS(Blocks_are_serialized)( lhs_buffer ) ) &&
        ( NS(Blocks_are_serialized)( rhs_buffer ) ) &&
        ( NS(Blocks_get_num_of_blocks)( lhs_buffer ) ==
          NS(Blocks_get_num_of_blocks)( rhs_buffer ) ) )
    {
        NS(BlockInfo) const* lhs_it  =
            NS(Blocks_get_const_block_infos_begin)( lhs_buffer );

        NS(BlockInfo) const* lhs_end =
            NS(Blocks_get_const_block_infos_end)( lhs_buffer );

        NS(BlockInfo) const* rhs_it  =
            NS(Blocks_get_const_block_infos_begin)( rhs_buffer );

        if( ( lhs_it != 0 ) && ( lhs_end != 0 ) && ( rhs_it != 0 ) )
        {
            cmp_result = 0;

            for( ; lhs_it != lhs_end ; ++lhs_it, ++rhs_it )
            {
                NS(Particles) const* lhs_particles =
                    NS(Blocks_get_const_particles)( lhs_it );

                NS(Particles) const* rhs_particles =
                    NS(Blocks_get_const_particles)( rhs_it );

                cmp_result = NS(Particles_compare_values_with_treshold)(
                    lhs_particles, rhs_particles, treshold );

                if( cmp_result != 0 ) break;
            }
        }
    }

    return cmp_result;
}

void NS(Particles_buffer_get_max_difference)(
    NS(Blocks)* SIXTRL_RESTRICT max_diff_buffer,
    NS(block_size_t)* SIXTRL_RESTRICT max_diff_indices,
    const NS(Blocks) *const SIXTRL_RESTRICT lhs_buffer,
    const NS(Blocks) *const SIXTRL_RESTRICT rhs_buffer )
{
    if( ( lhs_buffer != 0 ) &&
        ( rhs_buffer != 0 ) &&
        ( max_diff_buffer != 0 ) &&
        ( NS(Blocks_are_serialized)( lhs_buffer ) ) &&
        ( NS(Blocks_are_serialized)( rhs_buffer ) ) &&
        ( NS(Blocks_are_serialized)( max_diff_buffer ) ) &&
        ( NS(Blocks_get_num_of_blocks)( lhs_buffer ) ==
          NS(Blocks_get_num_of_blocks)( rhs_buffer ) ) &&
        ( NS(Blocks_get_num_of_blocks)( max_diff_buffer ) ==
          NS(Blocks_get_num_of_blocks)( lhs_buffer ) ) )
    {
        NS(BlockInfo) const* lhs_it  =
            NS(Blocks_get_const_block_infos_begin)( lhs_buffer );

        NS(BlockInfo) const* lhs_end =
            NS(Blocks_get_const_block_infos_end)( lhs_buffer );

        NS(BlockInfo) const* rhs_it  =
            NS(Blocks_get_const_block_infos_begin)( rhs_buffer );

        NS(BlockInfo)* max_diff_it =
            NS(Blocks_get_block_infos_begin)( max_diff_buffer );

        if( ( lhs_it != 0 ) && ( lhs_end != 0 ) && ( rhs_it != 0 ) )
        {
            for( ; lhs_it != lhs_end ; ++lhs_it, ++rhs_it )
            {
                NS(Particles) const* lhs_particles =
                    NS(Blocks_get_const_particles)( lhs_it );

                NS(Particles) const* rhs_particles =
                    NS(Blocks_get_const_particles)( rhs_it );

                NS(Particles)* max_diff =
                    NS(Blocks_get_particles)( max_diff_it );

                NS(Particles_get_max_difference)(
                    max_diff, max_diff_indices, lhs_particles, rhs_particles );

                if( max_diff_indices != 0 )
                {
                    max_diff_indices = max_diff_indices + 20;
                }
            }
        }

        return;
    }
}

void NS(Particles_buffer_print)(
    FILE* SIXTRL_RESTRICT fp,
    const NS(Blocks) *const SIXTRL_RESTRICT particles_buffer )
{
    if( ( fp != 0 ) && ( particles_buffer != 0 ) &&
        ( NS(Blocks_are_serialized)( particles_buffer ) ) )
    {
        NS(block_size_t) const num_of_particle_blocks =
            NS(Blocks_get_num_of_blocks)( particles_buffer );

        NS(BlockInfo) const* block_it  =
            NS(Blocks_get_const_block_infos_begin)( particles_buffer );

        NS(BlockInfo) const* block_end =
            NS(Blocks_get_const_block_infos_end)( particles_buffer );

        NS(block_size_t) ii = 0u;

        for( ; block_it != block_end ; ++block_it, ++ii )
        {
            NS(Particles) const* particles =
                NS(Blocks_get_const_particles)( block_it );

            if( num_of_particle_blocks > 1u )
            {
                fprintf(
                    fp, "-------------------------------------------------"
                        "-------------------------------------------------"
                        "------------------------\r\n" );

               fprintf( fp, "particle block index = %8lu\r\n", ii );
            }

            NS(Particles_print)( fp, particles );
        }
    }

    return;
}

void NS(Particles_buffer_print_max_diff)(
    FILE* SIXTRL_RESTRICT fp,
    const NS(Blocks) *const SIXTRL_RESTRICT max_diff_buffer,
    NS(block_size_t) const* max_diff_indices )
{
    if( ( fp != 0 ) && ( max_diff_buffer != 0 ) &&
        ( NS(Blocks_are_serialized)( max_diff_buffer ) ) )
    {
        NS(block_size_t) const num_of_particle_blocks =
            NS(Blocks_get_num_of_blocks)( max_diff_buffer );

        NS(BlockInfo) const* block_it  =
            NS(Blocks_get_const_block_infos_begin)( max_diff_buffer );

        NS(BlockInfo) const* block_end =
            NS(Blocks_get_const_block_infos_end)( max_diff_buffer );

        NS(block_size_t) ii = 0u;

        for( ; block_it != block_end ; ++block_it, ++ii )
        {
            NS(Particles) const* max_diff =
                NS(Blocks_get_const_particles)( block_it );

            if( num_of_particle_blocks > 1u )
            {
                fprintf(
                    fp, "-------------------------------------------------"
                        "-------------------------------------------------"
                        "------------------------\r\n" );

               fprintf( fp, "particle block index = %8lu\r\n", ii );
            }

            NS(Particles_print_max_diff)( fp, max_diff, max_diff_indices );

            if( max_diff_indices != 0 )
            {
                max_diff_indices = max_diff_indices + 20;
            }
        }
    }

    return;
}

/* ------------------------------------------------------------------------- */

/* end: tests/sixtracklib/testlib/details/test_particles_tools.c */
