#include "sixtracklib/testlib/test_particles_tools.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/testlib/random.h"

extern void NS(Particles_random_init)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p );

extern void NS(Particles_realistic_init)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p );

/* ------------------------------------------------------------------------- */

extern int NS(Particles_have_same_structure)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs );

extern int NS(Particles_map_to_same_memory)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs );

extern int NS(Particles_compare_values)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs );

extern int NS(Particles_compare_values_with_treshold)(
    SIXTRL_ARGPTR_DEC   const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC   const NS(Particles) *const SIXTRL_RESTRICT rhs,
    NS(particle_real_t) const treshold );

static int NS(compare_sequences_exact)(
    SIXTRL_DATAPTR_DEC void const* SIXTRL_RESTRICT lhs_values,
    SIXTRL_DATAPTR_DEC void const* SIXTRL_RESTRICT rhs_values,
    NS(buffer_size_t) const num_values,
    NS(buffer_size_t) const element_size );

static int NS(compare_real_sequences_with_treshold)(
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT lhs_values,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT rhs_values,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_first_out_of_bounds_index,
    NS(buffer_size_t) const num_values,
    NS(particle_real_t) const treshold );

static void NS(compare_real_sequences_and_get_max_difference)(
    NS(particle_real_ptr_t)    SIXTRL_RESTRICT ptr_max_diff,
    NS(buffer_size_t)*         SIXTRL_RESTRICT ptr_max_diff_index,
    NS(particle_real_t) const* SIXTRL_RESTRICT lhs_values,
    NS(particle_real_t) const* SIXTRL_RESTRICT rhs_values,
    NS(buffer_size_t)   const  num_values );

static void NS(compare_int64_sequences_and_get_max_difference)(
    NS(particle_index_ptr_t) SIXTRL_RESTRICT ptr_max_diff,
    NS(buffer_size_t)*       SIXTRL_RESTRICT ptr_max_diff_index,
    NS(particle_index_const_ptr_t) SIXTRL_RESTRICT lhs_values,
    NS(particle_index_const_ptr_t) SIXTRL_RESTRICT rhs_values,
    NS(buffer_size_t) const num_values );

extern void NS(Particles_get_max_difference)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT max_diff,
    NS(buffer_size_t)* SIXTRL_RESTRICT max_diff_indices,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs );



extern void NS(Particles_print)(
    FILE* SIXTRL_RESTRICT fp,
    const NS(Particles) *const SIXTRL_RESTRICT particles );

extern void NS(Particles_print_max_diff)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT max_diff,
    NS(buffer_size_t) const* max_diff_indices );

/* ------------------------------------------------------------------------- */

extern int NS(Particles_buffers_map_to_same_memory)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs_buffer,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs_buffer );

extern int NS(Particles_buffers_compare_values)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs_buffer,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs_buffer );

extern int NS(Particles_buffers_compare_values_with_treshold)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs_buffer,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs_buffer,
    NS(particle_real_t) const treshold );

extern void NS(Particles_buffer_get_max_difference)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT max_diff,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT max_diff_indices,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs );

extern void NS(Particles_buffer_print_stdout)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT particle_buffer );

extern void NS(Particles_buffer_print)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT particle_buffer );

extern void NS(Particles_buffer_print_max_diff)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT max_diff_buffer,
    NS(buffer_size_t) const* max_diff_indices );


/* ------------------------------------------------------------------------- */

int NS(compare_sequences_exact)(
    SIXTRL_DATAPTR_DEC void const* SIXTRL_RESTRICT lhs_values,
    SIXTRL_DATAPTR_DEC void const* SIXTRL_RESTRICT rhs_values,
    NS(buffer_size_t) const num_values,
    NS(buffer_size_t) const element_size )
{
    int cmp_result = -1;

    NS(buffer_size_t) const ATTR_LENGTH = num_values * element_size;

    if( ATTR_LENGTH > ( NS(buffer_size_t) )0u )
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
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT lhs_values,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT rhs_values,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_first_out_of_bounds_index,
    NS(buffer_size_t) const num_values,
    NS(particle_real_t) const treshold )
{
    int cmp_result = -1;

    if( ( lhs_values != 0 )  && ( rhs_values != 0 ) )
    {
        static NS(particle_real_t) const ZERO = ( NS(particle_real_t) )0.0L;
        NS(buffer_size_t) ii = 0;

        cmp_result = 0;

        for( ; ii < num_values ; ++ii )
        {
            NS(particle_real_t) const diff = lhs_values[ ii ] - rhs_values[ ii ];

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
    NS(particle_real_ptr_t) SIXTRL_RESTRICT ptr_max_diff,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_max_diff_index,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT lhs_values,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT rhs_values,
    NS(buffer_size_t) const num_values )
{
    static NS(particle_real_t) const ZERO = ( NS(particle_real_t) )0.0L;

    NS(buffer_size_t) max_diff_index = 0u;

    NS(particle_real_t) cmp_max_diff = ZERO;
    NS(particle_real_t) max_diff     = ZERO;

    if( ( lhs_values != 0  ) && ( rhs_values   != 0 ) &&
        ( num_values >  0u ) && ( ptr_max_diff != 0 ) )
    {
        NS(buffer_size_t) ii = 0u;

        for( ; ii < num_values ; ++ii )
        {
            NS(particle_real_t) const diff = lhs_values[ ii ] - rhs_values[ ii ];
            NS(particle_real_t) const cmp_diff = ( diff > ZERO ) ? diff : -diff;

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
        NS(particle_real_const_ptr_t) values =
            ( lhs_values != 0 ) ? lhs_values : rhs_values;

        NS(buffer_size_t) ii = 0u;

        for( ; ii < num_values ; ++ii )
        {
            NS(particle_real_t) const diff = values[ ii ];
            NS(particle_real_t) const cmp_diff = ( diff > ZERO ) ? diff : -diff;

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
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_max_diff_index,
    SIXTRL_INT64_T const* SIXTRL_RESTRICT lhs_values,
    SIXTRL_INT64_T const* SIXTRL_RESTRICT rhs_values,
    NS(buffer_size_t) const num_values )
{
    static SIXTRL_INT64_T const ZERO = ( SIXTRL_INT64_T )0u;

    NS(buffer_size_t) max_diff_index = 0u;
    SIXTRL_INT64_T max_diff = ZERO;
    SIXTRL_INT64_T cmp_max_diff = ZERO;

    if( ( lhs_values != 0  ) && ( rhs_values   != 0 ) &&
        ( num_values >  0u ) && ( ptr_max_diff != 0 ) )
    {
        NS(buffer_size_t) ii = 0u;

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

        NS(buffer_size_t) ii = 0u;

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
    SIXTRL_SIZE_T const NUM_PARTICLES = NS(Particles_get_num_of_particles)( p );

    if( ( p != 0 ) && ( NUM_PARTICLES > ( SIXTRL_SIZE_T )0u ) )
    {
        SIXTRL_SIZE_T ii;
        SIXTRL_INT64_T particle_id = 0;

        NS(particle_real_t) const Q0     = ( NS(particle_real_t) )1.0;
        NS(particle_real_t) const MASS0  = ( NS(particle_real_t) )938272013.0;
        NS(particle_real_t) const BETA0  = ( NS(particle_real_t) )0.9999999895816052;
        NS(particle_real_t) const GAMMA0 = ( NS(particle_real_t) )6927.628566067013;
        NS(particle_real_t) const P0C    = ( NS(particle_real_t) )6499999932280.434;

        NS(particle_real_t) const S      = ( NS(particle_real_t) )0.0;
        NS(particle_real_t) const X0     = ( NS(particle_real_t) )0.0007145746958428162;
        NS(particle_real_t) const Y      = ( NS(particle_real_t) )0.0006720936794100408;
        NS(particle_real_t) const PX     = ( NS(particle_real_t) )-1.7993573732749124e-05;
        NS(particle_real_t) const PY     = ( NS(particle_real_t) )7.738322282156584e-06;
        NS(particle_real_t) const ZETA   = ( NS(particle_real_t) )1.70026098624819983075e-05;
        NS(particle_real_t) const PSIGMA = ( NS(particle_real_t) )0.00027626172996313825;
        NS(particle_real_t) const DELTA  = ( NS(particle_real_t) )0.00027626172996228097;
        NS(particle_real_t) const RPP    = ( NS(particle_real_t) )0.9997238145695025;
        NS(particle_real_t) const RVV    = ( NS(particle_real_t) )0.999999999994246;
        NS(particle_real_t) const CHI    = ( NS(particle_real_t) )1.0;

        NS(particle_real_t) const DELTA_X = ( NS(particle_real_t) )1e-9;

        for( ii = 0 ; ii < NUM_PARTICLES ; ++ii, ++particle_id )
        {
            NS(particle_real_t) const X = X0 + DELTA_X * ii;

            NS(Particles_set_q0_value)(            p, ii, Q0 );
            NS(Particles_set_mass0_value)(         p, ii, MASS0 );
            NS(Particles_set_beta0_value)(         p, ii, BETA0 );
            NS(Particles_set_gamma0_value)(        p, ii, GAMMA0 );
            NS(Particles_set_p0c_value)(           p, ii, P0C );

            NS(Particles_set_particle_id_value)(   p, ii, particle_id );
            NS(Particles_set_at_element_id_value)( p, ii, -1 );
            NS(Particles_set_at_turn_value)(       p, ii, -1 );
            NS(Particles_set_state_value)(         p, ii,  0 );

            NS(Particles_set_s_value)(             p, ii, S );
            NS(Particles_set_x_value)(             p, ii, X );
            NS(Particles_set_y_value)(             p, ii, Y );
            NS(Particles_set_px_value)(            p, ii, PX );
            NS(Particles_set_py_value)(            p, ii, PY );
            NS(Particles_set_zeta_value)(          p, ii, ZETA );

            NS(Particles_set_psigma_value)(        p, ii, PSIGMA );
            NS(Particles_set_delta_value)(         p, ii, DELTA );
            NS(Particles_set_rpp_value)(           p, ii, RPP );
            NS(Particles_set_rvv_value)(           p, ii, RVV );
            NS(Particles_set_chi_value)(           p, ii, CHI );
        }
    }

    return;
}

void NS(Particles_random_init)( NS(Particles)* SIXTRL_RESTRICT p )
{
    SIXTRL_SIZE_T const NUM_PARTICLES = NS(Particles_get_num_of_particles)( p );

    if( ( p != 0 ) && ( NUM_PARTICLES > ( SIXTRL_SIZE_T )0u ) )
    {
        NS(particle_real_t) const TWO_PI  = 2.0 * M_PI;
        NS(particle_real_t) const Q0      = 1.0;
        NS(particle_real_t) const MASS0   = 1.0;
        NS(particle_real_t) const BETA0   = 1.0;
        NS(particle_real_t) const GAMMA0  = 1.0;
        NS(particle_real_t) const P0C     = 1.0;

        NS(particle_real_t) const S       = 0.0;
        NS(particle_real_t) const MIN_X   = 0.0;
        NS(particle_real_t) const MAX_X   = 0.2;
        NS(particle_real_t) const DELTA_X = ( MAX_X - MIN_X );

        NS(particle_real_t) const MIN_Y   = 0.0;
        NS(particle_real_t) const MAX_Y   = 0.3;
        NS(particle_real_t) const DELTA_Y = ( MAX_Y - MIN_Y );

        NS(particle_real_t) const P       = 0.1;
        NS(particle_real_t) const ZETA    = 0.0;

        NS(particle_real_t) const PSIGMA  = 0.0;
        NS(particle_real_t) const RPP     = 1.0;
        NS(particle_real_t) const RVV     = 1.0;
        NS(particle_real_t) const DELTA   = 0.5;
        NS(particle_real_t) const CHI     = 0.0;

        int64_t const STATE  = INT64_C( 0 );

        size_t ii;
        int64_t particle_id = 0;

        for( ii = 0 ; ii < NUM_PARTICLES ; ++ii, ++particle_id )
        {
            NS(particle_real_t) const ANGLE = NS(Random_genrand64_real1)() * TWO_PI;
            NS(particle_real_t) const PX = P * cos( ANGLE );
            NS(particle_real_t) const PY = sqrt( P * P - PX * PX );
            NS(particle_real_t) const X  = MIN_X + NS(Random_genrand64_real1)() * DELTA_X;
            NS(particle_real_t) const Y  = MIN_Y + NS(Random_genrand64_real1)() * DELTA_Y;


            NS(Particles_set_q0_value)(            p, ii, Q0 );
            NS(Particles_set_mass0_value)(         p, ii, MASS0 );
            NS(Particles_set_beta0_value)(         p, ii, BETA0 );
            NS(Particles_set_gamma0_value)(        p, ii, GAMMA0 );
            NS(Particles_set_p0c_value)(           p, ii, P0C );

            NS(Particles_set_particle_id_value)(   p, ii, particle_id );
            NS(Particles_set_at_element_id_value)( p, ii, -1 );
            NS(Particles_set_at_turn_value)(       p, ii, -1 );
            NS(Particles_set_state_value)(         p, ii, STATE );

            NS(Particles_set_s_value)(             p, ii, S );
            NS(Particles_set_x_value)(             p, ii, X );
            NS(Particles_set_y_value)(             p, ii, Y );
            NS(Particles_set_px_value)(            p, ii, PX );
            NS(Particles_set_py_value)(            p, ii, PY );
            NS(Particles_set_zeta_value)(          p, ii, ZETA );

            NS(Particles_set_psigma_value)(        p, ii, PSIGMA );
            NS(Particles_set_delta_value)(         p, ii, DELTA );
            NS(Particles_set_rpp_value)(           p, ii, RPP );
            NS(Particles_set_rvv_value)(           p, ii, RVV );
            NS(Particles_set_chi_value)(           p, ii, CHI );
        }
    }

    return;
}


int NS(Particles_have_same_structure)(
    const st_Particles *const SIXTRL_RESTRICT lhs,
    const st_Particles *const SIXTRL_RESTRICT rhs )
{
    return ( ( lhs != 0 ) && ( rhs != 0 ) &&
             ( NS(Particles_get_num_of_particles)( lhs ) ==
               NS(Particles_get_num_of_particles)( rhs ) ) )
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
            ( NS(Particles_get_const_zeta)( lhs ) ==
              NS(Particles_get_const_zeta)( rhs ) ) &&
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
            ( NS(Particles_get_const_at_element_id)( lhs ) ==
              NS(Particles_get_const_at_element_id)( rhs ) ) &&
            ( NS(Particles_get_const_at_turn)( lhs ) ==
              NS(Particles_get_const_at_turn)( rhs ) ) &&
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
        NS(buffer_size_t) const NUM_PARTICLES =
            NS(Particles_get_num_of_particles)( lhs );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_q0)( lhs ),
            NS(Particles_get_const_q0)( rhs ),
            NUM_PARTICLES, sizeof( NS(particle_real_t) ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_mass0)( lhs ),
            NS(Particles_get_const_mass0)( rhs ),
            NUM_PARTICLES, sizeof( NS(particle_real_t) ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_beta0)( lhs ),
            NS(Particles_get_const_beta0)( rhs ),
            NUM_PARTICLES, sizeof( NS(particle_real_t) ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_gamma0)( lhs ),
            NS(Particles_get_const_gamma0)( rhs ),
            NUM_PARTICLES, sizeof( NS(particle_real_t) ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_p0c)( lhs ),
            NS(Particles_get_const_p0c)( rhs ),
            NUM_PARTICLES, sizeof( NS(particle_real_t) ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_s)( lhs ),
            NS(Particles_get_const_s)( rhs ),
            NUM_PARTICLES, sizeof( NS(particle_real_t) ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_x)( lhs ),
            NS(Particles_get_const_x)( rhs ),
            NUM_PARTICLES, sizeof( NS(particle_real_t) ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_y)( lhs ),
            NS(Particles_get_const_y)( rhs ),
            NUM_PARTICLES, sizeof( NS(particle_real_t) ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_px)( lhs ),
            NS(Particles_get_const_px)( rhs ),
            NUM_PARTICLES, sizeof( NS(particle_real_t) ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_py)( lhs ),
            NS(Particles_get_const_py)( rhs ),
            NUM_PARTICLES, sizeof( NS(particle_real_t) ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_zeta)( lhs ),
            NS(Particles_get_const_zeta)( rhs ),
            NUM_PARTICLES, sizeof( NS(particle_real_t) ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_psigma)( lhs ),
            NS(Particles_get_const_psigma)( rhs ),
            NUM_PARTICLES, sizeof( NS(particle_real_t) ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_delta)( lhs ),
            NS(Particles_get_const_delta)( rhs ),
            NUM_PARTICLES, sizeof( NS(particle_real_t) ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_rpp)( lhs ),
            NS(Particles_get_const_rpp)( rhs ),
            NUM_PARTICLES, sizeof( NS(particle_real_t) ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_rvv)( lhs ),
            NS(Particles_get_const_rvv)( rhs ),
            NUM_PARTICLES, sizeof( NS(particle_real_t) ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_chi)( lhs ),
            NS(Particles_get_const_chi)( rhs ),
            NUM_PARTICLES, sizeof( NS(particle_real_t) ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_particle_id)( lhs ),
            NS(Particles_get_const_particle_id)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_INT64_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_at_element_id)( lhs ),
            NS(Particles_get_const_at_element_id)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_INT64_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_at_turn)( lhs ),
            NS(Particles_get_const_at_turn)( rhs ),
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
    NS(particle_real_t) const treshold )
{
    int cmp_result = -1;

    if( NS(Particles_have_same_structure)( lhs, rhs ) )
    {
        NS(buffer_size_t) const NUM_PARTICLES =
            NS(Particles_get_num_of_particles)( lhs );

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
            NS(Particles_get_const_zeta)( lhs ),
            NS(Particles_get_const_zeta)( rhs ), 0, NUM_PARTICLES, treshold );

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
            NS(Particles_get_const_at_element_id)( lhs ),
            NS(Particles_get_const_at_element_id)( rhs ),
            NUM_PARTICLES, sizeof( SIXTRL_INT64_T ) );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(compare_sequences_exact)(
            NS(Particles_get_const_at_turn)( lhs ),
            NS(Particles_get_const_at_turn)( rhs ),
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
    NS(buffer_size_t)* SIXTRL_RESTRICT max_diff_indices,
    const NS(Particles) *const SIXTRL_RESTRICT lhs,
    const NS(Particles) *const SIXTRL_RESTRICT rhs )
{
    if( ( max_diff != 0 ) && ( max_diff_indices != 0 ) &&
        ( NS(Particles_have_same_structure)( lhs, rhs ) ) &&
        ( NS(Particles_get_num_of_particles)( lhs ) > 0u ) &&
        ( NS(Particles_get_num_of_particles)( max_diff ) >= 1u ) )
    {
        NS(buffer_size_t) dummy_max_diff_indices[ 20 ] =
        {
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
        };

        NS(buffer_size_t) const NUM_PARTICLES =
            NS(Particles_get_num_of_particles)( lhs );

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
            NS(Particles_get_zeta)( max_diff ), &max_diff_indices[ 10 ],
            NS(Particles_get_const_zeta)( lhs ),
            NS(Particles_get_const_zeta)( rhs ), NUM_PARTICLES );

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
            NS(Particles_get_at_element_id)( max_diff ),
            &max_diff_indices[ 17 ],
            NS(Particles_get_const_at_element_id)( lhs ),
            NS(Particles_get_const_at_element_id)( rhs ), NUM_PARTICLES );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        NS(compare_int64_sequences_and_get_max_difference)(
            NS(Particles_get_at_turn)( max_diff ),
            &max_diff_indices[ 18 ],
            NS(Particles_get_const_at_turn)( lhs ),
            NS(Particles_get_const_at_turn)( rhs ), NUM_PARTICLES );

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
    NS(buffer_size_t) const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    if( ( fp != 0 ) && ( particles != 0 ) && ( num_particles > 0u ) )
    {
        NS(buffer_size_t) ii = 0u;

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

            fprintf( fp, "zeta           = %.16f\r\n",
                     NS(Particles_get_zeta_value)( particles, ii ) );

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

            fprintf( fp, "at_elem_id     = %18ld\r\n",
                     NS(Particles_get_at_element_id_value)( particles, ii ) );

            fprintf( fp, "at_turn        = %18ld\r\n",
                     NS(Particles_get_at_turn_value)( particles, ii ) );

            fprintf( fp, "state          = %18ld\r\n\r\n",
                     NS(Particles_get_state_value)( particles, ii ) );
        }
    }

    return;
}

void NS(Particles_print_max_diff)(
    FILE* SIXTRL_RESTRICT fp,
    const NS(Particles) *const SIXTRL_RESTRICT max_diff,
    NS(buffer_size_t) const* max_diff_indices )
{
    NS(buffer_size_t) const num_particles =
        NS(Particles_get_num_of_particles)( max_diff );

    NS(buffer_size_t) const dummy_max_diff_indices[ 20 ] =
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

        fprintf( fp, "Delta |zeta|           = %.16e  "
					 "max diff at index = %8lu\r\n",
                    NS(Particles_get_zeta_value)( max_diff, 0 ),
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

        fprintf( fp, "Delta |at_elem_id|     = %22ld  "
                     "max diff at index = %8lu\r\n",
                    NS(Particles_get_at_element_id_value)( max_diff, 0 ),
                    max_diff_indices[ 17 ]  );

        fprintf( fp, "Delta |at_turn|        = %22ld  "
                     "max diff at index = %8lu\r\n",
                    NS(Particles_get_at_turn_value)( max_diff, 0 ),
                    max_diff_indices[ 18 ]  );

        fprintf( fp, "Delta |state|          = %22ld  "
                     "max diff at index = %8lu\r\n\r\n",
                    NS(Particles_get_state_value)( max_diff, 0 ),
                    max_diff_indices[ 19 ]  );
    }

    return;
}

/* ------------------------------------------------------------------------- */

int NS(Particles_buffers_map_to_same_memory)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs_buffer,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs_buffer )
{
    int maps_to_same_memory = 0;

    if( ( lhs_buffer != SIXTRL_NULLPTR ) &&
        ( rhs_buffer != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_get_num_of_objects)( lhs_buffer ) ==
          NS(Buffer_get_num_of_objects)( rhs_buffer ) ) )
    {
        typedef NS(Object) info_t;
        typedef SIXTRL_DATAPTR_DEC info_t        const* ptr_to_info_t;
        typedef SIXTRL_DATAPTR_DEC NS(Particles) const* ptr_to_particles_t;

        ptr_to_info_t lhs_it = ( ptr_to_info_t )( uintptr_t
            )NS(Buffer_get_objects_begin_addr)( lhs_buffer );

        ptr_to_info_t lhs_end = ( ptr_to_info_t )( uintptr_t
            )NS(Buffer_get_objects_end_addr)( lhs_buffer );

        ptr_to_info_t rhs_it  = ( ptr_to_info_t )( uintptr_t
            )NS(Buffer_get_objects_begin_addr)( rhs_buffer );

        if( ( ( lhs_it != SIXTRL_NULLPTR ) && ( lhs_end != SIXTRL_NULLPTR ) &&
              ( rhs_it != SIXTRL_NULLPTR ) ) ||
            ( ( lhs_it == SIXTRL_NULLPTR ) && ( lhs_end == SIXTRL_NULLPTR ) &&
              ( rhs_it == SIXTRL_NULLPTR ) ) )
        {
            maps_to_same_memory = 1;

            for( ; lhs_it != lhs_end ; ++lhs_it, ++rhs_it )
            {
                ptr_to_particles_t lhs_particles = ( ptr_to_particles_t )(
                    uintptr_t )NS(BufferIndex_get_const_particles)( lhs_it );

                ptr_to_particles_t rhs_particles = ( ptr_to_particles_t )(
                    uintptr_t )NS(BufferIndex_get_const_particles)( rhs_it );

                if( ( lhs_particles == SIXTRL_NULLPTR ) ||
                    ( rhs_particles == SIXTRL_NULLPTR ) ||
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

int NS(Particles_buffers_compare_values)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs_buffer,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs_buffer )
{
    int cmp_result = -1;

    if( ( lhs_buffer != SIXTRL_NULLPTR ) &&
        ( rhs_buffer != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_get_num_of_objects)( lhs_buffer ) ==
          NS(Buffer_get_num_of_objects)( rhs_buffer ) ) )
    {
        typedef NS(Object) info_t;
        typedef SIXTRL_DATAPTR_DEC info_t        const* ptr_to_info_t;
        typedef SIXTRL_DATAPTR_DEC NS(Particles) const* ptr_to_particles_t;

        ptr_to_info_t lhs_it = ( ptr_to_info_t )( uintptr_t
            )NS(Buffer_get_objects_begin_addr)( lhs_buffer );

        ptr_to_info_t lhs_end = ( ptr_to_info_t )( uintptr_t
            )NS(Buffer_get_objects_end_addr)( lhs_buffer );

        ptr_to_info_t rhs_it  = ( ptr_to_info_t )( uintptr_t
            )NS(Buffer_get_objects_begin_addr)( rhs_buffer );

        if( ( ( lhs_it != SIXTRL_NULLPTR ) && ( lhs_end != SIXTRL_NULLPTR ) &&
              ( rhs_it != SIXTRL_NULLPTR ) ) ||
            ( ( lhs_it == SIXTRL_NULLPTR ) && ( lhs_end == SIXTRL_NULLPTR ) &&
              ( rhs_it == SIXTRL_NULLPTR ) ) )
        {
            cmp_result = 0;

            for( ; lhs_it != lhs_end ; ++lhs_it, ++rhs_it )
            {
                ptr_to_particles_t lhs_particles = ( ptr_to_particles_t )(
                    uintptr_t )NS(BufferIndex_get_const_particles)( lhs_it );

                ptr_to_particles_t rhs_particles = ( ptr_to_particles_t )(
                    uintptr_t )NS(BufferIndex_get_const_particles)( rhs_it );

                cmp_result = NS(Particles_compare_values)(
                    lhs_particles, rhs_particles );

                if( cmp_result != 0 ) break;
            }
        }
    }

    return cmp_result;
}

int NS(Particles_buffers_compare_values_with_treshold)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs_buffer,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs_buffer,
    NS(particle_real_t) const treshold )
{
    int cmp_result = -1;

    if( ( lhs_buffer != SIXTRL_NULLPTR ) &&
        ( rhs_buffer != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_get_num_of_objects)( lhs_buffer ) ==
          NS(Buffer_get_num_of_objects)( rhs_buffer ) ) )
    {
        typedef NS(Object) info_t;
        typedef SIXTRL_DATAPTR_DEC info_t        const* ptr_to_info_t;
        typedef SIXTRL_DATAPTR_DEC NS(Particles) const* ptr_to_particles_t;

        ptr_to_info_t lhs_it = ( ptr_to_info_t )( uintptr_t
            )NS(Buffer_get_objects_begin_addr)( lhs_buffer );

        ptr_to_info_t lhs_end = ( ptr_to_info_t )( uintptr_t
            )NS(Buffer_get_objects_end_addr)( lhs_buffer );

        ptr_to_info_t rhs_it  = ( ptr_to_info_t )( uintptr_t
            )NS(Buffer_get_objects_begin_addr)( rhs_buffer );

        if( ( ( lhs_it != SIXTRL_NULLPTR ) && ( lhs_end != SIXTRL_NULLPTR ) &&
              ( rhs_it != SIXTRL_NULLPTR ) ) ||
            ( ( lhs_it == SIXTRL_NULLPTR ) && ( lhs_end == SIXTRL_NULLPTR ) &&
              ( rhs_it == SIXTRL_NULLPTR ) ) )
        {
            cmp_result = 0;

            for( ; lhs_it != lhs_end ; ++lhs_it, ++rhs_it )
            {
                ptr_to_particles_t lhs_particles = ( ptr_to_particles_t )(
                    uintptr_t )NS(BufferIndex_get_const_particles)( lhs_it );

                ptr_to_particles_t rhs_particles = ( ptr_to_particles_t )(
                    uintptr_t )NS(BufferIndex_get_const_particles)( rhs_it );

                cmp_result = NS(Particles_compare_values_with_treshold)(
                    lhs_particles, rhs_particles, treshold );

                if( cmp_result != 0 ) break;
            }
        }
    }

    return cmp_result;
}

void NS(Particles_buffers_get_max_difference)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT max_diff_buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT max_diff_indices,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs_buffer,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs_buffer )
{
    if( ( lhs_buffer      != SIXTRL_NULLPTR ) &&
        ( rhs_buffer      != SIXTRL_NULLPTR ) &&
        ( max_diff_buffer != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_get_num_of_objects)( lhs_buffer ) ==
          NS(Buffer_get_num_of_objects)( rhs_buffer ) ) &&
        ( NS(Buffer_get_num_of_objects)( max_diff_buffer ) ==
          NS(Buffer_get_num_of_objects)( lhs_buffer ) ) )
    {
        typedef NS(Object) info_t;
        typedef SIXTRL_DATAPTR_DEC info_t*              ptr_to_info_t;
        typedef SIXTRL_DATAPTR_DEC info_t const*        ptr_to_const_info_t;
        typedef SIXTRL_DATAPTR_DEC NS(Particles)*       ptr_to_particles_t;
        typedef SIXTRL_DATAPTR_DEC NS(Particles) const* ptr_to_const_particles_t;

        ptr_to_const_info_t lhs_it  = ( ptr_to_const_info_t )( uintptr_t
            )NS(Buffer_get_objects_begin_addr)( lhs_buffer );

        ptr_to_const_info_t lhs_end = ( ptr_to_const_info_t )( uintptr_t
            )NS(Buffer_get_objects_end_addr)( lhs_buffer );

        ptr_to_const_info_t rhs_it = ( ptr_to_const_info_t )( uintptr_t
            )NS(Buffer_get_objects_begin_addr)( rhs_buffer );

        ptr_to_info_t max_diff_it = ( ptr_to_info_t )( uintptr_t
            )NS(Buffer_get_objects_begin_addr)( lhs_buffer );

        if( ( lhs_it != SIXTRL_NULLPTR ) && ( lhs_end != SIXTRL_NULLPTR ) &&
            ( rhs_it != SIXTRL_NULLPTR ) )
        {
            for( ; lhs_it != lhs_end ; ++lhs_it, ++rhs_it, ++max_diff_it )
            {
                ptr_to_const_particles_t lhs_particles =
                    NS(BufferIndex_get_const_particles)( lhs_it );

                ptr_to_const_particles_t rhs_particles =
                    NS(BufferIndex_get_const_particles)( rhs_it );

                ptr_to_particles_t max_diff =
                    NS(BufferIndex_get_particles)( max_diff_it );

                NS(Particles_get_max_difference)(
                    max_diff, max_diff_indices, lhs_particles, rhs_particles );

                if( max_diff_indices != SIXTRL_NULLPTR )
                {
                    max_diff_indices =
                        max_diff_indices + NS(PARTICLES_NUM_DATAPTRS);
                }
            }
        }

        return;
    }
}

void NS(Particles_buffer_print_stdout)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT particle_buffer )
{
    NS(Particles_buffer_print)( stdout, particle_buffer );
    return;
}


void NS(Particles_buffer_print)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( particles_buffer != SIXTRL_NULLPTR ) )
    {
        typedef NS(buffer_size_t) buf_size_t;
        typedef SIXTRL_DATAPTR_DEC NS(Object)    const* ptr_to_info_t;
        typedef SIXTRL_DATAPTR_DEC NS(Particles) const* ptr_to_particles_t;

        buf_size_t const nn = NS(Buffer_get_num_of_objects)( particles_buffer );

        ptr_to_info_t obj_it  = ( ptr_to_info_t )( uintptr_t
            )NS(Buffer_get_objects_begin_addr)( particles_buffer );

        ptr_to_info_t obj_end = ( ptr_to_info_t )( uintptr_t
            )NS(Buffer_get_objects_end_addr)( particles_buffer );

        buf_size_t ii = ( buf_size_t )0u;

        for( ; obj_it != obj_end ; ++obj_it, ++ii )
        {
            ptr_to_particles_t particles =
                NS(BufferIndex_get_const_particles)( obj_it );

            if( nn > 1u )
            {
                fprintf(
                    fp, "-------------------------------------------------"
                        "-------------------------------------------------"
                        "------------------------\r\n" );

               fprintf( fp, "particle block index = %8lu / %8lu\r\n",
                             ii + 1, nn );
            }

            NS(Particles_print)( fp, particles );
        }
    }

    return;
}

void NS(Particles_buffer_print_max_diff)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT max_diff_buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* max_diff_indices )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( max_diff_buffer != SIXTRL_NULLPTR ) )
    {
        typedef NS(buffer_size_t) buf_size_t;
        typedef SIXTRL_DATAPTR_DEC NS(Object)    const* ptr_to_info_t;
        typedef SIXTRL_DATAPTR_DEC NS(Particles) const* ptr_to_particles_t;

        buf_size_t const nn = NS(Buffer_get_num_of_objects)( max_diff_buffer );

        ptr_to_info_t obj_it  = ( ptr_to_info_t )( uintptr_t
            )NS(Buffer_get_objects_begin_addr)( max_diff_buffer );

        ptr_to_info_t obj_end = ( ptr_to_info_t )( uintptr_t
            )NS(Buffer_get_objects_end_addr)( max_diff_buffer );

        buf_size_t ii = ( buf_size_t )0u;

        for( ; obj_it != obj_end ; ++obj_it, ++ii )
        {
            ptr_to_particles_t max_diff =
                NS(BufferIndex_get_const_particles)( obj_it );

            if( nn > 1u )
            {
                fprintf(
                    fp, "-------------------------------------------------"
                        "-------------------------------------------------"
                        "------------------------\r\n" );

               fprintf( fp, "particle block index = %8lu / %8lu\r\n",
                             ii + 1, nn );
            }

            NS(Particles_print_max_diff)( fp, max_diff, max_diff_indices );

            if( max_diff_indices != SIXTRL_NULLPTR )
            {
                max_diff_indices =
                    max_diff_indices + NS(PARTICLES_NUM_DATAPTRS);
            }
        }
    }

    return;
}

/* ------------------------------------------------------------------------- */

/* end: tests/sixtracklib/testlib/details/test_particles_tools.c */
