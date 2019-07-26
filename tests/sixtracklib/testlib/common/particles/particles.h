#ifndef SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_COMMON_PARTICLES_HEADER_H__
#define SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_COMMON_PARTICLES_HEADER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_STATIC SIXTRL_FN void NS(Particles_print_out_single)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const index );

SIXTRL_STATIC SIXTRL_FN void NS(Particles_print_out)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN bool NS(Particles_have_same_structure)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs
);

SIXTRL_STATIC SIXTRL_FN bool NS(Particles_map_to_same_memory)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs
);

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN int NS(Particles_compare_real_values)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs
);

SIXTRL_STATIC SIXTRL_FN int
NS(Particles_compare_real_values_with_treshold)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs,
    NS(particle_real_t) const treshold );

SIXTRL_STATIC SIXTRL_FN int NS(Particles_compare_integer_values)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs
);

SIXTRL_STATIC SIXTRL_FN int NS(Particles_compare_values)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs
);

SIXTRL_STATIC SIXTRL_FN int NS(Particles_compare_values_with_treshold)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs,
    NS(particle_real_t) const treshold );

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_random_init)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_realistic_init)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_print_out_single_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_print_out_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Particles_have_same_structure_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs
);

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Particles_map_to_same_memory_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs
);

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Particles_compare_real_values_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs
);

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Particles_compare_real_values_with_treshold_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs,
    NS(particle_real_t) const treshold );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Particles_compare_integer_values_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs
);

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Particles_compare_values_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs
);

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Particles_compare_values_with_treshold_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs,
    NS(particle_real_t) const treshold );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_get_max_difference)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT max_diff,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT max_diff_indices,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs
);

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_print_max_diff_out)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const
        SIXTRL_RESTRICT max_diff,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(buffer_size_t) const* max_diff_indices );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_print_single)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_print_max_diff)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const
        SIXTRL_RESTRICT max_diff,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(buffer_size_t) const* max_diff_indices );

SIXTRL_STATIC SIXTRL_FN void NS(Particles_print_out_single)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const index );

SIXTRL_STATIC SIXTRL_FN void NS(Particles_print_out)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Particles_buffer_have_same_structure)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Particles_buffers_map_to_same_memory)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Particles_buffers_compare_values)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(Particles_buffers_compare_values_with_treshold)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs,
    NS(particle_real_t) const treshold );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_buffers_get_max_difference)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT max_diff,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT max_diff_indices,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_buffer_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT p );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_buffer_print_max_diff)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT max_diff,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* max_diff_indices );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_buffer_print_out)(
     SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT p );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_buffer_print_max_diff_out)(
     SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT max_diff,
     SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* max_diff_indices );

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */
/* helper functions */

SIXTRL_STATIC SIXTRL_FN int NS(Particles_compare_sequences_exact)(
    SIXTRL_PARTICLE_DATAPTR_DEC void const* SIXTRL_RESTRICT lhs_values,
    SIXTRL_PARTICLE_DATAPTR_DEC void const* SIXTRL_RESTRICT rhs_values,
    NS(buffer_size_t) const num_values,
    NS(buffer_size_t) const element_size );

SIXTRL_STATIC SIXTRL_FN int NS(Particles_compare_real_sequences_with_treshold)(
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT lhs_values,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT rhs_values,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_first_out_of_bounds_index,
    NS(buffer_size_t) const num_values,
    NS(particle_real_t) const treshold );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* ===                    Inline implementation                          === */
/* ========================================================================= */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(Particles_compare_sequences_exact)(
    SIXTRL_PARTICLE_DATAPTR_DEC void const* SIXTRL_RESTRICT lhs_values,
    SIXTRL_PARTICLE_DATAPTR_DEC void const* SIXTRL_RESTRICT rhs_values,
    NS(buffer_size_t) const num_values,
    NS(buffer_size_t) const element_size )
{
    int cmp_result = -1;

    NS(buffer_size_t) const ATTR_LENGTH = num_values * element_size;

    if( ATTR_LENGTH > ( NS(buffer_size_t) )0u )
    {
        if( ( lhs_values != 0 ) && ( rhs_values != 0 ) )
        {
            #if !defined( _GPUCODE )
            cmp_result = memcmp( lhs_values, rhs_values, ATTR_LENGTH );
            #else
            typedef NS(buffer_size_t)   buf_size_t;
            typedef SIXTRL_PARTICLE_DATAPTR_DEC SIXTRL_UINIT8_T const* ptr_cmp_t;

            SIXTRL_STATIC_VAR real_t const ZERO = ( real_t )0.0;
            buf_size_t ii = ( buf_size_t )0u;

            ptr_cmp_t _lhs_values = ( ptr_cmp_t )lhs_values;
            ptr_cmp_t _rhs_values = ( ptr_cmp_t )rhs_values;

            cmp_result = 0;

            for( ; ii < ATTR_LENGTH ; ++ii )
            {
                if( _lhs_values[ ii ] > _rhs_values[ ii ] )
                {
                    cmp_result = -1;
                    break;
                }
                else if( _lhs_values[ ii ] < _rhs_values[ ii ] )
                {
                    cmp_result = +1;
                    break;
                }
            }
            #endif /* !defined( _GPUCODE ) */
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

SIXTRL_INLINE int NS(Particles_compare_real_sequences_with_treshold)(
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT lhs_values,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT rhs_values,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_first_out_of_bounds_index,
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

/* ------------------------------------------------------------------------- */


SIXTRL_INLINE void NS(Particles_print_out_single)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const
        SIXTRL_RESTRICT particles, NS(buffer_size_t) const index )
{
    #if defined( _GPUCODE )

    NS(buffer_size_t) const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    if( ( particles != SIXTRL_NULLPTR ) && ( index < num_particles ) )
    {
        printf( "q0             = %.16f\r\n",
                 NS(Particles_get_q0_value)( particles, index ) );

        printf( "mass0          = %.16f\r\n",
                 NS(Particles_get_mass0_value)( particles, index ) );

        printf( "beta0          = %.16f\r\n",
                 NS(Particles_get_beta0_value)( particles, index ) );

        printf( "gamma0         = %.16f\r\n",
                 NS(Particles_get_gamma0_value)( particles, index ) );

        printf( "p0c            = %.16f\r\n",
                 NS(Particles_get_p0c_value)( particles, index ) );

        printf( "s              = %.16f\r\n",
                 NS(Particles_get_s_value)( particles, index ) );

        printf( "x              = %.16f\r\n",
                 NS(Particles_get_x_value)( particles, index ) );

        printf( "y              = %.16f\r\n",
                 NS(Particles_get_y_value)( particles, index ) );

        printf( "px             = %.16f\r\n",
                 NS(Particles_get_px_value)( particles, index ) );

        printf( "py             = %.16f\r\n",
                 NS(Particles_get_py_value)( particles, index ) );

        printf( "zeta           = %.16f\r\n",
                 NS(Particles_get_zeta_value)( particles, index ) );

        printf( "psigma         = %.16f\r\n",
                 NS(Particles_get_psigma_value)( particles, index ) );

        printf( "delta          = %.16f\r\n",
                 NS(Particles_get_delta_value)( particles, index ) );

        printf( "rpp            = %.16f\r\n",
                 NS(Particles_get_rpp_value)( particles, index ) );

        printf( "rvv            = %.16f\r\n",
                 NS(Particles_get_rvv_value)( particles, index ) );

        printf( "chi            = %.16f\r\n",
                 NS(Particles_get_chi_value)( particles, index ) );

        printf( "charge_ratio   = %.16f\r\n",
                 NS(Particles_get_charge_ratio_value)( particles, index ) );

        printf( "particle_id    = %18ld\r\n",
                 NS(Particles_get_particle_id_value)( particles, index ) );

        printf( "at_elem_id     = %18ld\r\n",
                 NS(Particles_get_at_element_id_value)( particles, index ) );

        printf( "at_turn        = %18ld\r\n",
                 NS(Particles_get_at_turn_value)( particles, index ) );

        printf( "state          = %18ld\r\n\r\n",
                 NS(Particles_get_state_value)( particles, index ) );
    }

    #else /* !defined( _GPUCODE ) */

    NS(Particles_print_single)( stdout, particles, index );

    #endif /* !defined( _GPUCODE ) */

    return;
}

SIXTRL_INLINE void NS(Particles_print_out)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const
        SIXTRL_RESTRICT particles )
{
    NS(buffer_size_t) const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    if( ( particles != 0 ) && ( num_particles > 0u ) )
    {
        NS(buffer_size_t) ii = 0u;

        for( ; ii < num_particles ; ++ii )
        {
            if( num_particles > 1u )
            {
                printf( "particle id    = %8lu\r\n", ii );
            }

            NS(Particles_print_out_single)( particles, ii );
        }
    }

    return;
}

/* ------------------------------------------------------------------------ */

SIXTRL_INLINE bool NS(Particles_have_same_structure)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs )
{
    return ( ( lhs != 0 ) && ( rhs != 0 ) &&
             ( NS(Particles_get_num_of_particles)( lhs ) ==
               NS(Particles_get_num_of_particles)( rhs ) ) );
}


SIXTRL_INLINE bool NS(Particles_map_to_same_memory)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs )
{
    bool result = false;

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
            ( NS(Particles_get_const_charge_ratio)( lhs ) ==
              NS(Particles_get_const_charge_ratio)( rhs ) ) &&
            ( NS(Particles_get_const_particle_id)( lhs ) ==
              NS(Particles_get_const_particle_id)( rhs ) ) &&
            ( NS(Particles_get_const_at_element_id)( lhs ) ==
              NS(Particles_get_const_at_element_id)( rhs ) ) &&
            ( NS(Particles_get_const_at_turn)( lhs ) ==
              NS(Particles_get_const_at_turn)( rhs ) ) &&
            ( NS(Particles_get_const_state)( lhs ) ==
              NS(Particles_get_const_state)( rhs ) ) );
    }

    return result;
}


SIXTRL_INLINE int NS(Particles_compare_real_values)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs )
{
    int cmp_result = -1;

    if( NS(Particles_have_same_structure)( lhs, rhs ) )
    {
        NS(buffer_size_t) const NUM_PARTICLES =
            NS(Particles_get_num_of_particles)( lhs );

        NS(buffer_size_t) const REAL_SIZE  = sizeof( NS(particle_real_t) );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_q0)( lhs ),
            NS(Particles_get_const_q0)( rhs ), NUM_PARTICLES, REAL_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_mass0)( lhs ),
            NS(Particles_get_const_mass0)( rhs ), NUM_PARTICLES, REAL_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_beta0)( lhs ),
            NS(Particles_get_const_beta0)( rhs ), NUM_PARTICLES, REAL_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_gamma0)( lhs ),
            NS(Particles_get_const_gamma0)( rhs ), NUM_PARTICLES, REAL_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_p0c)( lhs ),
            NS(Particles_get_const_p0c)( rhs ), NUM_PARTICLES, REAL_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_s)( lhs ),
            NS(Particles_get_const_s)( rhs ), NUM_PARTICLES, REAL_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_x)( lhs ),
            NS(Particles_get_const_x)( rhs ), NUM_PARTICLES, REAL_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_y)( lhs ),
            NS(Particles_get_const_y)( rhs ), NUM_PARTICLES, REAL_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_px)( lhs ),
            NS(Particles_get_const_px)( rhs ), NUM_PARTICLES, REAL_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_py)( lhs ),
            NS(Particles_get_const_py)( rhs ), NUM_PARTICLES, REAL_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_zeta)( lhs ),
            NS(Particles_get_const_zeta)( rhs ), NUM_PARTICLES, REAL_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_psigma)( lhs ),
            NS(Particles_get_const_psigma)( rhs ), NUM_PARTICLES, REAL_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_delta)( lhs ),
            NS(Particles_get_const_delta)( rhs ), NUM_PARTICLES, REAL_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_rpp)( lhs ),
            NS(Particles_get_const_rpp)( rhs ), NUM_PARTICLES, REAL_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_rvv)( lhs ),
            NS(Particles_get_const_rvv)( rhs ), NUM_PARTICLES, REAL_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_chi)( lhs ),
            NS(Particles_get_const_chi)( rhs ), NUM_PARTICLES, REAL_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_charge_ratio)( lhs ),
            NS(Particles_get_const_charge_ratio)( rhs ),
            NUM_PARTICLES, REAL_SIZE );
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


SIXTRL_INLINE int NS(Particles_compare_integer_values)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    const NS(Particles) *const SIXTRL_PARTICLE_ARGPTR_DEC SIXTRL_RESTRICT rhs )
{
    int cmp_result = -1;

    if( NS(Particles_have_same_structure)( lhs, rhs ) )
    {
        NS(buffer_size_t) const NUM_PARTICLES =
            NS(Particles_get_num_of_particles)( lhs );

        NS(buffer_size_t) const INDEX_SIZE = sizeof( SIXTRL_INT64_T );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_particle_id)( lhs ),
            NS(Particles_get_const_particle_id)( rhs ),
            NUM_PARTICLES, INDEX_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_at_element_id)( lhs ),
            NS(Particles_get_const_at_element_id)( rhs ),
            NUM_PARTICLES, INDEX_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_at_turn)( lhs ),
            NS(Particles_get_const_at_turn)( rhs ),
            NUM_PARTICLES, INDEX_SIZE );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_sequences_exact)(
            NS(Particles_get_const_state)( lhs ),
            NS(Particles_get_const_state)( rhs ),
            NUM_PARTICLES, INDEX_SIZE );
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

SIXTRL_INLINE int NS(Particles_compare_real_values_with_treshold)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs,
    NS(particle_real_t) const treshold )
{
    int cmp_result = -1;

    if( NS(Particles_have_same_structure)( lhs, rhs ) )
    {
        NS(buffer_size_t) const NUM_PARTICLES =
            NS(Particles_get_num_of_particles)( lhs );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_q0)( lhs ),
            NS(Particles_get_const_q0)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_mass0)( lhs ),
            NS(Particles_get_const_mass0)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_beta0)( lhs ),
            NS(Particles_get_const_beta0)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_gamma0)( lhs ),
            NS(Particles_get_const_gamma0)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_p0c)( lhs ),
            NS(Particles_get_const_p0c)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_s)( lhs ),
            NS(Particles_get_const_s)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_x)( lhs ),
            NS(Particles_get_const_x)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_y)( lhs ),
            NS(Particles_get_const_y)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_px)( lhs ),
            NS(Particles_get_const_px)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_py)( lhs ),
            NS(Particles_get_const_py)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_zeta)( lhs ),
            NS(Particles_get_const_zeta)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_psigma)( lhs ),
            NS(Particles_get_const_psigma)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_delta)( lhs ),
            NS(Particles_get_const_delta)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_rpp)( lhs ),
            NS(Particles_get_const_rpp)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_rvv)( lhs ),
            NS(Particles_get_const_rvv)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_chi)( lhs ),
            NS(Particles_get_const_chi)( rhs ), 0, NUM_PARTICLES, treshold );

        if( cmp_result != 0 ) return cmp_result;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        cmp_result = NS(Particles_compare_real_sequences_with_treshold)(
            NS(Particles_get_const_charge_ratio)( lhs ),
            NS(Particles_get_const_charge_ratio)( rhs ), 0,
                NUM_PARTICLES, treshold );
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

SIXTRL_INLINE int NS(Particles_compare_values)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs )
{
    int cmp_result = NS(Particles_compare_real_values)( lhs, rhs );

    if( cmp_result == 0 )
    {
        cmp_result = NS(Particles_compare_integer_values)( lhs, rhs );
    }

    return cmp_result;
}


SIXTRL_INLINE int NS(Particles_compare_values_with_treshold)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs,
    NS(particle_real_t) const treshold )
{
    int cmp_result = NS(Particles_compare_real_values_with_treshold)(
        lhs, rhs, treshold );

    if( cmp_result == 0 )
    {
        cmp_result = NS(Particles_compare_integer_values)( lhs, rhs );
    }

    return cmp_result;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_COMMON_PARTICLES_HEADER_H__ */

/* end: tests/sixtracklib/testlib/common/particles/particles.h */
