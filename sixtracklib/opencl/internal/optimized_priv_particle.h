#ifndef SIXTRACKLIB_OPENCL_INTERNAL_OPTIMIZED_PRIVATE_PARTICLE_H__
#define SIXTRACKLIB_OPENCL_INTERNAL_OPTIMIZED_PRIVATE_PARTICLE_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/particles_defines.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __OPENCL_VERSION__ )

    SIXTRL_STATIC SIXTRL_DEVICE_FN
    void NS(OpenCl1x_init_optimized_priv_particle)(
        SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT part,
        NS(particle_real_t)* SIXTRL_RESTRICT real_values,
        NS(particle_index_t)* SIXTRL_RESTRICT index_values );

    /* --------------------------------------------------------------------- */

    SIXTRL_INLINE void NS(OpenCl1x_init_optimized_priv_particle)(
        SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT part,
        NS(particle_real_t)* SIXTRL_RESTRICT real_values,
        NS(particle_index_t)* SIXTRL_RESTRICT index_values )
    {
        SIXTRL_ASSERT( part != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( real_values  != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( index_values != SIXTRL_NULLPTR );

        NS(Particles_set_num_of_particles)(        part, 1u );
        NS(Particles_assign_ptr_to_q0)(            part, &real_values[  0 ] );
        NS(Particles_assign_ptr_to_mass0)(         part, &real_values[  1 ] );
        NS(Particles_assign_ptr_to_beta0)(         part, &real_values[  2 ] );
        NS(Particles_assign_ptr_to_gamma0)(        part, &real_values[  3 ] );
        NS(Particles_assign_ptr_to_p0c)(           part, &real_values[  4 ] );

        NS(Particles_assign_ptr_to_s)(             part, &real_values[  5 ] );
        NS(Particles_assign_ptr_to_x)(             part, &real_values[  6 ] );
        NS(Particles_assign_ptr_to_y)(             part, &real_values[  7 ] );
        NS(Particles_assign_ptr_to_px)(            part, &real_values[  8 ] );
        NS(Particles_assign_ptr_to_py)(            part, &real_values[  9 ] );
        NS(Particles_assign_ptr_to_zeta)(          part, &real_values[ 10 ] );

        NS(Particles_assign_ptr_to_psigma)(        part, &real_values[ 11 ] );
        NS(Particles_assign_ptr_to_delta)(         part, &real_values[ 12 ] );
        NS(Particles_assign_ptr_to_rpp)(           part, &real_values[ 13 ] );
        NS(Particles_assign_ptr_to_rvv)(           part, &real_values[ 14 ] );
        NS(Particles_assign_ptr_to_chi)(           part, &real_values[ 15 ] );
        NS(Particles_assign_ptr_to_charge_ratio)(  part, &real_values[ 16 ] );

        NS(Particles_assign_ptr_to_particle_id)(   part, &index_values[ 0 ] );
        NS(Particles_assign_ptr_to_at_element_id)( part, &index_values[ 1 ] );
        NS(Particles_assign_ptr_to_at_turn)(       part, &index_values[ 2 ] );
        NS(Particles_assign_ptr_to_state)(         part, &index_values[ 3 ] );

        return;
    }

#endif /* defined( __OPENCL_VERSION__ ) */

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_OPTIMIZED_PRIVATE_PARTICLE_H__ */
/* end: sixtracklib/opencl/internal/optimized_priv_particle.h */
