#ifndef SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_PRIV_PARTICLES_OPTIMIZED_KERNEL_CL__
#define SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_PRIV_PARTICLES_OPTIMIZED_KERNEL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/internal/particles_defines.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/track.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#pragma OPENCL_EXTENSION cl_khr_int32_extended_atomics

__kernel void NS(Track_particles_beam_elements_priv_particles_optimized_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements_buf,
    SIXTRL_UINT64_T const num_turns,
    SIXTRL_BUFFER_DATAPTR_DEC int* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*        obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*  obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)* ptr_particles_t;

    int success_flag = ( int )0u;
    buf_size_t const slot_size = ( buf_size_t )8u;

    if( ( !NS(ManagedBuffer_needs_remapping( particles_buf,     slot_size ) ) ) &&
        ( !NS(ManagedBuffer_needs_remapping( beam_elements_buf, slot_size ) ) ) )
    {
        size_t const stride             = get_local_size( 0 );
        size_t global_particle_id       = get_global_id( 0 );
        size_t object_begin_particle_id = ( size_t )0u;

        obj_iter_t part_block_it  = NS(ManagedBuffer_get_objects_index_begin)(
                particles_buf, slot_size );

        obj_iter_t part_block_end =
            NS(ManagedBuffer_get_objects_index_end)( particles_buf, slot_size );

        obj_const_iter_t be_begin =
            NS(ManagedBuffer_get_const_objects_index_begin)(
                beam_elements_buf, slot_size );

        obj_const_iter_t be_end   =
            NS(ManagedBuffer_get_const_objects_index_end)(
                beam_elements_buf, slot_size );

        for( ; part_block_it != part_block_end ; ++part_block_it )
        {
            ptr_particles_t particles_data = ( ptr_particles_t )(
                uintptr_t )NS(Object_get_begin_addr)( part_block_it );

            size_t const in_num_particles =
                ( particles_data != SIXTRL_NULLPTR ) ?
                    particles_data->num_particles : ( size_t )0u;

            size_t const object_end_particle_id =
                object_begin_particle_id + in_num_particles;

            SIXTRL_ASSERT( NS(Object_get_type_id)( part_block_it ) ==
                           NS(OBJECT_TYPE_PARTICLE) );

            if( ( global_particle_id <  object_end_particle_id   ) &&
                ( global_particle_id >= object_begin_particle_id ) )
            {
                NS(particle_real_t) q0             = ( NS(particle_real_t) )0.0;
                NS(particle_real_t) mass0          = ( NS(particle_real_t) )0.0;
                NS(particle_real_t) beta0          = ( NS(particle_real_t) )0.0;
                NS(particle_real_t) gamma0         = ( NS(particle_real_t) )0.0;
                NS(particle_real_t) p0c            = ( NS(particle_real_t) )0.0;

                NS(particle_real_t) s              = ( NS(particle_real_t) )0.0;
                NS(particle_real_t) x              = ( NS(particle_real_t) )0.0;
                NS(particle_real_t) y              = ( NS(particle_real_t) )0.0;
                NS(particle_real_t) px             = ( NS(particle_real_t) )0.0;
                NS(particle_real_t) py             = ( NS(particle_real_t) )0.0;
                NS(particle_real_t) zeta           = ( NS(particle_real_t) )0.0;

                NS(particle_real_t) psigma         = ( NS(particle_real_t) )0.0;
                NS(particle_real_t) delta          = ( NS(particle_real_t) )0.0;
                NS(particle_real_t) rpp            = ( NS(particle_real_t) )0.0;
                NS(particle_real_t) rvv            = ( NS(particle_real_t) )0.0;
                NS(particle_real_t) chi            = ( NS(particle_real_t) )0.0;
                NS(particle_real_t) charge_ratio   = ( NS(particle_real_t) )0.0;

                NS(particle_index_t) part_id       = ( NS(particle_index_t) )0;
                NS(particle_index_t) at_element_id = ( NS(particle_index_t) )0;
                NS(particle_index_t) at_turn       = ( NS(particle_index_t) )0;
                NS(particle_index_t) state         = ( NS(particle_index_t) )0;

                SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) particles;

                SIXTRL_UINT64_T turn = ( SIXTRL_UINT64_T )0u;

                size_t const particle_id =
                    global_particle_id - object_begin_particle_id;

                NS(Particles_set_num_of_particles)(        &particles, 1u );

                NS(Particles_assign_ptr_to_q0)(            &particles, &q0 );
                NS(Particles_assign_ptr_to_mass0)(         &particles, &mass0 );
                NS(Particles_assign_ptr_to_beta0)(         &particles, &beta0 );
                NS(Particles_assign_ptr_to_gamma0)(        &particles, &gamma0 );
                NS(Particles_assign_ptr_to_p0c)(           &particles, &p0c );

                NS(Particles_assign_ptr_to_s)(             &particles, &s );
                NS(Particles_assign_ptr_to_x)(             &particles, &x );
                NS(Particles_assign_ptr_to_y)(             &particles, &y );
                NS(Particles_assign_ptr_to_px)(            &particles, &px );
                NS(Particles_assign_ptr_to_py)(            &particles, &py );
                NS(Particles_assign_ptr_to_zeta)(          &particles, &zeta );

                NS(Particles_assign_ptr_to_psigma)(        &particles, &psigma );
                NS(Particles_assign_ptr_to_delta)(         &particles, &delta );
                NS(Particles_assign_ptr_to_rpp)(           &particles, &rpp );
                NS(Particles_assign_ptr_to_rvv)(           &particles, &rvv );
                NS(Particles_assign_ptr_to_chi)(           &particles, &chi );
                NS(Particles_assign_ptr_to_charge_ratio)(  &particles, &charge_ratio );

                NS(Particles_assign_ptr_to_particle_id)(   &particles, &part_id );
                NS(Particles_assign_ptr_to_at_element_id)( &particles, &at_element_id );
                NS(Particles_assign_ptr_to_at_turn)(       &particles, &at_turn );
                NS(Particles_assign_ptr_to_state)(         &particles, &state );

                SIXTRL_ASSERT( particle_id <
                    NS(Particles_get_num_of_particles)( particles ) );

                success_flag |= NS(Particles_from_generic_addr_data)(
                    &particles, particles_data, particle_id );

                SIXTRL_ASSERT( success_flag == ( int )0 );

                for( ; turn < num_turns ; ++turn )
                {
                    success_flag |= NS(Track_particle_beam_element_objs)(
                        &particles, 0u, be_begin, be_end );
                }

                success_flag |=  NS(Particles_back_to_generic_addr_data)(
                    particles_data, &particles, particle_id );
            }

            object_begin_particle_id = object_end_particle_id;
        }
    }
    else
    {
        if( NS(ManagedBuffer_needs_remapping( particles_buf, slot_size ) ) )
        {
            success_flag |= -2;
        }

        if( NS(ManagedBuffer_needs_remapping( beam_elements_buf, slot_size ) ) )
        {
            success_flag |= -4;
        }
    }

    if( ( success_flag != 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {

        atomic_or( ptr_success_flag, success_flag );
    }

    return;
}

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_PRIV_PARTICLES_OPTIMIZED_KERNEL_CL__ */
/* end: sixtracklib/opencl/private/track_particles_priv_particles_optimized_kernel.cl */
