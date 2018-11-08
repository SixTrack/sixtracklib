#ifndef SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_OPTIMIZED_PRIV_PARTICLES_KERNEL_CL__
#define SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_OPTIMIZED_PRIV_PARTICLES_KERNEL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/internal/particles_defines.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/track.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(Track_particles_single_turn_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buf );

__kernel void NS(Track_particles_until_turn_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buf,
    SIXTRL_INT64_T const turn );

__kernel void NS(Track_particles_elem_by_elem_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT elem_by_elem_buf,
    SIXTRL_UINT64_T const io_particle_blocks_offset );

/* ========================================================================= */

__kernel void NS(Track_particles_single_turn_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buf )
{
    typedef NS(buffer_size_t)                                   buf_size_t;
    typedef NS(particle_num_elements_t)                         num_element_t;
    typedef NS(particle_real_t)                                 real_t;
    typedef NS(particle_index_t)                                index_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*           obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*     obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)* ptr_particles_t;

    buf_size_t const slot_size = ( buf_size_t )8u;
    num_element_t particle_id  = ( num_element_t )get_global_id( 0 );
    num_element_t const stride = ( num_element_t )get_global_size( 0 );

    obj_const_iter_t be_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
        belem_buf, slot_size );

    obj_const_iter_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
        belem_buf, slot_size );

    obj_iter_t pb_it = NS(ManagedBuffer_get_objects_index_begin)(
        particles_buf, slot_size );

    ptr_particles_t in_particles = ( ptr_particles_t )( uintptr_t
        )NS(Object_get_begin_addr)( pb_it );

    num_element_t const num_particles = ( in_particles != SIXTRL_NULLPTR )
        ? in_particles->num_particles : ( num_element_t )0u;

    real_t real_values[] =
    {
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0
    };

    index_t index_values[] =
    {
        ( index_t )0, ( index_t )0, ( index_t )0, ( index_t )0
    };

    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) particles;

    NS(Particles_set_num_of_particles)(        &particles, 1u );
    NS(Particles_assign_ptr_to_q0)(            &particles, &real_values[  0 ] );
    NS(Particles_assign_ptr_to_mass0)(         &particles, &real_values[  1 ] );
    NS(Particles_assign_ptr_to_beta0)(         &particles, &real_values[  2 ] );
    NS(Particles_assign_ptr_to_gamma0)(        &particles, &real_values[  3 ] );
    NS(Particles_assign_ptr_to_p0c)(           &particles, &real_values[  4 ] );

    NS(Particles_assign_ptr_to_s)(             &particles, &real_values[  5 ] );
    NS(Particles_assign_ptr_to_x)(             &particles, &real_values[  6 ] );
    NS(Particles_assign_ptr_to_y)(             &particles, &real_values[  7 ] );
    NS(Particles_assign_ptr_to_px)(            &particles, &real_values[  8 ] );
    NS(Particles_assign_ptr_to_py)(            &particles, &real_values[  9 ] );
    NS(Particles_assign_ptr_to_zeta)(          &particles, &real_values[ 10 ] );

    NS(Particles_assign_ptr_to_psigma)(        &particles, &real_values[ 11 ] );
    NS(Particles_assign_ptr_to_delta)(         &particles, &real_values[ 12 ] );
    NS(Particles_assign_ptr_to_rpp)(           &particles, &real_values[ 13 ] );
    NS(Particles_assign_ptr_to_rvv)(           &particles, &real_values[ 14 ] );
    NS(Particles_assign_ptr_to_chi)(           &particles, &real_values[ 15 ] );
    NS(Particles_assign_ptr_to_charge_ratio)(  &particles, &real_values[ 16 ] );

    NS(Particles_assign_ptr_to_particle_id)(   &particles, &index_values[ 0 ] );
    NS(Particles_assign_ptr_to_at_element_id)( &particles, &index_values[ 1 ] );
    NS(Particles_assign_ptr_to_at_turn)(       &particles, &index_values[ 2 ] );
    NS(Particles_assign_ptr_to_state)(         &particles, &index_values[ 3 ] );

    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)( particles_buf, slot_size ) );
    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)( belem_buf, slot_size ) );

    SIXTRL_ASSERT( pb_it != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( NS(Object_get_type_id)( pb_it ) !=
                   NS(OBJECT_TYPE_PARTICLES ) );

    SIXTRL_ASSERT( NS(Object_get_begin_addr)( pb_it ) !=
                   ( NS(buffer_addr_t) )0u );

    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
        particles_buffer, slot_size ) == ( buf_size_t )1u );

    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
        belem_buf, slot_size ) > ( buf_size_t )0u );

    SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
        particles_buffer, slot_size ) );

    SIXTRL_ASSERT( in_particles != SIXTRL_NULLPTR );

    for( ; particle_id < num_particles ; particle_id += stride )
    {
        NS(Particles_from_generic_addr_data)(
            &particles, in_particles, particle_id );

        NS(Track_particle_beam_elements_obj)(
            &particles, 0u, be_begin, be_end );

        NS(Track_particle_increment_at_turn)( &particles, 0u );

        NS(Particles_back_to_generic_addr_data)(
            in_particles, &particles, particle_id );
    }

    return;
}

/* ------------------------------------------------------------------------- */

__kernel void NS(Track_particles_until_turn_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buf,
    SIXTRL_INT64_T const turn )
{
    typedef NS(buffer_size_t)                                buf_size_t;
    typedef NS(particle_num_elements_t)                      num_element_t;
    typedef NS(particle_real_t)                              real_t;
    typedef NS(particle_index_t)                             index_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*           obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*     obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)* ptr_particles_t;

    buf_size_t const slot_size = ( buf_size_t )8u;
    num_element_t particle_id  = ( num_element_t )get_global_id( 0 );
    num_element_t const stride = ( num_element_t )get_global_size( 0 );

    obj_const_iter_t be_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
        belem_buf, slot_size );

    obj_const_iter_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
        belem_buf, slot_size );

    obj_iter_t pb_it = NS(ManagedBuffer_get_objects_index_begin)(
        particles_buf, slot_size );

    ptr_particles_t in_particles = ( ptr_particles_t )( uintptr_t
        )NS(Object_get_begin_addr)( pb_it );

    num_element_t const num_particles = ( in_particles != SIXTRL_NULLPTR )
        ? in_particles->num_particles : ( num_element_t )0u;

    real_t real_values[] =
    {
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0
    };

    index_t index_values[] =
    {
        ( index_t )0, ( index_t )0, ( index_t )0, ( index_t )0
    };

    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) particles;

    NS(Particles_set_num_of_particles)(        &particles, 1u );
    NS(Particles_assign_ptr_to_q0)(            &particles, &real_values[  0 ] );
    NS(Particles_assign_ptr_to_mass0)(         &particles, &real_values[  1 ] );
    NS(Particles_assign_ptr_to_beta0)(         &particles, &real_values[  2 ] );
    NS(Particles_assign_ptr_to_gamma0)(        &particles, &real_values[  3 ] );
    NS(Particles_assign_ptr_to_p0c)(           &particles, &real_values[  4 ] );

    NS(Particles_assign_ptr_to_s)(             &particles, &real_values[  5 ] );
    NS(Particles_assign_ptr_to_x)(             &particles, &real_values[  6 ] );
    NS(Particles_assign_ptr_to_y)(             &particles, &real_values[  7 ] );
    NS(Particles_assign_ptr_to_px)(            &particles, &real_values[  8 ] );
    NS(Particles_assign_ptr_to_py)(            &particles, &real_values[  9 ] );
    NS(Particles_assign_ptr_to_zeta)(          &particles, &real_values[ 10 ] );

    NS(Particles_assign_ptr_to_psigma)(        &particles, &real_values[ 11 ] );
    NS(Particles_assign_ptr_to_delta)(         &particles, &real_values[ 12 ] );
    NS(Particles_assign_ptr_to_rpp)(           &particles, &real_values[ 13 ] );
    NS(Particles_assign_ptr_to_rvv)(           &particles, &real_values[ 14 ] );
    NS(Particles_assign_ptr_to_chi)(           &particles, &real_values[ 15 ] );
    NS(Particles_assign_ptr_to_charge_ratio)(  &particles, &real_values[ 16 ] );

    NS(Particles_assign_ptr_to_particle_id)(   &particles, &index_values[ 0 ] );
    NS(Particles_assign_ptr_to_at_element_id)( &particles, &index_values[ 1 ] );
    NS(Particles_assign_ptr_to_at_turn)(       &particles, &index_values[ 2 ] );
    NS(Particles_assign_ptr_to_state)(         &particles, &index_values[ 3 ] );

    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)( particles_buf, slot_size ) );
    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)( belem_buf, slot_size ) );

    SIXTRL_ASSERT( pb_it != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( NS(Object_get_type_id)( pb_it ) !=
                   NS(OBJECT_TYPE_PARTICLES ) );

    SIXTRL_ASSERT( NS(Object_get_begin_addr)( pb_it ) !=
                   ( NS(buffer_addr_t) )0u );

    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
        particles_buffer, slot_size ) == ( buf_size_t )1u );

    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
        belem_buf, slot_size ) > ( buf_size_t )0u );

    SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
        particles_buffer, slot_size ) );

    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );

    for( ; particle_id < num_particles ; particle_id += stride )
    {
        NS(Particles_from_generic_addr_data)(
            &particles, in_particles, particle_id );

        NS(Track_particle_until_turn_obj)(
            &particles, 0u, be_begin, be_end, turn );

        NS(Particles_back_to_generic_addr_data)(
            in_particles, &particles, particle_id );
    }

    return;
}

/* ------------------------------------------------------------------------- */

__kernel void NS(Track_particles_elem_by_elem_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT elem_by_elem_buf,
    SIXTRL_UINT64_T const io_particle_blocks_offset )
{
    typedef NS(buffer_size_t)                                   buf_size_t;
    typedef NS(particle_num_elements_t)                         num_element_t;
    typedef NS(particle_real_t)                                 real_t;
    typedef NS(particle_index_t)                                index_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*           obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*     obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)* ptr_particles_t;

    buf_size_t const slot_size = ( buf_size_t )8u;
    num_element_t particle_id  = ( num_element_t )get_global_id( 0 );
    num_element_t const stride = ( num_element_t )get_global_size( 0 );

    obj_const_iter_t be_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
        belem_buf, slot_size );

    obj_const_iter_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
        belem_buf, slot_size );

    obj_iter_t io_obj_begin = NS(ManagedBuffer_get_objects_index_begin)(
        elem_by_elem_buf, slot_size );

    obj_iter_t pb_it = NS(ManagedBuffer_get_objects_index_begin)(
        particles_buf, slot_size );

    ptr_particles_t in_particles = ( ptr_particles_t )( uintptr_t
        )NS(Object_get_begin_addr)( pb_it );

    num_element_t const num_particles = ( in_particles != SIXTRL_NULLPTR )
        ? in_particles->num_particles : ( num_element_t )0u;

    real_t real_values[] =
    {
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0
    };

    index_t index_values[] =
    {
        ( index_t )0, ( index_t )0, ( index_t )0, ( index_t )0
    };

    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) particles;

    NS(Particles_set_num_of_particles)(        &particles, 1u );
    NS(Particles_assign_ptr_to_q0)(            &particles, &real_values[  0 ] );
    NS(Particles_assign_ptr_to_mass0)(         &particles, &real_values[  1 ] );
    NS(Particles_assign_ptr_to_beta0)(         &particles, &real_values[  2 ] );
    NS(Particles_assign_ptr_to_gamma0)(        &particles, &real_values[  3 ] );
    NS(Particles_assign_ptr_to_p0c)(           &particles, &real_values[  4 ] );

    NS(Particles_assign_ptr_to_s)(             &particles, &real_values[  5 ] );
    NS(Particles_assign_ptr_to_x)(             &particles, &real_values[  6 ] );
    NS(Particles_assign_ptr_to_y)(             &particles, &real_values[  7 ] );
    NS(Particles_assign_ptr_to_px)(            &particles, &real_values[  8 ] );
    NS(Particles_assign_ptr_to_py)(            &particles, &real_values[  9 ] );
    NS(Particles_assign_ptr_to_zeta)(          &particles, &real_values[ 10 ] );

    NS(Particles_assign_ptr_to_psigma)(        &particles, &real_values[ 11 ] );
    NS(Particles_assign_ptr_to_delta)(         &particles, &real_values[ 12 ] );
    NS(Particles_assign_ptr_to_rpp)(           &particles, &real_values[ 13 ] );
    NS(Particles_assign_ptr_to_rvv)(           &particles, &real_values[ 14 ] );
    NS(Particles_assign_ptr_to_chi)(           &particles, &real_values[ 15 ] );
    NS(Particles_assign_ptr_to_charge_ratio)(  &particles, &real_values[ 16 ] );

    NS(Particles_assign_ptr_to_particle_id)(   &particles, &index_values[ 0 ] );
    NS(Particles_assign_ptr_to_at_element_id)( &particles, &index_values[ 1 ] );
    NS(Particles_assign_ptr_to_at_turn)(       &particles, &index_values[ 2 ] );
    NS(Particles_assign_ptr_to_state)(         &particles, &index_values[ 3 ] );

    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)( particles_buf, slot_size ) );
    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)( belem_buf, slot_size ) );
    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)( elem_by_elem_buf, slot_size ) );

    SIXTRL_ASSERT( be_begin     != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end       != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( io_obj_begin != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( pb_it != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( NS(Object_get_type_id)( pb_it ) !=
                   NS(OBJECT_TYPE_PARTICLES ) );

    SIXTRL_ASSERT( NS(Object_get_begin_addr)( pb_it ) !=
                   ( NS(buffer_addr_t) )0u );

    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
        particles_buffer, slot_size ) == ( buf_size_t )1u );

    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
        belem_buf, slot_size ) > ( buf_size_t )0u );

    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)( elem_by_elem_buf ) >=
        ( io_particle_blocks_offset + NS(ManagedBuffer_get_num_objects)(
            belem_buf, slot_size ) ) );

    SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
        particles_buffer, slot_size ) );

    SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
        elem_by_elem_buf, slot_size ) );

    SIXTRL_ASSERT( in_particles != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( in_particles->at_element_id != SIXTRL_NULLPTR );

    io_obj_begin = io_obj_begin + io_particle_blocks_offset;

    for( ; particle_id < num_particles ; particle_id += stride )
    {
        obj_iter_t       io_obj_it = io_obj_begin;
        obj_const_iter_t be_it     = be_begin;

        NS(Particles_from_generic_addr_data)(
            &particles, in_particles, particle_id );

        index_t beam_element_id =
            NS(Particles_get_at_element_id_value)( &particles, 0u );

        for( ; be_it != be_end ; ++be_it, ++io_obj_it )
        {
            ptr_particles_t dump_particles = ( ptr_particles_t
                )NS(Object_get_begin_ptr)( io_obj_it );

            SIXTRL_ASSERT( dump_particles != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( NS(Object_get_type_id)( io_obj_it ) ==
                           NS(OBJECT_TYPE_PARTICLES) );

            NS(Particles_back_to_generic_addr_data)(
                dump_particles, &particles, particle_id );

            if( 0 != NS(Track_particle_beam_element_obj)(
                    &particles, 0u, beam_element_id++, be_it ) )
            {
                break;
            }
        }

        if( NS(Particles_get_state_value)( &particles, 0u ) == ( index_t )1u )
        {
            NS(Particles_set_at_element_id_value)( &particles, 0u, beam_element_id );
        }

        NS(Particles_back_to_generic_addr_data)(
            in_particles, &particles, particle_id );
    }

    return;
}

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_OPTIMIZED_PRIV_PARTICLES_KERNEL_CL__ */

/* end: sixtracklib/opencl/kernels/track_particles_optimized_priv_particles.cl */
