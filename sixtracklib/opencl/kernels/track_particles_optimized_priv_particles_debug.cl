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

#pragma OPENCL_EXTENSION cl_khr_int32_extended_atomics

__kernel void NS(Track_particles_single_turn_opt_pp_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*  SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*  SIXTRL_RESTRICT belem_buffer,
    SIXTRL_INT64_T const increment_turn,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T* SIXTRL_RESTRICT ptr_success_flag );

__kernel void NS(Track_particles_until_turn_opt_pp_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_INT64_T const until_turn,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T* SIXTRL_RESTRICT ptr_success_flag );

__kernel void NS(Track_particles_elem_by_elem_opt_pp_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT elem_by_elem_buffer,
    SIXTRL_INT64_T  const until_turn,
    SIXTRL_UINT64_T const out_particle_blocks_offset,
    SIXTRL_INT64_T  const min_particle_id,
    SIXTRL_INT64_T  const max_particle_id,
    SIXTRL_INT64_T  const min_element_id,
    SIXTRL_INT64_T  const max_element_id,
    SIXTRL_INT64_T  const min_turn_id,
    SIXTRL_INT64_T  const max_turn_id,
    SIXTRL_INT64_T  const elem_by_elem_index_ordering,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T* SIXTRL_RESTRICT ptr_success_flag );

/* ========================================================================= */

__kernel void NS(Track_particles_single_turn_opt_pp_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_INT64_T const increment_turn,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t)                                   buf_size_t;
    typedef NS(particle_num_elements_t)                         num_element_t;
    typedef NS(particle_real_t)                                 real_t;
    typedef NS(particle_index_t)                                index_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*           obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*     obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)* ptr_particles_t;

    buf_size_t const  slot_size = ( buf_size_t )8u;
    SIXTRL_INT32_T success_flag = ( SIXTRL_INT32_T )-1;

    if( ( !NS(ManagedBuffer_needs_remapping)( particles_buffer, slot_size ) ) &&
        (  NS(ManagedBuffer_get_num_objects)(
                particles_buffer, slot_size ) == ( buf_size_t )1u ) &&
        ( !NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) ) )
    {
        num_element_t particle_index = ( num_element_t )get_global_id( 0 );
        num_element_t const stride   = ( num_element_t )get_global_size( 0 );

        obj_const_iter_t be_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
            belem_buffer, slot_size );

        obj_const_iter_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
            belem_buffer, slot_size );

        obj_iter_t pb_it = NS(ManagedBuffer_get_objects_index_begin)(
            particles_buffer, slot_size );

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

        index_t start_element_id = ( index_t )0u;

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

        SIXTRL_ASSERT( pb_it != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( NS(Object_get_type_id)( pb_it ) !=
                       NS(OBJECT_TYPE_PARTICLES ) );

        SIXTRL_ASSERT( NS(Object_get_begin_addr)( pb_it ) !=
                       ( NS(buffer_addr_t) )0u );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
            particles_buffer, slot_size ) == ( buf_size_t )1u );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
            belem_buffer, slot_size ) > ( buf_size_t )0u );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            particles_buffer, slot_size ) );

        SIXTRL_ASSERT( in_particles != SIXTRL_NULLPTR );

        success_flag = ( SIXTRL_INT32_T )0u;

        for( ; particle_index < num_particles ; particle_index += stride )
        {
            success_flag |= NS(Particles_copy_from_generic_addr_data)(
                &particles, 0, in_particles, particle_index );

            start_element_id =
                NS(Particles_get_at_element_id_value)( &particles, 0 );

            success_flag |= NS(Track_particle_beam_elements_obj)(
                &particles, 0u, be_begin, be_end );

            if( ( success_flag == 0 ) &&
                ( increment_turn != ( SIXTRL_INT64_T )0 ) &&
                ( NS(Particles_is_not_lost_value)( &particles, 0 ) ) )
            {
                NS(Particles_increment_at_turn_value)( &particles, 0 );
                NS(Particles_set_at_element_id_value)( &particles, 0, start_element_id );
            }

            success_flag |= NS(Particles_copy_to_generic_addr_data)(
                in_particles, particle_index, &particles, 0 );

            if( success_flag != 0 ) break;
        }
    }
    else if( NS(ManagedBuffer_needs_remapping)( particles_buffer, slot_size) )
    {
        success_flag |= ( SIXTRL_INT32_T )-2;
    }
    else if( NS(ManagedBuffer_get_num_objects)(
                particles_buffer, slot_size ) != ( buf_size_t )1u )
    {
        success_flag |= ( SIXTRL_INT32_T )-4;
    }
    else if( NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) )
    {
        success_flag |= ( SIXTRL_INT32_T )-8;
    }

    if( ( success_flag == 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {
        atomic_or( ptr_success_flag, success_flag );
    }

    return;
}

/* ------------------------------------------------------------------------- */

__kernel void NS(Track_particles_until_turn_opt_pp_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_INT64_T const until_turn,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t)                                   buf_size_t;
    typedef NS(particle_num_elements_t)                         num_element_t;
    typedef NS(particle_real_t)                                 real_t;
    typedef NS(particle_index_t)                                index_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*           obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*     obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)* ptr_particles_t;

    buf_size_t const  slot_size = ( buf_size_t )8u;
    SIXTRL_INT32_T success_flag = ( SIXTRL_INT32_T )-1;

    if( ( !NS(ManagedBuffer_needs_remapping)( particles_buffer, slot_size ) ) &&
        (  NS(ManagedBuffer_get_num_objects)(
                particles_buffer, slot_size ) == ( buf_size_t )1u ) &&
        ( !NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) ) )
    {
        buf_size_t const slot_size    = ( buf_size_t )8u;
        num_element_t particle_index  = ( num_element_t )get_global_id( 0 );
        num_element_t const stride    = ( num_element_t )get_global_size( 0 );

        obj_const_iter_t be_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
            belem_buffer, slot_size );

        obj_const_iter_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
            belem_buffer, slot_size );

        obj_iter_t pb_it = NS(ManagedBuffer_get_objects_index_begin)(
            particles_buffer, slot_size );

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

        SIXTRL_ASSERT( pb_it != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( NS(Object_get_type_id)( pb_it ) !=
                       NS(OBJECT_TYPE_PARTICLES ) );

        SIXTRL_ASSERT( NS(Object_get_begin_addr)( pb_it ) !=
                       ( NS(buffer_addr_t) )0u );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
            particles_buffer, slot_size ) == ( buf_size_t )1u );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
            belem_buffer, slot_size ) > ( buf_size_t )0u );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            particles_buffer, slot_size ) );

        SIXTRL_ASSERT( in_particles != SIXTRL_NULLPTR );

        success_flag = 0;

        for( ; particle_index < num_particles ; particle_index += stride )
        {
            success_flag |= NS(Particles_copy_from_generic_addr_data)(
                &particles, 0, in_particles, particle_index );

            success_flag |= NS(Track_particle_until_turn_obj)(
                &particles, 0u, be_begin, be_end, until_turn );

            success_flag |= NS(Particles_copy_to_generic_addr_data)(
                in_particles, particle_index, &particles, 0 );

            if( success_flag != 0 ) break;
        }
    }
    else if( NS(ManagedBuffer_needs_remapping)( particles_buffer, slot_size) )
    {
        success_flag |= ( SIXTRL_INT32_T )-2;
    }
    else if( NS(ManagedBuffer_get_num_objects)(
                particles_buffer, slot_size ) != ( buf_size_t )1u )
    {
        success_flag |= ( SIXTRL_INT32_T )-4;
    }
    else if( NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) )
    {
        success_flag |= ( SIXTRL_INT32_T )-8;
    }

    if( ( success_flag != 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {
        atomic_or( ptr_success_flag, success_flag );
    }

    return;
}

/* ------------------------------------------------------------------------- */

__kernel void NS(Track_particles_elem_by_elem_opt_pp_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT elem_by_elem_buffer,
    SIXTRL_INT64_T  const until_turn,
    SIXTRL_UINT64_T const out_particle_blocks_offset,
    SIXTRL_INT64_T  const min_particle_id,
    SIXTRL_INT64_T  const max_particle_id,
    SIXTRL_INT64_T  const min_element_id,
    SIXTRL_INT64_T  const max_element_id,
    SIXTRL_INT64_T  const min_turn_id,
    SIXTRL_INT64_T  const max_turn_id,
    SIXTRL_INT64_T  const elem_by_elem_index_ordering,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t)                                   buf_size_t;
    typedef NS(particle_num_elements_t)                         num_element_t;
    typedef NS(particle_real_t)                                 real_t;
    typedef NS(particle_index_t)                                index_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*           obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*     obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)* ptr_particles_t;

    buf_size_t const  slot_size = ( buf_size_t )8u;
    SIXTRL_INT32_T success_flag = ( SIXTRL_INT32_T )-1;

    if( ( !NS(ManagedBuffer_needs_remapping)( particles_buffer, slot_size ) ) &&
        (  NS(ManagedBuffer_get_num_objects)(
                particles_buffer, slot_size ) == ( buf_size_t )1u ) &&
        ( !NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) ) )
    {
        num_element_t particle_index = ( num_element_t )get_global_id( 0 );
        num_element_t const stride   = ( num_element_t )get_global_size( 0 );

        obj_const_iter_t be_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
            belem_buffer, slot_size );

        obj_const_iter_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
            belem_buffer, slot_size );

        obj_iter_t out_obj_begin = NS(ManagedBuffer_get_objects_index_begin)(
            elem_by_elem_buf, slot_size );

        obj_iter_t pb_it = NS(ManagedBuffer_get_objects_index_begin)(
            particles_buffer, slot_size );

        obj_iter_t elem_by_elem_it = NS(ManagedBuffer_get_objects_index_begin)(
        elem_by_elem_buffer, slot_size );

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
        ptr_particles_t elem_by_elem_particles = SIXTRL_NULLPTR;

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

        SIXTRL_ASSERT( be_begin     != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( be_end       != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( out_obj_begin != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( pb_it != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( NS(Object_get_type_id)( pb_it ) !=
                       NS(OBJECT_TYPE_PARTICLES ) );

        SIXTRL_ASSERT( NS(Object_get_begin_addr)( pb_it ) !=
                       ( NS(buffer_addr_t) )0u );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
            particles_buffer, slot_size ) == ( buf_size_t )1u );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
            belem_buffer, slot_size ) > ( buf_size_t )0u );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)( elem_by_elem_buf ) >=
            ( out_particle_blocks_offset + NS(ManagedBuffer_get_num_objects)(
                belem_buffer, slot_size ) ) );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            particles_buffer, slot_size ) );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            elem_by_elem_buf, slot_size ) );

        SIXTRL_ASSERT( in_particles != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( in_particles->at_element_id != SIXTRL_NULLPTR );

        elem_by_elem_it = elem_by_elem_it + out_particle_blocks_offset;
        SIXTRL_ASSERT( NS(Object_get_type_id)( elem_by_elem_it ) ==
                    NS(OBJECT_TYPE_PARTICLES) );

        elem_by_elem_particles = ( ptr_particles_t )( uintptr_t
            )NS(Object_get_begin_addr)( elem_by_elem_it );

        for( ; particle_index < num_particles ; particle_index += stride )
        {
            index_t particle_id      = ( index_t )0u;
            index_t start_element_id = ( index_t )0u;
            index_t at_turn          = ( index_t )0u;

            success_flag = NS(Particles_copy_from_generic_addr_data)(
                &particles, 0, in_particles, particle_idx );

            start_element_id = NS(Particles_get_at_element_id_value)( &particls, 0 );
            particle_id = NS(Particles_get_particle_id_value)( &particles, 0 );
            at_turn = NS(Particles_get_at_turn_value)( &particles, 0 );

            while( ( success_flag == ( SIXTRL_INT32_T )0 ) &&
                   ( at_turn < until_turn ) )
            {
                obj_const_iter_t be_it = be_begin;

                for( ; be_it != be_end ; ++be_it )
                {
                    num_element_t const elem_by_elem_idx  =
                        NS(Track_element_by_element_get_out_particle_index)(
                            min_particle_id, max_particle_id, particle_id,
                            min_element_id,  max_element_id,
                            NS(Particles_get_at_element_id_value)( &particles, 0 ),
                            min_turn, max_turn, at_turn, elem_by_elem_index_ordering );

                    if( ( elem_by_elem_idx < ( num_element_t ) ) ||
                        ( elem_by_elem_idx >
                            ( num_element_t )elem_by_elem_particles->num_particles ) )
                    {
                        success_flag |= ( SIXTRL_INT32_T )-4;
                        break;
                    }

                    success_flag |= NS(Particles_copy_to_generic_addr_data)(
                        elem_by_elem_particles, elem_by_elem_idx, &particles, 0 );

                    success_flag |= NS(Track_particle_beam_element_obj)(
                        &particles, 0, be_it );

                    if( ( success_flag != 0 ) ||
                        ( NS(Particles_is_lost_value)( &particles, 0 ) ) )
                    {
                        success_flag |= -8;
                        break;
                    }
                }

                if( success_flag == ( SIXTRL_INT32_T )0 )
                {
                    NS(Particles_increment_at_turn_value)( &particles, 0 );
                    at_turn = NS(Particles_get_at_turn_value)( &particles, 0 );

                    NS(Particles_set_at_element_id_value)(
                        &particles, 0, start_element_id );
                }

                if( success_flag != ( SIXTRL_INT32_T )0 )
                {
                    break;
                }
            }

            success_flag |= NS(Particles_copy_to_generic_addr_data)(
                in_particles, particle_index, &particles, 0 );

            if( success_flag != ( SIXTRL_INT32_T )0 )
            {
                break;
            }
        }
    }
    else if( NS(ManagedBuffer_needs_remapping)( particles_buffer, slot_size) )
    {
        success_flag |= ( SIXTRL_INT32_T )-2;
    }
    else if( NS(ManagedBuffer_get_num_objects)(
                particles_buffer, slot_size ) != ( buf_size_t )1u )
    {
        success_flag |= ( SIXTRL_INT32_T )-4;
    }
    else if( NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) )
    {
        success_flag |= ( SIXTRL_INT32_T )-8;
    }
    else if( NS(ManagedBuffer_needs_remapping)( elem_by_elem_buf, slot_size ) )
    {
        success_flag |= ( SIXTRL_INT32_T )-16;
    }
    else if( ( NS(ManagedBuffer_get_num_objects)( belem_buffer, slot_size ) +
        out_particle_blocks_offset ) > NS(ManagedBuffer_get_num_objects)(
                elem_by_elem_buf, slot_size ) )
    {
        success_flag |= ( SIXTRL_INT32_T )-32;
    }

    if( ( success_flag != 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {
        atomic_or( ptr_success_flag, success_flag );
    }

    return;
}

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_OPTIMIZED_PRIV_PARTICLES_KERNEL_CL__ */

/* end: sixtracklib/opencl/kernels/track_particles_optimized_priv_particles.cl */
