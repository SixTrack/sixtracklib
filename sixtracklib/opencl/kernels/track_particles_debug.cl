#ifndef SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_DEBUG_KERNEL_CL__
#define SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_DEBUG_KERNEL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/internal/particles_defines.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/track.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#pragma OPENCL_EXTENSION cl_khr_int32_extended_atomics

__kernel void NS(Track_particles_single_turn_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer,
    SIXTRL_INT64_T const increment_turn,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T*
        SIXTRL_RESTRICT ptr_success_flag );

__kernel void NS(Track_particles_until_turn_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer, SIXTRL_INT64_T const until_turn,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T*
        SIXTRL_RESTRICT ptr_success_flag );

__kernel void NS(Track_particles_elem_by_elem_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT elem_by_elem_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_INT64_T const until_turn,
    SIXTRL_UINT64_T const out_buffer_index_offset,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T*
        SIXTRL_RESTRICT ptr_success_flag );

/* ========================================================================= */

__kernel void NS(Track_particles_single_turn_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer, SIXTRL_INT64_T const increment_turn,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T*
        SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t)                                buf_size_t;
    typedef NS(particle_num_elements_t)                      num_element_t;
    typedef NS(particle_index_t)                             index_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*  obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC     NS(Particles)*     ptr_particles_t;

    buf_size_t const  slot_size = ( buf_size_t )8u;
    SIXTRL_INT32_T success_flag = ( SIXTRL_INT32_T )-1;

    if( ( !NS(ManagedBuffer_needs_remapping)( particles_buffer, slot_size) ) &&
        (  NS(ManagedBuffer_get_num_objects)(
                particles_buffer, slot_size ) == ( buf_size_t )1u ) &&
        ( !NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) ) )
    {
        num_element_t idx = ( num_element_t )get_global_id( 0 );
        num_element_t const stride = ( num_element_t )get_global_size( 0 );

        obj_const_iter_t be_begin =
            NS(ManagedBuffer_get_const_objects_index_begin)(
                belem_buffer, slot_size );

        obj_const_iter_t be_end =
            NS(ManagedBuffer_get_const_objects_index_end)(
                belem_buffer, slot_size );

        ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
            particles_buffer, 0u, slot_size );

        num_element_t const num_particles =
            NS(Particles_get_num_of_particles)( particles );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
            belem_buffer, slot_size ) > ( buf_size_t )0u );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            particles_buffer, slot_size ) );

        SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );

        success_flag = ( SIXTRL_INT32_T )0u;

        for( ; idx < num_particles ; idx += stride )
        {
            index_t const start_beam_element_id =
                NS(Particles_get_at_element_id_value)( particles, idx );

            success_flag |= NS(Track_particle_beam_elements_obj)(
                particles, idx, be_begin, be_end );

            if( ( success_flag   == ( SIXTRL_INT32_T )0 ) &&
                ( increment_turn != ( SIXTRL_INT64_T )0 ) &&
                ( NS(Particles_is_not_lost_value)( particles, idx ) ) )
            {
                NS(Particles_increment_at_turn_value)( particles, idx );
                NS(Particles_set_at_element_id_value)(
                    particles, idx, start_beam_element_id );
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

    if( ( success_flag != 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {
        atomic_or( ptr_success_flag, success_flag );
    }

    return;
}

/* ------------------------------------------------------------------------- */

__kernel void NS(Track_particles_until_turn_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer, SIXTRL_INT64_T const until_turn,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T*
        SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t)                                buf_size_t;
    typedef NS(particle_num_elements_t)                      num_element_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*  obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC     NS(Particles)*     ptr_particles_t;

    buf_size_t const  slot_size = ( buf_size_t )8u;
    SIXTRL_INT32_T success_flag = ( SIXTRL_INT32_T )-1;

    if( ( !NS(ManagedBuffer_needs_remapping)( particles_buffer, slot_size) ) &&
        (  NS(ManagedBuffer_get_num_objects)(
                particles_buffer, slot_size ) == ( buf_size_t )1u ) &&
        ( !NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) ) )
    {
        num_element_t idx = ( num_element_t )get_global_id( 0 );
        num_element_t const stride = ( num_element_t )get_global_size( 0 );

        obj_const_iter_t be_begin =
            NS(ManagedBuffer_get_const_objects_index_begin)(
                belem_buffer, slot_size );

        obj_const_iter_t be_end =
            NS(ManagedBuffer_get_const_objects_index_end)(
                belem_buffer, slot_size );

        ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
            particles_buffer, 0u, slot_size );

        num_element_t const num_particles =
            NS(Particles_get_num_of_particles)( particles );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
            belem_buffer, slot_size ) > ( buf_size_t )0u );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            particles_buffer, slot_size ) );

        SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );

        success_flag = ( SIXTRL_INT32_T )0u;

        for( ; idx < num_particles ; idx += stride )
        {
            success_flag |= NS(Track_particle_until_turn_obj)(
                particles, idx, be_begin, be_end, until_turn );
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

__kernel void NS(Track_particles_elem_by_elem_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT elem_by_elem_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT elem_by_elem_config, SIXTRL_INT64_T const until_turn,
    SIXTRL_UINT64_T const out_buffer_index_offset,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t)                                buf_size_t;
    typedef NS(particle_num_elements_t)                      num_element_t;
    typedef NS(particle_index_t)                             index_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*  obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC     NS(Particles)*     ptr_particles_t;

    buf_size_t const slot_size = ( buf_size_t )8u;
    SIXTRL_INT32_T success = ( SIXTRL_INT32_T )-1;

    if( ( !NS(ManagedBuffer_needs_remapping)( elem_by_elem_buffer,
                                              slot_size ) ) &&
        (  NS(ManagedBuffer_get_num_objects)(
                elem_by_elem_buffer, slot_size ) > out_buffer_index_offset ) &&
        ( !NS(ManagedBuffer_needs_remapping)( particles_buffer,
                                              slot_size) ) &&
        (  NS(ManagedBuffer_get_num_objects)(
                particles_buffer, slot_size ) >= ( buf_size_t )1u ) &&
        ( !NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) ) )
    {
        num_element_t idx = ( num_element_t )get_global_id( 0 );
        num_element_t const stride = ( num_element_t )get_global_size( 0 );
        buf_size_t const slot_size = ( buf_size_t )8u;

        obj_const_iter_t be_begin =
            NS(ManagedBuffer_get_const_objects_index_begin)(
                belem_buffer, slot_size );

        obj_const_iter_t be_end =
            NS(ManagedBuffer_get_const_objects_index_end)(
                belem_buffer, slot_size );

        ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
            particles_buffer, 0u, slot_size );

        num_element_t const num_particles =
            NS(Particles_get_num_of_particles)( particles );

        NS(ElemByElemConfig_assign_managed_particles_out_buffer)(
            elem_by_elem_config, elem_by_elem_buffer,
                out_buffer_index_offset, slot_size );

        SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( be_begin  != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( be_end    != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            particles_buffer, slot_size ) );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            elem_by_elem_buffer, slot_size ) );

        success = ( SIXTRL_INT32_T )0u;

        for( ; idx < num_particles ; idx += stride )
        {
            success |= NS(Track_particle_element_by_element_until_turn_objs)(
                particles, idx, elem_by_elem_config,
                    be_begin, be_end, until_turn );
        }
    }
    else if( NS(ManagedBuffer_needs_remapping)( particles_buffer, slot_size) )
    {
        success |= ( SIXTRL_INT32_T )-2;
    }
    else if( NS(ManagedBuffer_get_num_objects)(
                particles_buffer, slot_size ) != ( buf_size_t )1u )
    {
        success |= ( SIXTRL_INT32_T )-4;
    }
    else if( NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) )
    {
        success |= ( SIXTRL_INT32_T )-8;
    }
    else if( NS(ManagedBuffer_needs_remapping)(
            elem_by_elem_buffer, slot_size ) )
    {
        success |= ( SIXTRL_INT32_T )-16;
    }

    if( ( success != 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {
        atomic_or( ptr_success_flag, success );
    }

    return;
}

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_DEBUG_KERNEL_CL__ */

/* end: sixtracklib/opencl/kernels/track_particles_debug.cl */
