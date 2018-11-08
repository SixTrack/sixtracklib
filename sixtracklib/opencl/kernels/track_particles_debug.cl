#ifndef SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_DEBUG_KERNEL_CL__
#define SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_DEBUG_KERNEL_CL__

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

__kernel void NS(Track_particles_single_turn_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*  SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*  SIXTRL_RESTRICT belem_buf,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T* SIXTRL_RESTRICT ptr_success_flag );

__kernel void NS(Track_particles_until_turn_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT  particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT  belem_buf,
    SIXTRL_INT64_T const turn,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T* SIXTRL_RESTRICT ptr_success_flag );

__kernel void NS(Track_particles_elem_by_elem_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT elem_by_elem_buf,
    SIXTRL_UINT64_T const io_particle_blocks_offset,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T* SIXTRL_RESTRICT ptr_success_flag );

/* ========================================================================= */

__kernel void NS(Track_particles_single_turn_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*  SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*  SIXTRL_RESTRICT belem_buf,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t)                                buf_size_t;
    typedef NS(particle_num_elements_t)                      num_element_t;
    typedef NS(particle_index_t)                             index_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*        obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*  obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC     NS(Particles)*     ptr_particles_t;

    buf_size_t const  slot_size = ( buf_size_t )8u;
    SIXTRL_INT32_T success_flag = ( SIXTRL_INT32_T )-1;

    if( ( !NS(ManagedBuffer_needs_remapping)( particles_buf, slot_size) ) &&
        (  NS(ManagedBuffer_get_num_objects)(
                particles_buf, slot_size ) == ( buf_size_t )1u ) &&
        ( !NS(ManagedBuffer_needs_remapping)( belem_buf, slot_size ) ) )
    {
        num_element_t particle_id   = ( num_element_t )get_global_id( 0 );
        num_element_t const stride  = ( num_element_t )get_local_size( 0 );

        obj_const_iter_t be_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
            belem_buf, slot_size );

        obj_const_iter_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
            belem_buf, slot_size );

        ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
        particles_buf, 0u, slot_size );

        num_element_t const num_particles =
            NS(Particles_get_num_of_particles)( particles );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
            belem_buf, slot_size ) > ( buf_size_t )0u );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            particles_buffer, slot_size ) );

        SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );

        success_flag = ( SIXTRL_INT32_T )0u;

        for( ; particle_id < num_particles ; particle_id += stride )
        {
            success_flag |= NS(Track_particle_beam_elements_obj)(
                particles, particle_id, be_begin, be_end );

            NS(Track_particle_increment_at_turn)( particles, particle_id );
        }
    }

    if( ( success_flag != 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {
        atomic_or( ptr_success_flag, success_flag );
    }

    return;
}

/* ------------------------------------------------------------------------- */

__kernel void NS(Track_particles_until_turn_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belem_buf,
    SIXTRL_INT64_T const turn,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t)                                buf_size_t;
    typedef NS(particle_num_elements_t)                      num_element_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*        obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*  obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC     NS(Particles)*     ptr_particles_t;

    buf_size_t const  slot_size = ( buf_size_t )8u;
    SIXTRL_INT32_T success_flag = ( SIXTRL_INT32_T )-1;

    if( ( !NS(ManagedBuffer_needs_remapping)( particles_buf, slot_size) ) &&
        (  NS(ManagedBuffer_get_num_objects)(
                particles_buf, slot_size ) == ( buf_size_t )1u ) &&
        ( !NS(ManagedBuffer_needs_remapping)( belem_buf, slot_size ) ) )
    {
        num_element_t particle_id  = ( num_element_t )get_global_id( 0 );
        num_element_t const stride = ( num_element_t )get_local_size( 0 );

        obj_const_iter_t be_begin =
            NS(ManagedBuffer_get_const_objects_index_begin)(
                belem_buf, slot_size );

        obj_const_iter_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
            belem_buf, slot_size );

        ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
        particles_buf, 0u, slot_size );

        num_element_t const num_particles =
            NS(Particles_get_num_of_particles)( particles );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
            belem_buf, slot_size ) > ( buf_size_t )0u );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            particles_buffer, slot_size ) );

        SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );

        success_flag = ( SIXTRL_INT32_T )0u;

        for( ; particle_id < num_particles ; particle_id += stride )
        {
            success_flag |= NS(Track_particle_until_turn_obj)(
                particles, particle_id, be_begin, be_end, turn );
        }
    }

    if( ( success_flag != 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {
        atomic_or( ptr_success_flag, success_flag );
    }

    return;
}

/* ------------------------------------------------------------------------- */

__kernel void NS(Track_particles_elem_by_elem_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belem_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT elem_by_elem_buf,
    SIXTRL_UINT64_T const io_particle_blocks_offset,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t)                                buf_size_t;
    typedef NS(particle_num_elements_t)                      num_element_t;
    typedef NS(particle_index_t)                             index_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*        obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*  obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC     NS(Particles)*     ptr_particles_t;


    buf_size_t const  slot_size = ( buf_size_t )8u;
    SIXTRL_INT32_T success_flag = ( SIXTRL_INT32_T )-1;

    if( ( !NS(ManagedBuffer_needs_remapping)( elem_by_elem_buf, slot_size ) ) &&
        ( !NS(ManagedBuffer_needs_remapping)( particles_buf, slot_size) ) &&
        (  NS(ManagedBuffer_get_num_objects)(
                particles_buf, slot_size ) == ( buf_size_t )1u ) &&
        ( !NS(ManagedBuffer_needs_remapping)( belem_buf, slot_size ) ) &&
        ( ( NS(ManagedBuffer_get_num_objects)( belem_buf, slot_size ) +
            io_particle_blocks_offset ) <= NS(ManagedBuffer_get_num_objects)(
                elem_by_elem_buf, slot_size ) ) )
    {
        buf_size_t const slot_size = ( buf_size_t )8u;
        num_element_t particle_id  = ( num_element_t )get_global_id( 0 );
        num_element_t const stride = ( num_element_t )get_local_size( 0 );

        obj_const_iter_t be_begin =
            NS(ManagedBuffer_get_const_objects_index_begin)(
                belem_buf, slot_size );

        obj_const_iter_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
            belem_buf, slot_size );

        ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
        particles_buf, 0u, slot_size );

        obj_iter_t io_obj_begin = NS(ManagedBuffer_get_objects_index_begin)(
            elem_by_elem_buf, slot_size );

        num_element_t const num_particles =
            NS(Particles_get_num_of_particles)( particles );

        SIXTRL_ASSERT( be_begin     != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( be_end       != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( io_obj_begin != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            elem_by_elem_buf, slot_size ) );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
            belem_buf, slot_size ) > ( buf_size_t )0u );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            particles_buffer, slot_size ) );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            elem_by_elem_buf, slot_size ) );

        SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );

        io_obj_begin = io_obj_begin + io_particle_blocks_offset;

        SIXTRL_ASSERT( NS(Object_get_type_id)( io_obj_begin ) ==
                       NS(OBJECT_TYPE_PARTICLE) );

        SIXTRL_ASSERT( NS(Object_get_begin_addr)( io_obj_begin ) != 0u );
        SIXTRL_ASSERT( NS(Particles_get_num_of_particles)(
            ( ptr_particles_t )( uintptr_t )NS(Object_get_begin_addr)(
                io_obj_begin ) ) >= num_particles );

        success_flag = ( SIXTRL_INT32_T )0u;

        for( ; particle_id < num_particles ; particle_id += stride )
        {
            success_flag |= NS(Track_particle_element_by_element_obj)(
                particles, particle_id,
                NS(Particles_get_at_element_id_value)( particles, particle_id ),
                be_begin, be_end, io_obj_begin );
        }
    }

    if( ( success_flag != 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {
        atomic_or( ptr_success_flag, success_flag );
    }

    return;
}

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_DEBUG_KERNEL_CL__ */

/* end: sixtracklib/opencl/kernels/track_particles_debug.cl */
