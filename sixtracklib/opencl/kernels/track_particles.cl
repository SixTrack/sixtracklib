#ifndef SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_KERNEL_CL__
#define SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_KERNEL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/internal/particles_defines.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/track.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(Remap_particles_beam_elements_buffers_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements_buf );

__kernel void NS(Track_particles_beam_elements_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements_buf,
    SIXTRL_UINT64_T const num_turns );

/* ========================================================================= */

__kernel void NS(Remap_particles_beam_elements_buffers_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements_buf )
{
    typedef NS(buffer_size_t) buf_size_t;

    size_t const global_id   = get_global_id( 0 );
    size_t const global_size = get_global_size( 0 );

    size_t const gid_to_remap_particles_buffer     = ( size_t )0u;
    size_t const gid_to_remap_beam_elements_buffer = ( global_size > 1u )
        ? ( gid_to_remap_particles_buffer + 1u )
        : ( gid_to_remap_particles_buffer );

    if( global_id <= gid_to_remap_beam_elements_buffer )
    {
        buf_size_t const slot_size = ( buf_size_t )8u;

        SIXTRL_ASSERT( particles_buf != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( particles_buf != beam_elements_buf );

        if( global_id == gid_to_remap_particles_buffer )
        {
            NS(ManagedBuffer_remap)( particles_buf, slot_size )
        }

        if( global_id == gid_to_remap_beam_elements_buffer )
        {
            NS(ManagedBuffer_remap)( beam_elements_buf, slot_size );
        }
    }

    return;
}

/* ------------------------------------------------------------------------- */

__kernel void NS(Track_particles_beam_elements_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements_buf,
    SIXTRL_UINT64_T const num_turns )
{
    typedef NS(buffer_size_t)                                buf_size_t;
    typedef NS(particle_num_elements_t)                      num_element_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*        obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*  obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC     NS(Particles)*     ptr_particles_t;

    buf_size_t const slot_size = ( buf_size_t )8u;
    num_element_t particle_id  = ( num_element_t )get_global_id( 0 );
    num_element_t const stride = ( num_element_t )get_local_size( 0 );

    obj_const_iter_t be_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
        beam_elements_buf, slot_size );

    obj_const_iter_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
        beam_elements_buf, slot_size );

    ptr_particles_t particles = NS(ManagedBuffer_get_num_objects)(
        particles_buf, 0u );

    num_element_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping(
        particles_buf, slot_size ) );

    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping(
        beam_elements_buf, slot_size ) );

    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
        particles_buffer, slot_size ) == ( buf_size_t )1u );

    SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
        particles_buffer, slot_size ) );

    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );

    for( ; particle_id < num_particles ; particle_id += stride )
    {
        NS(Track_particle_until_turn_obj)(
            particles, particle_id, be_begin, be_end, num_turns );
    }

    return;
}

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_KERNEL_CL__ */

/* end: sixtracklib/opencl/kernels/track_particles.cl */
