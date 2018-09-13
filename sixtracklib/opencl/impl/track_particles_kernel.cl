#ifndef SIXTRACKLIB_OPENCL_IMPL_MANAGED_TRACK_PARTICLES_KERNEL_OPENCL_CL__
#define SIXTRACKLIB_OPENCL_IMPL_MANAGED_TRACK_PARTICLES_KERNEL_OPENCL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/managed_buffer_minimal.h"
    #include "sixtracklib/common/impl/managed_buffer_remap.h"
    #include "sixtracklib/common/impl/particles_defines.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/track.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#pragma OPENCL_EXTENSION cl_khr_int32_extended_atomics

__kernel void NS(Remap_particles_beam_elements_buffers_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements_buf,
    SIXTRL_BUFFER_DATAPTR_DEC int* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;

    size_t const global_id                     = get_global_id( 0 );
    size_t const global_size                   = get_global_size( 0 );

    size_t const gid_to_remap_particles_buffer     = ( size_t )0u;
    size_t const gid_to_remap_beam_elements_buffer = ( global_size > 1u )
        ? ( gid_to_remap_particles_buffer + 1u )
        : ( gid_to_remap_particles_buffer );

    if( global_id <= gid_to_remap_beam_elements_buffer )
    {
        buf_size_t const slot_size = ( buf_size_t )8u;
        int success_flag = ( int )0u;

        if( global_id == gid_to_remap_particles_buffer )
        {
            if( ( particles_buf != SIXTRL_NULLPTR ) &&
                ( beam_elements_buf != particles_buf ) )
            {
                if( 0 != NS(ManagedBuffer_remap)( particles_buf, slot_size ) )
                {
                    success_flag |= -2;
                }
            }
            else
            {
                success_flag |= -1;
            }
        }

        if( global_id == gid_to_remap_beam_elements_buffer )
        {
            if( ( beam_elements_buf != SIXTRL_NULLPTR ) &&
                ( particles_buf != beam_elements_buf ) )
            {
                if( 0 != NS(ManagedBuffer_remap)(
                        beam_elements_buf, slot_size ) )
                {
                    success_flag |= -4;
                }
            }
            else
            {
                success_flag |= -1;
            }
        }

        if( ( success_flag != 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
        {

            atomic_or( ptr_success_flag, success_flag );
        }
    }

    return;
}


__kernel void NS(Track_particles_beam_elements_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements_buf,
    SIXTRL_UINT64_T const num_turns,
    SIXTRL_BUFFER_DATAPTR_DEC int* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*        obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*  obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC     NS(Particles)*     ptr_particles_t;

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
            ptr_particles_t particles = ( ptr_particles_t )(
                uintptr_t )NS(Object_get_begin_addr)( part_block_it );

            size_t const object_end_particle_id = object_begin_particle_id +
                NS(Particles_get_num_of_particles)( particles );

            SIXTRL_ASSERT( NS(Object_get_type_id)( part_block_it ) ==
                           NS(OBJECT_TYPE_PARTICLE) );

            if( ( global_particle_id <  object_end_particle_id   ) &&
                ( global_particle_id >= object_begin_particle_id ) )
            {
                size_t const particle_id =
                    global_particle_id - object_begin_particle_id;

                SIXTRL_UINT64_T turn = ( SIXTRL_UINT64_T )0u;

                SIXTRL_ASSERT( particle_id <
                    NS(Particles_get_num_of_particles)( particles ) );

                for( ; turn < num_turns ; ++turn )
                {
                    success_flag |= NS(Track_particle_beam_elements)(
                        particles, particle_id, be_begin, be_end );
                }
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

#endif /* SIXTRACKLIB_OPENCL_IMPL_MANAGED_TRACK_PARTICLES_KERNEL_OPENCL_CL__ */
/* end: sixtracklib/opencl/impl/track_particles_kernel.cl */
