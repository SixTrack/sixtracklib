#ifndef SIXTRACKLIB_OPENCL_KERNELS_BE_MONITORS_ASSIGN_IO_BUFFER_DEBUG_KERNEL_CL__
#define SIXTRACKLIB_OPENCL_KERNELS_BE_MONITORS_ASSIGN_IO_BUFFER_DEBUG_KERNEL_CL__

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

__kernel void NS(BeamMonitor_assign_io_buffer_from_offset_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT io_buffer,
    SIXTRL_UINT64_T const num_particles,
    SIXTRL_UINT64_T const io_particles_block_offset,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T* SIXTRL_RESTRICT ptr_success_flag );

/* ========================================================================= */

__kernel void NS(BeamMonitor_assign_io_buffer_from_offset_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT io_buffer,
    SIXTRL_UINT64_T const num_particles,
    SIXTRL_UINT64_T const io_particles_block_offset,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT32_T* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;
    buf_size_t const slot_size = ( size_t )8u;

    SIXTRL_INT32_T success_flag = ( SIXTRL_INT32_T )-1;

    if( ( num_particles > ( SIXTRL_UINT64_t )0u ) &&
        ( !NS(ManagedBuffer_needs_remapping)( beam_elements_buf, slot_size ) ) &&
        ( !NS(ManagedBuffer_needs_remapping)( io_buffer, slot_size ) ) &&
        (  NS(ManagedBuffer_get_num_objects)( io_buffer, slot_size ) >=
            io_particles_block_offset ) )
    {
        size_t const gid_to_assign_io_buffer = ( size_t )0u;
        size_t const global_id = get_global_id( 0 );

        success_flag = ( SIXTRL_INT32_T )0u;

        if( global_id == gid_to_assign_io_buffer )
        {
            success_flag = NS(BeamMonitor_assign_managed_io_buffer)(
                beam_elements_buf, io_buffer, num_particles,
                    io_particles_block_offset );
        }
    }

    if( ( success_flag != 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {
        atomic_or( ptr_success_flag, success_flag );
    }

    return;
}

#endif /* SIXTRACKLIB_OPENCL_KERNELS_BE_MONITORS_ASSIGN_IO_BUFFER_DEBUG_KERNEL_CL__ */

/* end: sixtracklib/opencl/kernels/be_monitors_assign_io_buffer_debug.cl */
