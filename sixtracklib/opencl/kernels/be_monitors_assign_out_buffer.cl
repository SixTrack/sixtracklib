#ifndef SIXTRACKLIB_OPENCL_KERNELS_BE_MONITORS_ASSIGN_OUT_BUFFER_CL__
#define SIXTRACKLIB_OPENCL_KERNELS_BE_MONITORS_ASSIGN_OUT_BUFFER_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/internal/particles_defines.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/output/output_buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(BeamMonitor_assign_out_buffer_from_offset_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer,
    SIXTRL_UINT64_T const out_particles_block_offset );

__kernel void NS(BeamMonitor_clear_all_line_obj_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements_buf );

/* ========================================================================= */

__kernel void NS(BeamMonitor_assign_out_buffer_from_offset_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer,
    SIXTRL_UINT64_T const out_particles_block_offset )
{
    typedef NS(buffer_size_t) buf_size_t;

    size_t const global_id   = get_global_id( 0 );
    size_t const global_size = get_global_size( 0 );
    size_t const gid_to_assign_out_buffer = ( size_t )0u;

    if( global_id == gid_to_assign_out_buffer )
    {
        buf_size_t const slot_size = ( buf_size_t )8u;

        SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
            beam_elements_buf, slot_size ) );

        SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
            out_buffer, slot_size ) );

        NS(BeamMonitor_assign_managed_particles_out_buffer)(
            beam_elements_buf, out_buffer, out_particles_block_offset, slot_size );
    }

    return;
}

__kernel void NS(BeamMonitor_clear_all_line_obj_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements_buf )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj_iter_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* ptr_beam_monitor_t;

    buf_size_t const slot_size = ( buf_size_t )8u;
    buf_size_t beam_element_id = get_global_id( 0 );
    buf_size_t const    stride = get_global_size( 0 );
    buf_size_t num_beam_elements = ( buf_size_t )0u;

    obj_iter_t be_begin = NS(ManagedBuffer_get_objects_index_begin)(
        beam_elements_buf, slot_size );

    SIXTRL_ASSERT( be_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
        beam_elements_buf, slot_size ) );

    num_beam_elements = NS(ManagedBuffer_get_num_objects)(
        beam_elements_buf, slot_size );

    SIXTRL_ASSERT( ( be_begin + num_beam_elements ) !=
        NS(ManagedBuffer_get_objects_index_end)(
            beam_elements_buf, slot_size ) );

    for( ; beam_element_id < num_beam_elements ; beam_element_id += stride )
    {
        obj_iter_t be_it = be_begin + beam_element_id;

        if( NS(Object_get_type_id)( be_it ) == NS(OBJECT_TYPE_BEAM_MONITOR) )
        {
            ptr_beam_monitor_t beam_monitor = ( ptr_beam_monitor_t)( uintptr_t
                )NS(Object_get_begin_addr)( be_it );

            SIXTRL_ASSERT( beam_monitor != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( NS(Object_get_size)( be_it ) >=
                           sizeof( NS(BeamMonitor) ) );

            NS(BeamMonitor_clear)( beam_monitor );
        }
    }

    return;
}

#endif /* SIXTRACKLIB_OPENCL_KERNELS_BE_MONITORS_ASSIGN_OUT_BUFFER_CL__ */

/* end: sixtracklib/opencl/kernels/be_monitors_assign_out_buffer.cl */
