#ifndef SIXTRACKLIB_OPENCL_KERNELS_BE_MONITORS_ASSIGN_OUT_BUFFER_DEBUG_KERNEL_CL__
#define SIXTRACKLIB_OPENCL_KERNELS_BE_MONITORS_ASSIGN_OUT_BUFFER_DEBUG_KERNEL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/be_monitor/output_buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(BeamMonitor_assign_out_buffer_from_offset_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT output_buffer,
    SIXTRL_INT64_T  const min_turn_id,
    SIXTRL_UINT64_T const out_buffer_index_offset,
    SIXTRL_UINT64_T const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flag )
{
    if( ( size_t )get_global_id( 0 ) == ( size_t )0u )
    {
        NS(arch_debugging_t) flags = SIXTRL_ARCH_DEBUGGING_GENERAL_FAILURE;

         NS(BeamMonitor_assign_managed_output_buffer_debug)( beam_elem_buffer,
            output_buffer, min_turn_id, out_buffer_index_offset, slot_size,
                &flags );

        if( ptr_status_flag != SIXTRL_NULLPTR )
        {
            *ptr_status_flag = flags;
        }
    }
}

__kernel void NS(BeamMonitor_clear_all_line_obj_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements_buf,
    SIXTRL_UINT64_T const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flag )
{
    if( ( size_t )get_global_id( 0 ) == ( size_t )0u )
    {
        NS(arch_debugging_t) flags = SIXTRL_ARCH_DEBUGGING_GENERAL_FAILURE;
        NS(BeamMonitor_clear_all_on_managed_buffer_debug)(
            beam_elements_buf, slot_size, &flags );

        if( ptr_status_flag != SIXTRL_NULLPTR )
        {
            *ptr_status_flag = flags;
        }
    }
}

#endif /* SIXTRACKLIB_OPENCL_KERNELS_BE_MONITORS_ASSIGN_OUT_BUFFER_DEBUG_KERNEL_CL__ */

/* end: sixtracklib/opencl/kernels/be_monitors_assign_out_buffer_debug.cl */
