#ifndef SIXTRACKLIB_OPENCL_KERNELS_BE_MONITORS_ASSIGN_OUT_BUFFER_CL__
#define SIXTRACKLIB_OPENCL_KERNELS_BE_MONITORS_ASSIGN_OUT_BUFFER_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/be_monitor/output_buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(BeamMonitor_assign_out_buffer_from_offset_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belements_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer,
    SIXTRL_INT64_T  const min_turn_id,
    SIXTRL_UINT64_T const out_buffer_offset_index,
    SIXTRL_UINT64_T const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    if( ( buf_size_t )get_global_id( 0 ) == ( buf_size_t )0u )
    {
        NS(BeamMonitor_assign_managed_output_buffer)( belements_buf,
            out_buffer, min_turn_id, out_buffer_offset_index, slot_size );
    }
}

__kernel void NS(BeamMonitor_clear_all_line_obj_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belements_buf,
    SIXTRL_UINT64_T const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    if( ( buf_size_t )get_global_id( 0 ) == ( buf_size_t )0u )
    {
        NS(BeamMonitor_clear_all_on_managed_buffer)( belements_buf, slot_size );
    }
}

#endif /* SIXTRACKLIB_OPENCL_KERNELS_BE_MONITORS_ASSIGN_OUT_BUFFER_CL__ */
/* end: sixtracklib/opencl/kernels/be_monitors_assign_out_buffer.cl */
