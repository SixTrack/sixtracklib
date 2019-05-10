#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/kernels/be_monitors_assign_out_buffer.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>

    #include <cuda_runtime_api.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/be_monitor/output_buffer.h"
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/cuda_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__global__ void NS(BeamMonitor_assign_out_buffer_from_offset_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT output_buffer,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_offset_index,
    NS(buffer_size_t) const slot_size )
{
    if( NS(Cuda_get_1d_thread_id_in_kernel)() == ( size_t )0u )
    {
        NS(BeamMonitor_assign_managed_output_buffer)( beam_elem_buffer,
            output_buffer, min_turn_id, out_buffer_offset_index, slot_size );
    }
}

__global__ void NS(BeamMonitor_assign_out_buffer_from_offset_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT output_buffer,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_offset_index,
    NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(ctrl_debug_flag_t)* SIXTRL_RESTRICT ptr_debug_flag )
{
    typedef NS(ctrl_debug_flag_t) flag_t;
    typedef NS(buffer_size_t)     buf_size_t;

    flag_t debug_flag = SIXTRL_CONTROLLER_DEBUG_FLAG_OK;
    size_t const active_thread_id = ( size_t )0u;

    if( active_thread_id < NS(Cuda_get_total_num_threads_in_kernel)() )
    {
        if( NS(Cuda_get_1d_thread_id_in_kernel)() == active_thread_id )
        {
            NS(BeamMonitor_assign_managed_output_buffer_debug)(
                beam_elem_buffer, output_buffer, min_turn_id,
                    out_buffer_offset_index, slot_size, &debug_flag );
        }
    }
    else
    {
        debug_flag |= SIXTRL_CONTROLLER_DEBUG_FLAG_GENERAL_FAILURE;
    }

    NS(Cuda_handle_debug_flag_in_kernel)( ptr_debug_flag, debug_flag );
}

/* end sixtracklib/cuda/kernels/be_monitors_assign_out_buffer.cu */
