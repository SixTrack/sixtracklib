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
    if( NS(Cuda_get_1d_thread_id_in_kernel)() == ( NS(buffer_size_t) )0u )
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
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_dbg_register )
{
    if( NS(Cuda_get_1d_thread_id_in_kernel)() == ( NS(buffer_size_t) )0u )
    {
        NS(arch_debugging_t) dbg = SIXTRL_ARCH_DEBUGGING_GENERAL_FAILURE;

        NS(BeamMonitor_assign_managed_output_buffer_debug)( beam_elem_buffer,
            output_buffer, min_turn_id, out_buffer_offset_index,
                slot_size, &dbg );

        if( ptr_dbg_register != SIXTRL_NULLPTR )
        {
            *ptr_dbg_register = dbg;
        }
    }
}

/* end sixtracklib/cuda/kernels/be_monitors_assign_out_buffer.cu */
