#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/kernels/assign_address_item.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>

    #include <cuda_runtime_api.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer/assign_address_item.h"
    #include "sixtracklib/common/buffer/assign_address_item_kernel_impl.h"
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/cuda_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__global__ void NS(AssignAddressItem_process_managed_buffer_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT assign_buffer,
    NS(buffer_size_t) const assign_slot_size,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT dest_buffer,
    NS(buffer_size_t) const dest_slot_size,
    NS(arch_size_t) const dest_buffer_id,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT src_buffer,
    NS(buffer_size_t) const src_slot_size,
    NS(arch_size_t) const src_buffer_id )
{
    NS(buffer_size_t) const start_idx = NS(Cuda_get_1d_thread_id_in_kernel)();
    NS(buffer_size_t) const stride = NS(Cuda_get_total_num_threads_in_kernel)();

    NS(AssignAddressItem_perform_address_assignment_kernel_impl)(
        assign_buffer, assign_slot_size, start_idx, stride,
        dest_buffer, dest_slot_size, dest_buffer_id,
        src_buffer,  src_slot_size,  src_buffer_id );
}
