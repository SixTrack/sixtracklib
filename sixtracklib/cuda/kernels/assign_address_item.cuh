#ifndef SIXTRACKLIB_CUDA_KERNELS_ASSIGN_ADDRESS_ITEM_CUDA_CUH__
#define SIXTRACKLIB_CUDA_KERNELS_ASSIGN_ADDRESS_ITEM_CUDA_CUH__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer/assign_address_item_kernel_impl.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

__global__ void NS(AssignAddressItem_process_managed_buffer_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT assign_buffer,
    NS(buffer_size_t) const assign_slot_size,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT dest_buffer,
    NS(buffer_size_t) const dest_slot_size,
    NS(arch_size_t) const dest_buffer_id,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT src_buffer,
    NS(buffer_size_t) const src_slot_size,
    NS(arch_size_t) const src_buffer_id );

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */
#endif /* SIXTRACKLIB_CUDA_KERNELS_ASSIGN_ADDRESS_ITEM_CUDA_CUH__ */
