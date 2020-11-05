#ifndef SIXTRACKLIB_OPENCL_KERNELS_PROCESS_ASSIGN_ADDRESS_ITEMS_CL_H__
#define SIXTRACKLIB_OPENCL_KERNELS_PROCESS_ASSIGN_ADDRESS_ITEMS_CL_H__


__kernel void NS(AssignAddressItem_process_managed_buffer_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT map_buffer_begin,
    NS(buffer_size_t) const start_item_idx, NS(buffer_size_t) const item_idx_stride,
    NS(buffer_size_t) const map_buffer_slot_size,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT dest_buffer_begin,
    SIXTRL_UINT64_T const dest_slot_size,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT src_buffer_begin,
    NS(buffer_size_t) const src_slot_size )
{

}

#endif /* SIXTRACKLIB_OPENCL_KERNELS_PROCESS_ASSIGN_ADDRESS_ITEMS_CL_H__ */
