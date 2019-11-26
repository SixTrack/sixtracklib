#ifndef SIXTRACKLIB_COMMON_BUFFER_ASSIGN_ADDRESS_ITEM_KERNEL_IMPL_H__
#define SIXTRACKLIB_COMMON_BUFFER_ASSIGN_ADDRESS_ITEM_KERNEL_IMPL_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer/assign_address_item.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_perform_address_assignment_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT assign_buffer,
    NS(buffer_size_t) const assign_slot_size,
    NS(buffer_size_t) const start_item_idx,
    NS(buffer_size_t) const item_idx_stride,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT dest_buffer,
    NS(buffer_size_t) const dest_slot_size,
    NS(arch_size_t) const dest_buffer_id,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT src_buffer,
    NS(buffer_size_t) const src_slot_size,
    NS(arch_size_t) const src_buffer_id );

/* ************************************************************************* */
/* ************************************************************************* */

SIXTRL_INLINE NS(arch_status_t)
NS(AssignAddressItem_perform_address_assignment_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT assign_buffer,
    NS(buffer_size_t) const assign_slot_size,
    NS(buffer_size_t) const start_item_idx,
    NS(buffer_size_t) const item_idx_stride,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT dest_buffer,
    NS(buffer_size_t) const dest_slot_size,
    NS(arch_size_t) const dest_buffer_id,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT src_buffer,
    NS(buffer_size_t) const src_slot_size,
    NS(arch_size_t) const src_buffer_id )
{
    NS(buffer_size_t) const num_assign_items =
        NS(ManagedBuffer_get_num_objects)( assign_buffer, assign_slot_size );

    NS(buffer_size_t) idx = start_item_idx;
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_SUCCESS;

    SIXTRL_ASSERT( assign_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( assign_slot_size > ( NS(buffer_size_t) )0u );
    SIXTRL_ASSERT( item_idx_stride  > ( NS(buffer_size_t) )0u );

    SIXTRL_ASSERT( dest_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( dest_buffer_id != SIXTRL_ARCH_ILLEGAL_BUFFER_ID );

    SIXTRL_ASSERT( src_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( src_buffer_id != SIXTRL_ARCH_ILLEGAL_BUFFER_ID );

    for( ; idx < num_assign_items ; idx += item_idx_stride )
    {
        SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const* item =
            NS(AssignAddressItem_const_from_managed_buffer)(
                assign_buffer, idx, assign_slot_size );

        if( ( item != SIXTRL_NULLPTR ) &&
            ( NS(AssignAddressItem_src_buffer_id)( item ) == src_buffer_id ) &&
            ( NS(AssignAddressItem_dest_buffer_id)( item ) ==
                dest_buffer_id ) )
        {
            NS(buffer_addr_t) const src_addr = (
                NS(AssignAddressItem_src_is_on_buffer)( item ) )
                ? NS(AssignAddressItem_src_pointer_addr_from_managed_buffer)(
                    item, src_buffer, src_slot_size )
                : NS(AssignAddressItem_src_pointer_addr_from_raw_memory)(
                    item, src_buffer );

            if( NS(AssignAddressItem_dest_is_on_buffer)( item ) )
            {
                status |= NS(AssignAddressItem_assign_fixed_addr_on_managed_buffer)(
                    item, dest_buffer, dest_slot_size, src_addr );
            }
            else if( NS(AssignAddressItem_dest_is_on_raw_memory)( item ) )
            {
                status |= NS(AssignAddressItem_assign_fixed_addr_on_raw_memory)(
                    item, dest_buffer, src_addr );
            }
            else
            {
                status |= SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
            }
        }
    }

    return status;
}

#endif /* SIXTRACKLIB_COMMON_BUFFER_ASSIGN_ADDRESS_ITEM_KERNEL_IMPL_H__ */
/* end: */
