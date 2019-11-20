#ifndef SIXTRACKLIB_COMMON_BUFFER_ASSIGN_ADDRESS_ITEM_KERNEL_IMPL_H__
#define SIXTRACKLIB_COMMON_BUFFER_ASSIGN_ADDRESS_ITEM_KERNEL_IMPL_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer/assign_address_item.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_assign_all_managed_buffer_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* assign_item_buffer,
    NS(buffer_size_t) const assign_slot_size,
    NS(buffer_size_t) const start_item_idx,
    NS(buffer_size_t) const item_idx_stride,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* dest_buffer,
    NS(buffer_size_t) const dest_slot_size,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* src_buffer,
    NS(buffer_size_t) const src_slot_size );

SIXTRL_INLINE NS(arch_status_t) NS(AssignAddressItem_assign_all_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* assign_item_buffer,
    NS(buffer_size_t) const assign_slot_size,
    NS(buffer_size_t) const start_item_idx,
    NS(buffer_size_t) const item_idx_stride,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* dest_buffer,
    NS(buffer_size_t) const dest_slot_size,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* src_buffer,
    NS(buffer_size_t) const src_slot_size )
{
    NS(buffer_size_t) const num_assign_items =
        NS(ManagedBuffer_get_num_of_objects)(
            assign_item_buffer, assign_slot_size );

    NS(buffer_size_t) idx = start_item_idx;

    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_SUCCESS;

    SIXTRL_ASSERT( assign_item_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( assign_slot_size > ( NS(buffer_size_t) )0u );
    SIXTRL_ASSERT( item_idx_stride  > ( NS(buffer_size_t) )0u );

    SIXTRL_ASSERT( dest_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( dest_slot_size > ( NS(buffer_size_t) )0u );

    SIXTRL_ASSERT( src_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( src_slot_size > ( NS(buffer_size_t) )0u );

    for( ; idx < num_assign_items ; idx += item_idx_stride )
    {
        SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const* item =
            NS(AssignAddressItem_const_from_managed_buffer)(
                assign_item_buffer, idx, assign_slot_size );

        if( item != SIXTRL_NULLPTR )
        {
            status |= NS(AssignAddressItem_perform_assignment_on_managed_buffer)(
                item, dest_buffer_begin, dest_slot_size,
                    src_buffer_begin, src_slot_size );
        }
    }

    return SIXTRL_ARCH_STATUS_SUCCESS;
}

#endif /* SIXTRACKLIB_COMMON_BUFFER_ASSIGN_ADDRESS_ITEM_KERNEL_IMPL_H__ */
/* end: */
