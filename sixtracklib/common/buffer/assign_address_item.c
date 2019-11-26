#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer/assign_address_item.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <string.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
    #include "sixtracklib/common/buffer/assign_address_item_kernel_impl.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

NS(arch_status_t) NS(AssignAddressItem_next_buffer_id)( void )
{
    static NS(arch_size_t) next_buffer_id =
        ( NS(arch_size_t) )SIXTRL_ARCH_MIN_USER_DEFINED_BUFFER_ID;

    return ( next_buffer_id < NS(ARCH_MAX_USER_DEFINED_BUFFER_ID) )
        ? next_buffer_id++ : NS(ARCH_ILLEGAL_BUFFER_ID);
}

NS(arch_status_t) NS(AssignAddressItem_perform_assignment)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT dest_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer)
        *const SIXTRL_RESTRICT src_buffer )
{
    return NS(AssignAddressItem_perform_assignment_on_managed_buffer)( item,
        NS(Buffer_get_data_begin)( dest_buffer ),
        NS(Buffer_get_slot_size)( dest_buffer ),
        NS(Buffer_get_const_data_begin)( src_buffer ),
        NS(Buffer_get_slot_size)( src_buffer ) );
}

NS(arch_status_t) NS(AssignAddressItem_assign_fixed_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT dest_buffer,
    NS(buffer_addr_t) const src_address )
{
    return NS(AssignAddressItem_assign_fixed_addr_on_managed_buffer)( item,
        NS(Buffer_get_data_begin)( dest_buffer ),
        NS(Buffer_get_slot_size)( dest_buffer ), src_address );
}

NS(arch_status_t) NS(AssignAddressItem_remap_assignment)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT dest_buffer,
    NS(buffer_addr_diff_t) const remap_offset )
{
    return NS(AssignAddressItem_remap_assignment_on_managed_buffer)( item,
        NS(Buffer_get_data_begin)( dest_buffer ),
        NS(Buffer_get_slot_size)( dest_buffer ), remap_offset );
}

/* ------------------------------------------------------------------------- */

SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_addr_t)*
NS(AssignAddressItem_dest_pointer_from_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    return NS(AssignAddressItem_dest_pointer_from_managed_buffer)(
        item, NS(Buffer_get_data_begin)( buffer ),
            NS(Buffer_get_slot_size)( buffer ) );
}

NS(buffer_addr_t) NS(AssignAddressItem_src_pointer_addr_from_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(AssignAddressItem_src_pointer_addr_from_managed_buffer)(
        item, NS(Buffer_get_const_data_begin)( buffer ),
            NS(Buffer_get_slot_size)( buffer ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
NS(AssignAddressItem_const_from_buffer)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    return NS(AssignAddressItem_const_from_obj_index)(
        NS(Buffer_get_const_object)( buffer, index ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_from_buffer)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    return NS(AssignAddressItem_from_obj_index)( NS(Buffer_get_object)(
        buffer, index ) );
}

/* ------------------------------------------------------------------------- */

bool NS(AssignAddressItem_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    ( void )ptr_requ_dataptrs;

    return NS(Buffer_can_add_trivial_object)( buffer,
        sizeof( NS(AssignAddressItem) ), ptr_requ_objects, ptr_requ_slots );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* NS(AssignAddressItem_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    NS(AssignAddressItem) item;
    NS(AssignAddressItem_preset)( &item );

    SIXTRL_ASSERT( ( NS(buffer_size_t) )0u ==
        NS(AssignAddressItem_num_dataptrs)( &item ) );

    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_trivial_object)( buffer,
            &item, sizeof( item ), NS(AssignAddressItem_type_id)( &item ) ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* NS(AssignAddressItem_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(object_type_id_t) const dest_elem_type_id,
    NS(buffer_size_t) const dest_buffer_id,
    NS(buffer_size_t) const dest_elem_index,
    NS(buffer_size_t) const dest_pointer_offset,
    NS(object_type_id_t) const src_elem_type_id,
    NS(buffer_size_t) const src_buffer_id,
    NS(buffer_size_t) const src_elem_index,
    NS(buffer_size_t) const src_pointer_offset )
{
    NS(AssignAddressItem) item;
    NS(AssignAddressItem_preset)( &item );
    NS(AssignAddressItem_set_dest_elem_type_id)( &item, dest_elem_type_id );
    NS(AssignAddressItem_set_dest_buffer_id)( &item, dest_buffer_id );
    NS(AssignAddressItem_set_dest_elem_index)( &item, dest_elem_index );
    NS(AssignAddressItem_set_dest_pointer_offset)( &item, dest_pointer_offset );
    NS(AssignAddressItem_set_src_elem_type_id)( &item, src_elem_type_id );
    NS(AssignAddressItem_set_src_buffer_id)( &item, src_buffer_id );
    NS(AssignAddressItem_set_src_elem_index)( &item, src_elem_index );
    NS(AssignAddressItem_set_src_pointer_offset)( &item, src_pointer_offset );

    SIXTRL_ASSERT( ( NS(buffer_size_t) )0u ==
        NS(AssignAddressItem_num_dataptrs)( &item ) );

    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_trivial_object)( buffer,
            &item, sizeof( item ), NS(AssignAddressItem_type_id)( &item ) ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* NS(AssignAddressItem_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const NS(AssignAddressItem) *const SIXTRL_RESTRICT item )
{
    SIXTRL_ASSERT( ( NS(buffer_size_t) )0u ==
        NS(AssignAddressItem_num_dataptrs)( item ) );

    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_trivial_object)( buffer,
            item, sizeof( NS(AssignAddressItem) ),
                NS(AssignAddressItem_type_id)( item ) ) );
}

/* ------------------------------------------------------------------------- */

bool NS(AssignAddressItem_compare_less_ext)(
    const NS(AssignAddressItem) *const SIXTRL_RESTRICT lhs,
    const NS(AssignAddressItem) *const SIXTRL_RESTRICT rhs )
{
    return NS(AssignAddressItem_compare_less)( lhs, rhs );
}

bool NS(AssignAddressItem_are_equal_ext)(
    const NS(AssignAddressItem) *const SIXTRL_RESTRICT lhs,
    const NS(AssignAddressItem) *const SIXTRL_RESTRICT rhs )
{
    return NS(AssignAddressItem_are_equal)( lhs, rhs );
}

bool NS(AssignAddressItem_are_not_equal_ext)(
    const NS(AssignAddressItem) *const SIXTRL_RESTRICT lhs,
    const NS(AssignAddressItem) *const SIXTRL_RESTRICT rhs )
{
    return NS(AssignAddressItem_are_not_equal)( lhs, rhs );
}

bool NS(AssignAddressItem_dest_src_are_equal_ext)(
    const NS(AssignAddressItem) *const SIXTRL_RESTRICT lhs,
    const NS(AssignAddressItem) *const SIXTRL_RESTRICT rhs )
{
    return NS(AssignAddressItem_dest_src_are_equal)( lhs, rhs );
}

bool NS(AssignAddressItem_dest_src_are_not_equal_ext)(
    const NS(AssignAddressItem) *const SIXTRL_RESTRICT lhs,
    const NS(AssignAddressItem) *const SIXTRL_RESTRICT rhs )
{
    return NS(AssignAddressItem_dest_src_are_not_equal)( lhs, rhs );
}

/* end: sixtracklib/common/buffer/assign_address_item.c */
