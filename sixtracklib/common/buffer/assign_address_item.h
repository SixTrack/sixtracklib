#ifndef SIXTRACKLIB_COMMON_BUFFER_ADDR_ASSIGNMENT_C99_H__
#define SIXTRACKLIB_COMMON_BUFFER_ADDR_ASSIGNMENT_C99_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#if !defined( SIXTRL_ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID )
    #define SIXTRL_ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID 0xffffffffffffffff
#endif /* !defined( SIXTRL_ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID ) */

#if !defined( _GPUCODE )
    SIXTRL_STATIC_VAR const NS(buffer_size_t)
    NS(ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID) =
        ( NS(buffer_size_t) )SIXTRL_ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID;
#endif /* !defined( _GPUCODE ) */

typedef struct NS(AssignAddressItem)
{
    NS(object_type_id_t) dest_elem_type_id     SIXTRL_ALIGN( 8 );
    NS(buffer_size_t)    dest_buffer_id        SIXTRL_ALIGN( 8 );
    NS(buffer_size_t)    dest_elem_index       SIXTRL_ALIGN( 8 );
    NS(buffer_size_t)    dest_pointer_offset   SIXTRL_ALIGN( 8 );
    NS(object_type_id_t) src_elem_type_id      SIXTRL_ALIGN( 8 );
    NS(buffer_size_t)    src_buffer_id         SIXTRL_ALIGN( 8 );
    NS(buffer_size_t)    src_elem_index        SIXTRL_ALIGN( 8 );
    NS(buffer_size_t)    src_pointer_offset    SIXTRL_ALIGN( 8 );
}
NS(AssignAddressItem);

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(AssignAddressItem_num_dataptrs)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(AssignAddressItem_num_slots)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_preset)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(AssignAddressItem)* SIXTRL_RESTRICT item );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t)
NS(AssignAddressItem_dest_elem_type_id)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(AssignAddressItem_dest_buffer_id)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(AssignAddressItem_dest_elem_index)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(AssignAddressItem_dest_pointer_offset)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item );


SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t)
NS(AssignAddressItem_src_elem_type_id)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(AssignAddressItem_src_buffer_id)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(AssignAddressItem_src_elem_index)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(AssignAddressItem_src_pointer_offset)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN void NS(AssignAddressItem_set_dest_elem_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item,
    NS(object_type_id_t) const type_id );

SIXTRL_STATIC SIXTRL_FN void NS(AssignAddressItem_set_dest_buffer_id)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const dest_buffer_id );

SIXTRL_STATIC SIXTRL_FN void NS(AssignAddressItem_set_dest_elem_index)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const dest_elem_index );

SIXTRL_STATIC SIXTRL_FN void NS(AssignAddressItem_set_dest_pointer_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const dest_pointer_offset );


SIXTRL_STATIC SIXTRL_FN void NS(AssignAddressItem_set_src_elem_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item,
    NS(object_type_id_t) const type_id );

SIXTRL_STATIC SIXTRL_FN void NS(AssignAddressItem_set_src_buffer_id)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const src_buffer_id );

SIXTRL_STATIC SIXTRL_FN void NS(AssignAddressItem_set_src_elem_index)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const src_elem_index );

SIXTRL_STATIC SIXTRL_FN void NS(AssignAddressItem_set_src_pointer_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const src_pointer_offset );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
NS(AssignAddressItem_managed_buffer_get_const_item)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_managed_buffer_get_item)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_addr_t)*
NS(AssignAddressItem_managed_buffer_dest_ptr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT dest_buffer_begin,
    NS(buffer_size_t) const dest_slot_size );

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t)
NS(AssignAddressItem_managed_buffer_src_pointer_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT src_buffer_begin,
    NS(buffer_size_t) const src_slot_size );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_managed_buffer_perform_assignment)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT dest_buffer_begin,
    NS(buffer_size_t) const dest_slot_size,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT src_buffer_begin,
    NS(buffer_size_t) const src_slot_size );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_managed_buffer_assign_fixed_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* dest_buffer_begin,
    NS(buffer_size_t) const dest_slot_size,
    NS(buffer_addr_t) const src_address );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_managed_buffer_remap_assignment)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* dest_buffer_begin,
    NS(buffer_size_t) const dest_slot_size,
    NS(buffer_addr_diff_t) const remap_offset );

SIXTRL_STATIC SIXTRL_FN bool  NS(AssignAddressItem_dest_is_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item );

SIXTRL_STATIC SIXTRL_FN bool  NS(AssignAddressItem_dest_is_on_raw_memory)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item );

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
NS(AssignAddressItem_buffer_get_const_item)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index );

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC
NS(AssignAddressItem)* NS(AssignAddressItem_buffer_get_item)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_perform_assignment)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT dest_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT src_buffer );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_assign_fixed_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT dest_buffer,
    NS(buffer_addr_t) const src_address );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_remap_assignment)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT dest_buffer,
    NS(buffer_addr_diff_t) const remap_offset );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN bool NS(AssignAddressItem_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(object_type_id_t) const dest_elem_type_id,
    NS(buffer_size_t) const dest_buffer_id,
    NS(buffer_size_t) const dest_elem_index,
    NS(buffer_size_t) const dest_pointer_offset,
    NS(object_type_id_t) const src_elem_type_id,
    NS(buffer_size_t) const src_buffer_id,
    NS(buffer_size_t) const src_elem_index,
    NS(buffer_size_t) const src_pointer_offset );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item );

#endif /* !defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #if !defined( _GPUCODE )
    #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */


SIXTRL_INLINE NS(buffer_size_t)
NS(AssignAddressItem_num_dataptrs)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item )
{
    ( void )item;
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(AssignAddressItem_num_slots)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t extent = NS(ManagedBuffer_get_slot_based_length)(
        sizeof( NS(AssignAddressItem) ), slot_size );

    ( void )item;

    SIXTRL_ASSERT( ( slot_size == ZERO ) || ( ( extent % slot_size ) == ZERO ) );
    return ( slot_size > ZERO ) ? ( extent / slot_size ) : ( ZERO );

}

SIXTRL_INLINE SIXTRL_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_preset)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(AssignAddressItem)* SIXTRL_RESTRICT item )
{
    if( item != SIXTRL_NULLPTR )
    {
        item->dest_elem_type_id   = ::NS(OBJECT_TYPE_NONE);
        item->dest_buffer_id      = SIXTRL_ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID;
        item->dest_elem_index     = ( NS(buffer_size_t) )0u;
        item->dest_pointer_offset = ( NS(buffer_size_t) )0u;

        item->src_elem_type_id    = ::NS(OBJECT_TYPE_NONE);
        item->src_buffer_id       = SIXTRL_ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID;
        item->src_elem_index      = ( NS(buffer_size_t) )0u;
        item->src_pointer_offset  = ( NS(buffer_size_t) )0u;
    }

    return item;
}

SIXTRL_INLINE NS(object_type_id_t) NS(AssignAddressItem_dest_elem_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item )
{
    return ( item != SIXTRL_NULLPTR )
        ? item->dest_elem_type_id : NS(OBJECT_TYPE_NONE);
}

SIXTRL_INLINE NS(buffer_size_t) NS(AssignAddressItem_dest_buffer_id)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item )
{
    return ( item != SIXTRL_NULLPTR )
        ? item->dest_buffer_id : SIXTRL_ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(AssignAddressItem_dest_elem_index)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item )
{
    return ( item != SIXTRL_NULLPTR )
        ? item->dest_elem_index : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(AssignAddressItem_dest_pointer_offset)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item )
{
    return ( item != SIXTRL_NULLPTR )
        ? item->dest_pointer_offset : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(object_type_id_t) NS(AssignAddressItem_src_elem_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item )
{
    return ( item != SIXTRL_NULLPTR )
        ? item->src_elem_type_id : NS(OBJECT_TYPE_NONE);
}

SIXTRL_INLINE NS(buffer_size_t) NS(AssignAddressItem_src_buffer_id)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item )
{
    return ( item != SIXTRL_NULLPTR )
        ? item->src_buffer_id : SIXTRL_ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(AssignAddressItem_src_elem_index)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item )
{
    return ( item != SIXTRL_NULLPTR )
        ? item->src_elem_index : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(AssignAddressItem_src_pointer_offset)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item )
{
    return ( item != SIXTRL_NULLPTR )
        ? item->src_pointer_offset : ( NS(buffer_size_t) )0u;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(AssignAddressItem_set_dest_elem_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item,
    NS(object_type_id_t) const type_id )
{
    if( item != SIXTRL_NULLPTR ) item->dest_elem_type_id = type_id;
}

SIXTRL_INLINE void NS(AssignAddressItem_set_dest_buffer_id)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const dest_buffer_id )
{
    if( item != SIXTRL_NULLPTR ) item->dest_buffer_id = dest_buffer_id;
}

SIXTRL_INLINE void NS(AssignAddressItem_set_dest_elem_index)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const dest_elem_index )
{
    if( item != SIXTRL_NULLPTR ) item->dest_elem_index = dest_elem_index;
}

SIXTRL_INLINE void NS(AssignAddressItem_set_dest_pointer_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const dest_pointer_offset )
{
    if( item != SIXTRL_NULLPTR )
        item->dest_pointer_offset = dest_pointer_offset;
}

SIXTRL_INLINE void NS(AssignAddressItem_set_src_elem_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item,
    NS(object_type_id_t) const type_id )
{
    if( item != SIXTRL_NULLPTR ) item->src_elem_type_id = type_id;
}

SIXTRL_INLINE void NS(AssignAddressItem_set_src_buffer_id)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const src_buffer_id )
{
    if( item != SIXTRL_NULLPTR ) item->src_buffer_id = src_buffer_id;
}

SIXTRL_INLINE void NS(AssignAddressItem_set_src_elem_index)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const src_elem_index )
{
    if( item != SIXTRL_NULLPTR ) item->src_elem_index = src_elem_index;
}

SIXTRL_INLINE void NS(AssignAddressItem_set_src_pointer_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const src_ptr_offset )
{
    if( item != SIXTRL_NULLPTR ) item->src_pointer_offset = src_ptr_offset;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(arch_status_t)
NS(AssignAddressItem_managed_buffer_perform_assignment)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT dest_buffer_begin,
    NS(buffer_size_t) const dest_slot_size,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT src_buffer_begin,
    NS(buffer_size_t) const src_slot_size )
{
    typedef NS(buffer_addr_t) addr_t;

    addr_t src_address = NS(AssignAddressItem_managed_buffer_src_pointer_addr)(
        item, src_buffer_begin, src_slot_size );

    return NS(AssignAddressItem_managed_buffer_assign_fixed_addr)(
        item, dest_buffer_begin, dest_slot_size, src_address );
}

SIXTRL_INLINE NS(arch_status_t)
NS(AssignAddressItem_managed_buffer_assign_fixed_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* dest_buffer_begin,
    NS(buffer_size_t) const dest_slot_size,
    NS(buffer_addr_t) const src_address )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_addr_t)* dest_ptr =
        NS(AssignAddressItem_managed_buffer_dest_ptr)(
            item, dest_buffer_begin, dest_slot_size );

    if( dest_ptr != SIXTRL_NULLPTR )
    {
        *dest_ptr = src_address;
        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t)
NS(AssignAddressItem_managed_buffer_remap_assignment)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* dest_buffer_begin,
    NS(buffer_size_t) const dest_slot_size,
    NS(buffer_addr_diff_t) const remap_offset )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_addr_t)* dest_ptr =
    NS(AssignAddressItem_managed_buffer_dest_ptr)(
        item, dest_buffer_begin, dest_slot_size );

    if( ( dest_ptr != SIXTRL_NULLPTR ) &&
        ( ( NS(AssignAddressItem_dest_is_on_raw_memory)( item ) ) ||
          ( NS(ManagedBuffer_check_addr_arithmetic)(
              *dest_ptr, remap_offset, dest_slot_size ) ) ||
          ( *dest_ptr == ::NS(buffer_addr_t){ 0 } ) ) )
    {
        *dest_ptr += remap_offset;
        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE bool NS(AssignAddressItem_dest_is_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item )
{
    return ( ( item != SIXTRL_NULLPTR) &&
             ( !NS(AssignAddressItem_dest_is_on_raw_memory)( item ) ) );
}

SIXTRL_INLINE bool NS(AssignAddressItem_dest_is_on_raw_memory)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item )
{
    return ( ( item != SIXTRL_NULLPTR ) &&
             ( NS(AssignAddressItem_dest_buffer_id)( item ) ==
               SIXTRL_ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID ) &&
             ( NS(AssignAddressItem_dest_elem_type_id)( item ) ==
               NS(OBJECT_TYPE_NONE) ) &&
             ( NS(AssignAddressItem_dest_elem_index)( item ) ==
               ( NS(buffer_size_t) )0u ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
NS(BufferIndex_get_const_assign_address_item)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const index_obj )
{
    typedef NS(AssignAddressItem) item_t;
    typedef SIXTRL_BUFFER_OBJ_DATAPTR_DEC item_t const* ptr_to_item_t;
    ptr_to_item_t ptr_to_item = SIXTRL_NULLPTR;

    if( ( index_obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( index_obj ) ==
          NS(OBJECT_TYPE_ASSIGN_ADDRESS_ITEM) ) &&
        ( NS(Object_get_size)( index_obj ) >= sizeof( item_t ) ) )
    {
        ptr_to_item = ( ptr_to_item_t )( uintptr_t
            )NS(Object_get_begin_addr)( index_obj );
    }

    return ptr_to_item;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
NS(BufferIndex_get_assign_address_item)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* index_obj )
{
    return ( SIXTRL_BUFFER_OBJ_DATAPTR_DEC NS(AssignAddressItem)*
        )NS(BufferIndex_get_const_assign_address_item)( index_obj );
}


SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
NS(AssignAddressItem_managed_buffer_get_const_item)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    return NS(BufferIndex_get_const_assign_address_item)(
        NS(ManagedBuffer_get_const_object)( pbuffer, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_managed_buffer_get_item)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size )
{
    return NS(BufferIndex_get_assign_address_item)(
        NS(ManagedBuffer_get_object)( pbuffer, be_index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_addr_t)*
NS(AssignAddressItem_managed_buffer_dest_ptr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT dest_buffer_begin,
    NS(buffer_size_t) const dest_slot_size )
{
    typedef NS(buffer_addr_t) addr_t;
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC addr_t* dest_ptr_t;

    dest_ptr_t dest_ptr = SIXTRL_NULLPTR;

    if( ( item != SIXTRL_NULLPTR ) && ( dest_buffer_begin != SIXTRL_NULLPTR ) )
    {
        SIXTRL_STATIC_VAR buf_size_t const ADDR_SIZE =
            ( buf_size_t )sizeof( addr_t );

        if( NS(AssignAddressItem_dest_buffer_id)( item ) !=
                SIXTRL_ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID )
        {
            buf_size_t const dest_elem_idx =
                NS(AssignAddressItem_dest_elem_index)( item );

            buf_size_t const dest_ptr_offset =
                NS(AssignAddressItem_dest_pointer_offset)( item );

            SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* dest_obj = SIXTRL_NULLPTR;
            addr_t dest_obj_addr = ( addr_t )0u;
            addr_t dest_ptr_addr = ( addr_t )0u;

            SIXTRL_ASSERT( NS(AssignAddressItem_dest_elem_type_id)(
                item ) != NS(OBJECT_TYPE_NONE) );

            SIXTRL_ASSERT( dest_slot_size > ( buf_size_t )0u );

            SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
                dest_buffer_begin, dest_slot_size ) );

            SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
                dest_buffer_begin, dest_slot_size ) > dest_elem_idx );

            dest_obj = NS(ManagedBuffer_get_object)(
                dest_buffer_begin, dest_elem_idx, dest_slot_size );

            dest_obj_addr = NS(Object_get_begin_addr)( dest_obj );
            dest_ptr_addr = dest_obj_addr + dest_ptr_offset;

            if( ( dest_obj != SIXTRL_NULLPTR ) &&
                ( dest_obj_addr != ( addr_t )0u ) &&
                ( ( dest_ptr_addr % dest_slot_size ) == ( addr_t )0u ) &&
                ( ( dest_ptr_addr % ADDR_SIZE ) == ( addr_t )0u ) &&
                ( NS(Object_get_size)( dest_obj ) >=
                    ( dest_ptr_offset + ADDR_SIZE ) ) &&
                ( NS(Object_get_type_id)( dest_obj ) ==
                  NS(AssignAddressItem_dest_elem_type_id)( item ) ) )
            {
                dest_ptr = ( dest_ptr_t )( uintptr_t )dest_ptr_addr;
            }
        }
        else
        {
            addr_t const dest_obj_addr =
                ( addr_t )( uintptr_t )dest_buffer_begin;

            buf_size_t const dest_ptr_offset =
                NS(AssignAddressItem_dest_pointer_offset)( item );

            addr_t const dest_ptr_addr = dest_obj_addr + dest_ptr_offset;

            SIXTRL_ASSERT( dest_slot_size == ( buf_size_t )0u );
            SIXTRL_ASSERT( dest_obj_addr != ( addr_t )0u );
            SIXTRL_ASSERT( ( ( dest_obj_addr + dest_ptr_offset ) %
                                sizeof( addr_t ) ) == ( addr_t )0u );
            SIXTRL_ASSERT( NS(AssignAddressItem_dest_elem_type_id)( item ) ==
                           NS(OBJECT_TYPE_NONE) );

            dest_ptr = ( dest_ptr_t )( uintptr_t )dest_ptr_addr;
        }
    }

    return dest_ptr;
}

SIXTRL_INLINE NS(buffer_addr_t)
NS(AssignAddressItem_managed_buffer_src_pointer_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT src_buffer_begin,
    NS(buffer_size_t) const src_slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) addr_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_src_obj_t;

    NS(buffer_addr_t) src_address = ( NS(buffer_addr_t) )0u;

    if( ( item != SIXTRL_NULLPTR ) && ( src_buffer_begin != SIXTRL_NULLPTR ) )
    {
        SIXTRL_STATIC_VAR buf_size_t const ADDR_SIZE =
            ( buf_size_t )sizeof( addr_t );

        if( ( NS(AssignAddressItem_src_buffer_id)( item ) !=
                SIXTRL_ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID ) )
        {
            buf_size_t const src_elem_idx =
                NS(AssignAddressItem_src_elem_index)( item );

            buf_size_t const src_ptr_offset =
                NS(AssignAddressItem_src_pointer_offset)( item );

            ptr_src_obj_t src_obj = SIXTRL_NULLPTR;
            addr_t src_obj_addr = ( addr_t )0u;
            addr_t src_ptr_addr = ( addr_t )0u;

            SIXTRL_ASSERT( NS(AssignAddressItem_src_elem_type_id)(
                item ) != NS(OBJECT_TYPE_NONE) );

            SIXTRL_ASSERT( src_slot_size > ( buf_size_t )0u );

            SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
                src_buffer_begin, src_slot_size ) );

            SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
                src_buffer_begin, src_slot_size ) > src_elem_idx );

            src_obj = NS(ManagedBuffer_get_const_object)(
                src_buffer_begin, src_elem_idx, src_slot_size );

            src_obj_addr = NS(Object_get_begin_addr)( src_obj );
            src_ptr_addr = src_obj_addr + src_ptr_offset;

            if( ( src_obj != SIXTRL_NULLPTR ) &&
                ( src_obj_addr != ( addr_t )0u ) &&
                ( ( ( src_ptr_addr ) % src_slot_size ) == ( addr_t )0u ) &&
                ( ( ( src_ptr_addr ) % ADDR_SIZE ) == ( addr_t )0u ) &&
                ( NS(Object_get_size)( src_obj ) >= (
                    src_ptr_offset + ADDR_SIZE ) ) &&
                ( NS(Object_get_type_id)( src_obj ) ==
                  NS(AssignAddressItem_src_elem_type_id)( item ) ) )
            {
                src_address = src_ptr_addr;
            }
        }
        else
        {
            addr_t const src_obj_addr = ( addr_t )( uintptr_t )src_buffer_begin;
            buf_size_t const src_ptr_offset =
                NS(AssignAddressItem_src_pointer_offset)( item );

            addr_t const src_ptr_addr = src_obj_addr + src_ptr_offset;

            SIXTRL_ASSERT( src_slot_size == ( buf_size_t )0u );
            SIXTRL_ASSERT( NS(AssignAddressItem_src_elem_type_id)( item ) ==
                           NS(OBJECT_TYPE_NONE) );

            if( ( src_obj_addr != ( addr_t )0u ) &&
                ( ( src_ptr_addr % ADDR_SIZE ) == ( addr_t )0u ) )
            {
                src_address = src_ptr_addr;
            }
        }
    }

    return src_address;
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
NS(AssignAddressItem_buffer_get_const_item)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* ptr_raw_t;
    return NS(AssignAddressItem_managed_buffer_get_const_item)(
        ( ptr_raw_t )( uintptr_t )NS(Buffer_get_data_begin_addr)( buffer ),
        index, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC
NS(AssignAddressItem)* NS(AssignAddressItem_buffer_get_item)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_raw_t;
    return NS(AssignAddressItem_managed_buffer_get_item)(
        ( ptr_raw_t )( uintptr_t )NS(Buffer_get_data_begin_addr)( buffer ),
        be_index, NS(Buffer_get_slot_size)( buffer ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_perform_assignment)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT dest_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT src_buffer )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* dest_buffer_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* src_buffer_t;

    dest_buffer_t dest_buffer_begin = ( dest_buffer_t )( uintptr_t
        )NS(Buffer_get_data_begin_addr)( dest_buffer );

    src_buffer_t src_buffer_begin = ( src_buffer_t )( uintptr_t
        )NS(Buffer_get_data_begin_addr)( src_buffer );

    return NS(AssignAddressItem_managed_buffer_perform_assignment)(
        item, dest_buffer_begin, NS(Buffer_get_slot_size)( dest_buffer ),
        src_buffer_begin, NS(Buffer_get_slot_size)( src_buffer ) );
}

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_assign_fixed_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT dest_buffer,
    NS(buffer_addr_t) const src_address )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* dest_buffer_t;
    dest_buffer_t dest_buffer_begin = ( dest_buffer_t )( uintptr_t
        )NS(Buffer_get_data_begin_addr)( dest_buffer );

    return NS(AssignAddressItem_managed_buffer_assign_fixed_addr)(
        item, dest_buffer_begin, NS(Buffer_get_slot_size)( dest_buffer ),
            src_address );
}

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_remap_assignment)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT dest_buffer,
    NS(buffer_addr_diff_t) const remap_offset )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* dest_buffer_t;
    dest_buffer_t dest_buffer_begin = ( dest_buffer_t )( uintptr_t
        )NS(Buffer_get_data_begin_addr)( dest_buffer );

    return NS(AssignAddressItem_managed_buffer_remap_assignment)(
        item, dest_buffer_begin, NS(Buffer_get_slot_size)( dest_buffer ),
            remap_offset );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE bool NS(AssignAddressItem_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    typedef NS(AssignAddressItem) item_t;
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_dataptrs =
        NS(AssignAddressItem_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts = SIXTRL_NULLPTR;
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return NS(Buffer_can_add_object)( buffer, sizeof( item_t ), num_dataptrs,
        sizes, counts, requ_objects, requ_slots, requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(AssignAddressItem) item_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC item_t* ptr_to_item_t;
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_dataptrs =
        NS(AssignAddressItem_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    item_t temp_obj;
    NS(AssignAddressItem_preset)( &temp_obj );

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return ( ptr_to_item_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( item_t ),
            NS(OBJECT_TYPE_ASSIGN_ADDRESS_ITEM), num_dataptrs, offsets,
                sizes, counts ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(object_type_id_t) const dest_type_id,
    NS(buffer_size_t) const dest_buffer_id,
    NS(buffer_size_t) const dest_elem_index,
    NS(buffer_size_t) const dest_ptr_offset,
    NS(object_type_id_t) const src_type_id,
    NS(buffer_size_t) const src_buffer_id,
    NS(buffer_size_t) const src_elem_index,
    NS(buffer_size_t) const src_ptr_offset )
{
    typedef NS(AssignAddressItem) item_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC item_t* ptr_to_item_t;
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_dataptrs =
        NS(AssignAddressItem_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    item_t temp_obj;
    NS(AssignAddressItem_preset)( &temp_obj );

    NS(AssignAddressItem_set_dest_elem_type_id)( &temp_obj, dest_type_id );
    NS(AssignAddressItem_set_dest_buffer_id)( &temp_obj, dest_buffer_id );
    NS(AssignAddressItem_set_dest_elem_index)( &temp_obj, dest_elem_index );
    NS(AssignAddressItem_set_dest_pointer_offset)( &temp_obj, dest_ptr_offset );

    NS(AssignAddressItem_set_src_elem_type_id)( &temp_obj, src_type_id );
    NS(AssignAddressItem_set_src_buffer_id)( &temp_obj, src_buffer_id );
    NS(AssignAddressItem_set_src_elem_index)( &temp_obj, src_elem_index );
    NS(AssignAddressItem_set_src_pointer_offset)( &temp_obj, src_ptr_offset );

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return ( ptr_to_item_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( item_t ),
            NS(OBJECT_TYPE_ASSIGN_ADDRESS_ITEM), num_dataptrs, offsets,
                sizes, counts ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item )
{
    return NS(AssignAddressItem_add)( buffer,
        NS(AssignAddressItem_dest_elem_type_id)( item ),
        NS(AssignAddressItem_dest_buffer_id)( item ),
        NS(AssignAddressItem_dest_elem_index)( item ),
        NS(AssignAddressItem_dest_pointer_offset)( item ),
        NS(AssignAddressItem_src_elem_type_id)( item ),
        NS(AssignAddressItem_src_buffer_id)( item ),
        NS(AssignAddressItem_src_elem_index)( item ),
        NS(AssignAddressItem_src_pointer_offset)( item ) );
}

#endif /* !defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BUFFER_ADDR_ASSIGNMENT_C99_H__ */
