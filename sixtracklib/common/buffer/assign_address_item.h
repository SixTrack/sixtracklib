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

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t)
NS(AssignAddressItem_type_id)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(AssignAddressItem) *const SIXTRL_RESTRICT item );

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

SIXTRL_STATIC SIXTRL_FN void NS(AssignAddressItem_clear)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item );

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

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
NS(AssignAddressItem_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    NS(Object)* SIXTRL_RESTRICT obj );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
NS(AssignAddressItem_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN bool  NS(AssignAddressItem_dest_is_on_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item );

SIXTRL_STATIC SIXTRL_FN bool  NS(AssignAddressItem_dest_is_on_raw_memory)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item );

SIXTRL_STATIC SIXTRL_FN bool  NS(AssignAddressItem_src_is_on_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item );

SIXTRL_STATIC SIXTRL_FN bool  NS(AssignAddressItem_src_is_on_raw_memory)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_addr_t)*
NS(AssignAddressItem_dest_pointer_from_raw_memory)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT dest_buffer_begin );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_addr_t)*
NS(AssignAddressItem_dest_pointer_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT dest_buffer_begin,
    NS(buffer_size_t) const dest_slot_size );

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t)
NS(AssignAddressItem_src_pointer_addr_from_raw_memory)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT src_buffer_begin );

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t)
NS(AssignAddressItem_src_pointer_addr_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT src_buffer_begin,
    NS(buffer_size_t) const src_slot_size );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_perform_assignment_on_raw_memory)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT dest_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char
        const* SIXTRL_RESTRICT src_buffer_begin );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_perform_assignment_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT dest_buffer_begin,
    NS(buffer_size_t) const dest_slot_size,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT src_buffer_begin,
    NS(buffer_size_t) const src_slot_size );

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_assign_fixed_addr_on_raw_memory)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT dest_buffer_begin,
    NS(buffer_addr_t) const src_address );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_assign_fixed_addr_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* dest_buffer_begin,
    NS(buffer_size_t) const dest_slot_size,
    NS(buffer_addr_t) const src_address );

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_remap_assignment_on_raw_memory)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* dest_buffer_begin,
    NS(buffer_addr_diff_t) const remap_offset );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(AssignAddressItem_remap_assignment_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* dest_buffer_begin,
    NS(buffer_size_t) const dest_slot_size,
    NS(buffer_addr_diff_t) const remap_offset );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(AssignAddressItem_perform_assignment)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT dest_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer)
        *const SIXTRL_RESTRICT src_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(AssignAddressItem_assign_fixed_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT dest_buffer,
    NS(buffer_addr_t) const src_address );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(AssignAddressItem_remap_assignment)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT dest_buffer,
    NS(buffer_addr_diff_t) const remap_offset );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_addr_t)*
NS(AssignAddressItem_dest_pointer_from_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_addr_t)
NS(AssignAddressItem_src_pointer_addr_from_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
NS(AssignAddressItem_const_from_buffer)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_from_buffer)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(AssignAddressItem_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
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

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(AssignAddressItem_assign_all_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* assign_item_buffer,
    NS(buffer_size_t) const assign_slot_size,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* dest_buffer,
    NS(buffer_size_t) const dest_slot_size,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* src_buffer,
    NS(buffer_size_t) const src_slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(AssignAddressItem_assign_all)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT map_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT dest_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT src_buffer );

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(AssignAddressItem_copy)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT source );

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

SIXTRL_INLINE NS(object_type_id_t) NS(AssignAddressItem_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC const
        NS(AssignAddressItem) *const SIXTRL_RESTRICT item )
{
    ( void )item;
    return NS(OBJECT_TYPE_ASSIGN_ADDRESS_ITEM);
}

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
        item->dest_elem_type_id = NS(OBJECT_TYPE_NONE);
        item->dest_buffer_id    = SIXTRL_ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID;

        item->src_elem_type_id  = NS(OBJECT_TYPE_NONE);
        item->src_buffer_id     = SIXTRL_ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID;

        NS(AssignAddressItem_clear)( item );
    }

    return item;
}

SIXTRL_INLINE void NS(AssignAddressItem_clear)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT item )
{
    SIXTRL_ASSERT( item != SIXTRL_NULLPTR );

    item->dest_elem_index     = ( NS(buffer_size_t) )0u;
    item->dest_pointer_offset = ( NS(buffer_size_t) )0u;
    item->src_elem_index      = ( NS(buffer_size_t) )0u;
    item->src_pointer_offset  = ( NS(buffer_size_t) )0u;
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
NS(AssignAddressItem_perform_assignment_on_raw_memory)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT dest_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT src_buffer_begin )
{
    NS(buffer_addr_t) const src_address =
        NS(AssignAddressItem_src_pointer_addr_from_raw_memory)(
            item, src_buffer_begin );

    return NS(AssignAddressItem_assign_fixed_addr_on_raw_memory)(
        item, dest_buffer_begin, src_address );
}


SIXTRL_INLINE NS(arch_status_t)
NS(AssignAddressItem_perform_assignment_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT dest_buffer_begin,
    NS(buffer_size_t) const dest_slot_size,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT src_buffer_begin,
    NS(buffer_size_t) const src_slot_size )
{
    NS(buffer_addr_t) const src_address =
    NS(AssignAddressItem_src_pointer_addr_from_managed_buffer)( item,
        src_buffer_begin, src_slot_size );

    return NS(AssignAddressItem_assign_fixed_addr_on_managed_buffer)(
        item, dest_buffer_begin, dest_slot_size, src_address );
}

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_INLINE NS(arch_status_t)
NS(AssignAddressItem_assign_fixed_addr_on_raw_memory)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT dest_buffer_begin,
    NS(buffer_addr_t) const src_address )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_addr_t)* dest_ptr =
        NS(AssignAddressItem_dest_pointer_from_raw_memory)(
            item, dest_buffer_begin );

    if( dest_ptr != SIXTRL_NULLPTR )
    {
        *dest_ptr = src_address;
        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t)
NS(AssignAddressItem_assign_fixed_addr_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const
        NS(AssignAddressItem) *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* dest_buffer_begin,
    NS(buffer_size_t) const dest_slot_size, NS(buffer_addr_t) const src_addr )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_addr_t)* dest_ptr =
        NS(AssignAddressItem_dest_pointer_from_managed_buffer)(
            item, dest_buffer_begin, dest_slot_size );

    if( dest_ptr != SIXTRL_NULLPTR )
    {
        *dest_ptr = src_addr;
        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_INLINE NS(arch_status_t)
NS(AssignAddressItem_remap_assignment_on_raw_memory)(
    SIXTRL_BUFFER_DATAPTR_DEC const
        NS(AssignAddressItem) *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* dest_buffer_begin,
    NS(buffer_addr_diff_t) const remap_offset )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_addr_t)* dest_ptr =
    NS(AssignAddressItem_dest_pointer_from_raw_memory)(
        item, dest_buffer_begin );

    if( ( dest_ptr != SIXTRL_NULLPTR ) &&
        ( ( NS(ManagedBuffer_check_addr_arithmetic)(
            *dest_ptr, remap_offset, ( NS(buffer_size_t) )1u ) ) ||
          ( ( *dest_ptr == ( NS(buffer_addr_t) )0u ) &&
            ( remap_offset > ( NS(buffer_addr_diff_t) )0u ) ) ) )
    {
        *dest_ptr += remap_offset;
        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }
    else if( dest_ptr == SIXTRL_NULLPTR )
    {
        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t)
NS(AssignAddressItem_remap_assignment_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const
        NS(AssignAddressItem) *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* dest_buffer_begin,
    NS(buffer_size_t) const dest_slot_size,
    NS(buffer_addr_diff_t) const remap_offset )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_addr_t)* dest_ptr =
    NS(AssignAddressItem_dest_pointer_from_managed_buffer)(
        item, dest_buffer_begin, dest_slot_size );

    if( ( NS(AssignAddressItem_dest_is_on_buffer)( item ) ) &&
        ( dest_ptr != SIXTRL_NULLPTR ) &&
        ( ( NS(ManagedBuffer_check_addr_arithmetic)(
              *dest_ptr, remap_offset, dest_slot_size ) ) ||
          ( *dest_ptr == ( NS(buffer_addr_t) )0u ) ) )
    {
        *dest_ptr += remap_offset;
        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_INLINE bool NS(AssignAddressItem_dest_is_on_buffer)(
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
             ( NS(AssignAddressItem_dest_elem_index)( item ) ==
               ( NS(buffer_size_t) )0u ) );
}

SIXTRL_INLINE bool NS(AssignAddressItem_src_is_on_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item )
{
    return ( ( item != SIXTRL_NULLPTR ) &&
             ( !NS(AssignAddressItem_src_is_on_raw_memory)( item ) ) );
}

SIXTRL_INLINE bool NS(AssignAddressItem_src_is_on_raw_memory)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem) *const
        SIXTRL_RESTRICT item )
{
    return ( ( item != SIXTRL_NULLPTR ) &&
             ( NS(AssignAddressItem_src_buffer_id)( item ) ==
               SIXTRL_ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID ) &&
             ( NS(AssignAddressItem_src_elem_index)( item ) ==
               ( NS(buffer_size_t) )0u ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
NS(AssignAddressItem_const_from_obj_index)(
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
NS(AssignAddressItem_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* index_obj )
{
    return ( SIXTRL_BUFFER_OBJ_DATAPTR_DEC NS(AssignAddressItem)*
        )NS(AssignAddressItem_const_from_obj_index)( index_obj );
}


SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
NS(AssignAddressItem_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    return NS(AssignAddressItem_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( pbuffer, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)*
NS(AssignAddressItem_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size )
{
    return NS(AssignAddressItem_from_obj_index)(
        NS(ManagedBuffer_get_object)( pbuffer, be_index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_addr_t)*
NS(AssignAddressItem_dest_pointer_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT dest_buffer_begin,
    NS(buffer_size_t) const dest_slot_size )
{
    typedef NS(buffer_addr_t) addr_t;
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC addr_t* dest_ptr_t;

    dest_ptr_t dest_ptr = SIXTRL_NULLPTR;

    if( ( NS(AssignAddressItem_dest_is_on_buffer)( item ) ) &&
        ( dest_buffer_begin != SIXTRL_NULLPTR ) &&
        ( dest_slot_size > ( buf_size_t )0u ) )
    {
        SIXTRL_STATIC_VAR buf_size_t const ADDR_SIZE =
            ( buf_size_t )sizeof( addr_t );

        buf_size_t const elem_idx =
            NS(AssignAddressItem_dest_elem_index)( item );

        buf_size_t const ptr_offset =
            NS(AssignAddressItem_dest_pointer_offset)( item );

        SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* dest_obj = SIXTRL_NULLPTR;
        addr_t obj_addr = ( addr_t )0u;
        addr_t ptr_addr = ( addr_t )0u;

        SIXTRL_ASSERT( !NS(AssignAddressItem_dest_is_on_raw_memory)( item ) );

        SIXTRL_ASSERT( NS(AssignAddressItem_dest_elem_type_id)(
            item ) != NS(OBJECT_TYPE_NONE) );

        SIXTRL_ASSERT( dest_slot_size > ( buf_size_t )0u );
        SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
            dest_buffer_begin, dest_slot_size ) );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
            dest_buffer_begin, dest_slot_size ) > elem_idx );

        dest_obj = NS(ManagedBuffer_get_object)(
            dest_buffer_begin, elem_idx, dest_slot_size );

        obj_addr = NS(Object_get_begin_addr)( dest_obj );
        ptr_addr = obj_addr + ptr_offset;

        if( ( dest_obj != SIXTRL_NULLPTR ) && ( obj_addr != ( addr_t )0u ) &&
            ( ( ptr_addr % dest_slot_size ) == ( addr_t )0u ) &&
            ( ( ptr_addr % ADDR_SIZE ) == ( addr_t )0u ) &&
            ( NS(Object_get_size)( dest_obj ) >= ( ptr_offset + ADDR_SIZE ) ) &&
            ( NS(Object_get_type_id)( dest_obj ) ==
                NS(AssignAddressItem_dest_elem_type_id)( item ) ) )
        {
            dest_ptr = ( dest_ptr_t )( uintptr_t )ptr_addr;
        }
    }

    return dest_ptr;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_addr_t)*
NS(AssignAddressItem_dest_pointer_from_raw_memory)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT dest_buffer_begin )
{
    typedef NS(buffer_addr_t) addr_t;
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC addr_t* dest_ptr_t;

    dest_ptr_t dest_ptr = SIXTRL_NULLPTR;

    if( ( NS(AssignAddressItem_dest_is_on_raw_memory)( item ) ) &&
        ( dest_buffer_begin != SIXTRL_NULLPTR ) )
    {
        addr_t const obj_addr = ( addr_t )( uintptr_t )dest_buffer_begin;

        buf_size_t const ptr_offset =
            NS(AssignAddressItem_dest_pointer_offset)( item );

        addr_t const ptr_addr = obj_addr + ptr_offset;
        SIXTRL_ASSERT( obj_addr != ( addr_t )0u );
        SIXTRL_ASSERT( ( ptr_addr % sizeof( addr_t ) ) == ( addr_t )0u );
        SIXTRL_ASSERT( !NS(AssignAddressItem_dest_is_on_buffer)( item ) );
        dest_ptr = ( dest_ptr_t )( uintptr_t )ptr_addr;
    }

    return dest_ptr;
}

SIXTRL_INLINE NS(buffer_addr_t)
NS(AssignAddressItem_src_pointer_addr_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) addr_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    addr_t address = ( addr_t )0u;

    if( ( NS(AssignAddressItem_src_is_on_buffer)( item ) ) &&
        ( buffer_begin != SIXTRL_NULLPTR ) )
    {
        SIXTRL_STATIC_VAR buf_size_t const ADDR_SIZE =
            ( buf_size_t )sizeof( addr_t );

        buf_size_t const elem_idx =
            NS(AssignAddressItem_src_elem_index)( item );

        buf_size_t const ptr_offset =
            NS(AssignAddressItem_src_pointer_offset)( item );

        ptr_obj_t obj = SIXTRL_NULLPTR;
        addr_t obj_addr = ( addr_t )0u;
        addr_t ptr_addr = ( addr_t )0u;

        SIXTRL_ASSERT( !NS(AssignAddressItem_src_is_on_raw_memory)( item ) );
        SIXTRL_ASSERT( NS(AssignAddressItem_src_elem_type_id)( item ) !=
            NS(OBJECT_TYPE_NONE) );

        SIXTRL_ASSERT( slot_size > ( buf_size_t )0u );

        SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
            buffer_begin, slot_size ) );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
            buffer_begin, slot_size ) > elem_idx );

        obj = NS(ManagedBuffer_get_const_object)(
            buffer_begin, elem_idx, slot_size );

        obj_addr = NS(Object_get_begin_addr)( obj );
        ptr_addr = obj_addr + ptr_offset;

        if( ( obj != SIXTRL_NULLPTR ) && ( obj_addr != ( addr_t )0u ) &&
            ( ( ( ptr_addr ) % slot_size ) == ( addr_t )0u ) &&
            ( ( ( ptr_addr ) % ADDR_SIZE ) == ( addr_t )0u ) &&
            ( NS(Object_get_size)( obj ) >= ( ptr_offset + ADDR_SIZE ) ) &&
            ( NS(Object_get_type_id)( obj ) ==
              NS(AssignAddressItem_src_elem_type_id)( item ) ) )
        {
            address = ptr_addr;
        }
    }

    return address;
}

NS(buffer_addr_t) NS(AssignAddressItem_src_pointer_addr_from_raw_memory)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT src_buffer_begin )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) addr_t;

    addr_t src_addr = ( addr_t )0u;

    if( ( NS(AssignAddressItem_src_is_on_raw_memory)( item ) ) &&
        ( src_buffer_begin != SIXTRL_NULLPTR ) )
    {
        addr_t const src_obj_addr = ( addr_t )( uintptr_t )src_buffer_begin;
        buf_size_t const src_ptr_offset =
            NS(AssignAddressItem_src_pointer_offset)( item );

        addr_t const src_ptr_addr = src_obj_addr + src_ptr_offset;
        SIXTRL_ASSERT( !NS(AssignAddressItem_src_is_on_buffer)( item ) );

        if( src_obj_addr != ( addr_t )0u ) src_addr = src_ptr_addr;
    }

    return src_addr;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(arch_status_t) NS(AssignAddressItem_copy)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT source )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_ASSERT( ( NS(buffer_size_t) )0u ==
        NS(AssignAddressItem_num_dataptrs)( dest ) );

    SIXTRL_ASSERT( ( NS(buffer_size_t) )0u ==
        NS(AssignAddressItem_num_dataptrs)( source ) );

    if( ( dest != SIXTRL_NULLPTR ) && ( source != SIXTRL_NULLPTR ) &&
        ( dest != source ) )
    {
        *dest = *source;
        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }
    else if( ( dest != SIXTRL_NULLPTR ) && ( dest == source ) )
    {
        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BUFFER_ADDR_ASSIGNMENT_C99_H__ */

