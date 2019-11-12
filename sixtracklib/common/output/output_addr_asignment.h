#ifndef SIXTRACKLIB_COMMON_OUTPUT_OUTPUT_ADDR_ASSIGNMENT_C99_H__
#define SIXTRACKLIB_COMMON_OUTPUT_OUTPUT_ADDR_ASSIGNMENT_C99_H__

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

typedef struct NS(OutAddrAssignmentItem)
{
    NS(object_type_id_t) elem_type_id          SIXTRL_ALIGN( 8 );
    NS(buffer_size_t)    elem_index            SIXTRL_ALIGN( 8 );
    NS(buffer_size_t)    dest_pointer_offset   SIXTRL_ALIGN( 8 );
    NS(buffer_size_t)    out_buffer_index      SIXTRL_ALIGN( 8 );
}
NS(OutAddrAssignmentItem);

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(OutAddrAssignmentItem_num_dataptrs)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(OutAddrAssignmentItem) *const SIXTRL_RESTRICT item );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(OutAddrAssignmentItem_num_slots)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(OutAddrAssignmentItem) *const SIXTRL_RESTRICT item,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_DATAPTR_DEC NS(OutAddrAssignmentItem)*
NS(OutAddrAssignmentItem_preset)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(OutAddrAssignmentItem)* SIXTRL_RESTRICT item );

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t)
NS(OutAddrAssignmentItem_elem_type_id)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(OutAddrAssignmentItem) *const SIXTRL_RESTRICT item );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(OutAddrAssignmentItem_elem_index)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(OutAddrAssignmentItem) *const SIXTRL_RESTRICT item );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(OutAddrAssignmentItem_dest_pointer_offset)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(OutAddrAssignmentItem) *const SIXTRL_RESTRICT item );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(OutAddrAssignmentItem_out_buffer_index)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(OutAddrAssignmentItem) *const SIXTRL_RESTRICT item );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN void NS(OutAddrAssignmentItem_set_elem_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)* SIXTRL_RESTRICT item,
    NS(object_type_id_t) const type_id );

SIXTRL_STATIC SIXTRL_FN void NS(OutAddrAssignmentItem_set_elem_index)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const elem_index );

SIXTRL_STATIC SIXTRL_FN void NS(OutAddrAssignmentItem_set_dest_pointer_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const dest_pointer_offset );

SIXTRL_STATIC SIXTRL_FN void NS(OutAddrAssignmentItem_set_out_buffer_index)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const out_buffer_index );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem const*
NS(OutAddrAssignmentItem_managed_buffer_get_const_item)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)*
NS(OutAddrAssignmentItem_managed_buffer_get_item)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem) const*
NS(OutAddrAssignmentItem_buffer_get_const_item)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index );

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC
NS(OutAddrAssignmentItem)* NS(OutAddrAssignmentItem_buffer_get_item)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index );

SIXTRL_STATIC SIXTRL_FN bool NS(OutAddrAssignmentItem_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)*
NS(OutAddrAssignmentItem_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)*
NS(OutAddrAssignmentItem_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(object_type_id_t) const type_id,
    NS(buffer_size_t) const elem_index,
    NS(buffer_size_t) const dest_pointer_offset,
    NS(buffer_size_t) const out_buffer_index );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)*
NS(OutAddrAssignmentItem_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const NS(OutAddrAssignmentItem) *const
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
    #if !defined( _GPUCODE )
    #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */


SIXTRL_INLINE NS(buffer_size_t)
NS(OutAddrAssignmentItem_num_dataptrs)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(OutAddrAssignmentItem) *const SIXTRL_RESTRICT item )
{
    ( void )item;
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(OutAddrAssignmentItem_num_slots)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(OutAddrAssignmentItem) *const SIXTRL_RESTRICT item,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t extent = NS(ManagedBuffer_get_slot_based_length)(
        sizeof( NS(OutAddrAssignmentItem) ), slot_size );

    ( void )item;

    SIXTRL_ASSERT( ( slot_size == ZERO ) || ( ( extent % slot_size ) == ZERO ) );
    return ( slot_size > ZERO ) ? ( extent / slot_size ) : ( ZERO );

}

SIXTRL_INLINE SIXTRL_DATAPTR_DEC NS(OutAddrAssignmentItem)*
NS(OutAddrAssignmentItem_preset)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(OutAddrAssignmentItem)* SIXTRL_RESTRICT item )
{
    if( item != SIXTRL_NULLPTR )
    {
        item->elem_type_id          = ::NS(OBJECT_TYPE_NONE);
        item->elem_index            = ( NS(buffer_size_t) )0u;
        item->dest_pointer_offset   = ( NS(buffer_size_t) )0u;
        item->out_buffer_index      = ( NS(buffer_size_t) )0u;
    }

    return item;
}

SIXTRL_INLINE NS(object_type_id_t) NS(OutAddrAssignmentItem_elem_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(OutAddrAssignmentItem)
        *const SIXTRL_RESTRICT item )
{
    return ( item != SIXTRL_NULLPTR )
        ? item->elem_type_id : NS(OBJECT_TYPE_NONE);
}

SIXTRL_INLINE NS(buffer_size_t)
NS(OutAddrAssignmentItem_elem_index)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(OutAddrAssignmentItem) *const SIXTRL_RESTRICT item )
{
    return ( item != SIXTRL_NULLPTR )
        ? item->elem_index : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(OutAddrAssignmentItem_dest_pointer_offset)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(OutAddrAssignmentItem) *const SIXTRL_RESTRICT item )
{
    return ( item != SIXTRL_NULLPTR )
        ? item->dest_pointer_offset : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(OutAddrAssignmentItem_out_buffer_index)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(OutAddrAssignmentItem) *const SIXTRL_RESTRICT item )
{
    return ( item != SIXTRL_NULLPTR )
        ? item->out_buffer_index : ( NS(buffer_size_t) )0u;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(OutAddrAssignmentItem_set_elem_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)* SIXTRL_RESTRICT item,
    NS(object_type_id_t) const type_id )
{
    if( item != SIXTRL_NULLPTR ) item->elem_type_id = type_id;
}

SIXTRL_INLINE void NS(OutAddrAssignmentItem_set_elem_index)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const elem_index )
{
    if( item != SIXTRL_NULLPTR ) item->elem_index = elem_index;
}

SIXTRL_INLINE void NS(OutAddrAssignmentItem_dest_pointer_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const dest_pointer_offset )
{
    if( item != SIXTRL_NULLPTR )
        item->dest_pointer_offset = dest_pointer_offset;
}

SIXTRL_INLINE void NS(OutAddrAssignmentItem_out_buffer_index)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)* SIXTRL_RESTRICT item,
    NS(buffer_size_t) const out_buffer_index )
{
    if( item != SIXTRL_NULLPTR ) item->out_buffer_index = out_buffer_index;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem) const*
NS(BufferIndex_get_const_out_addr_assignment_item)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const index_obj )
{
    typedef NS(OutAddrAssignmentItem) item_t;
    typedef SIXTRL_BUFFER_OBJ_DATAPTR_DEC item_t const* ptr_to_item_t;
    ptr_to_item_t ptr_to_item = SIXTRL_NULLPTR;

    if( ( index_obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( index_obj ) ==
          NS(OBJECT_TYPE_OUT_ADDR_ASSIGN_ITEM) ) &&
        ( NS(Object_get_size)( index_obj ) >= sizeof( item_t ) ) )
    {
        ptr_to_be = ( ptr_to_item_t )( uintptr_t
            )NS(Object_get_begin_addr)( index_obj );
    }

    return ptr_to_be;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)*
NS(BufferIndex_get_out_addr_assignment_item)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* index_obj )
{
    return ( SIXTRL_BUFFER_OBJ_DATAPTR_DEC NS(OutAddrAssignmentItem)*
        )NS(BufferIndex_get_const_out_addr_assignment_item)( index_obj );
}


SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem) const*
NS(OutAddrAssignmentItem_managed_buffer_get_const_item)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    return NS(BufferIndex_get_const_out_addr_assignment_item)(
        NS(ManagedBuffer_get_const_object)( pbuffer, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)*
NS(OutAddrAssignmentItem_managed_buffer_get_item)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size )
{
    return NS(BufferIndex_get_out_addr_assignment_item)(
        NS(ManagedBuffer_get_object)( pbuffer, index, slot_size ) );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem) const*
NS(OutAddrAssignmentItem_buffer_get_const_item)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* ptr_raw_t;
    return NS(OutAddrAssignmentItem_managed_buffer_get_const_item)(
        ( ptr_raw_t )( uintptr_t )NS(Buffer_get_data_begin_addr)( buffer ),
        index, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC
NS(OutAddrAssignmentItem)* NS(OutAddrAssignmentItem_buffer_get_item)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_raw_t;
    return NS(OutAddrAssignmentItem_managed_buffer_get_item)(
        ( ptr_raw_t )( uintptr_t )NS(Buffer_get_data_begin_addr)( buffer ),
        index, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE bool NS(OutAddrAssignmentItem_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    typedef NS(OutAddrAssignmentItem) item_t;
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_dataptrs =
        NS(OutAddrAssignmentItem_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts = SIXTRL_NULLPTR;
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return NS(Buffer_can_add_object)( buffer, sizeof( item_t ), num_dataptrs,
        sizes, counts, requ_objects, requ_slots, requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)*
NS(OutAddrAssignmentItem_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(OutAddrAssignmentItem) item_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC item_t* ptr_to_item_t;
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_dataptrs =
        NS(OutAddrAssignmentItem_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    item_t temp_obj;
    NS(OutAddrAssignmentItem_preset)( &temp_obj );

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return ( ptr_to_item_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( item_t ),
            NS(OBJECT_TYPE_DRIFT), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)*
NS(OutAddrAssignmentItem_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(object_type_id_t) const type_id,
    NS(buffer_size_t) const elem_index,
    NS(buffer_size_t) const dest_pointer_offset,
    NS(buffer_size_t) const out_buffer_index )
{
    typedef NS(OutAddrAssignmentItem) item_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC item_t* ptr_to_item_t;
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_dataptrs =
        NS(OutAddrAssignmentItem_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    item_t temp_obj;
    NS(OutAddrAssignmentItem_preset)( &temp_obj );
    NS(OutAddrAssignmentItem_set_elem_type_id)( &temp_obj, type_id );
    NS(OutAddrAssignmentItem_set_elem_index)( &temp_obj, elem_index );

    NS(OutAddrAssignmentItem_set_dest_pointer_offset)(
        &temp_obj, dest_pointer_offset );

    NS(OutAddrAssignmentItem_set_out_buffer)(
        &temp_obj, out_buffer_index );
    temp_obj.length = length;

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return ( ptr_to_item_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( item_t ),
            NS(OBJECT_TYPE_DRIFT), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(OutAddrAssignmentItem)*
NS(OutAddrAssignmentItem_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const NS(OutAddrAssignmentItem) *const
        SIXTRL_RESTRICT item )
{
    return NS(OutAddrAssignmentItem_add)( buffer,
        NS(OutAddrAssignmentItem_elem_type_id)( item ),
        NS(OutAddrAssignmentItem_elem_index)( item ),
        NS(OutAddrAssignmentItem_dest_pointer_offset)( item ),
        NS(OutAddrAssignmentItem_out_buffer_index)( item ) );
}

#endif /* !defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_OUTPUT_OUTPUT_ADDR_ASSIGNMENT_C99_H__ */
