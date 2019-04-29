#ifndef SIXTRACKLIB_COMMON_BUFFER_BUFFER_ARRAY_OBJECT_H__
#define SIXTRACKLIB_COMMON_BUFFER_BUFFER_ARRAY_OBJECT_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

typedef struct NS(BufferArrayObj)
{
    NS(buffer_addr_t)    begin_addr       SIXTRL_ALIGN( 8u );
    NS(buffer_addr_t)    offset_addr      SIXTRL_ALIGN( 8u );
    NS(buffer_size_t)    num_elements     SIXTRL_ALIGN( 8u );
    NS(buffer_size_t)    max_num_elements SIXTRL_ALIGN( 8u );
    NS(buffer_size_t)    capacity         SIXTRL_ALIGN( 8u );
    NS(buffer_size_t)    slot_size        SIXTRL_ALIGN( 8u );
    NS(object_type_id_t) base_type_id     SIXTRL_ALIGN( 8u );
}
NS(BufferArrayObj);

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)*
NS(BufferArrayObj_preset)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferArrayObj)* SIXTRL_RESTRICT array );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t) NS(BufferArrayObj_get_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array );

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t)
NS(BufferArrayObj_get_element_offset_list_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array );


SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC void const*
NS(BufferArrayObj_get_const_data_begin)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC void const*
NS(BufferArrayObj_get_const_data_end)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array );


SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC void*
NS(BufferArrayObj_get_data_begin)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC void*
NS(BufferArrayObj_get_data_end)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array );


SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_size_t) const*
NS(BufferArrayObj_get_element_offset_list_begin)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_size_t) const*
NS(BufferArrayObj_get_element_offset_list_end)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferArrayObj_get_element_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array,
    NS(buffer_size_t) const index );


SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferArrayObj_get_num_elements)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferArrayObj_get_max_num_elements)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(BufferArrayObj_get_length)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(BufferArrayObj_get_capacity)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(BufferArrayObj_get_slot_size)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array );

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(BufferArrayObj_get_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t)
NS(BufferArrayObj_get_element_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array,
    NS(buffer_size_t) const index );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferArrayObj_get_element_length)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array,
    NS(buffer_size_t) const index );

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t)
NS(BufferArrayObj_get_element_end_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array,
    NS(buffer_size_t) const index );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN void NS(BufferArrayObj_set_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_addr_t) const begin_addr );

SIXTRL_STATIC SIXTRL_FN void
NS(BufferArrayObj_set_element_offset_list_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_addr_t) const offset_list_begin_addr );

SIXTRL_STATIC SIXTRL_FN void NS(BufferArrayObj_set_num_elements)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const num_elements );

SIXTRL_STATIC SIXTRL_FN void NS(BufferArrayObj_set_max_num_elements)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const max_num_elements );

SIXTRL_STATIC SIXTRL_FN void NS(BufferArrayObj_set_length)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const length );

SIXTRL_STATIC SIXTRL_FN void NS(BufferArrayObj_set_capacity)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const capacity );

SIXTRL_STATIC SIXTRL_FN void NS(BufferArrayObj_set_slot_size)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN void NS(BufferArrayObj_set_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(object_type_id_t) const num_elements );

SIXTRL_STATIC SIXTRL_FN void NS(BufferArrayObj_set_element_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const index, NS(buffer_size_t) const offset );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN void NS(BufferArrayObj_clear)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array );

SIXTRL_STATIC SIXTRL_FN bool NS(BufferArrayObj_append_element)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    SIXTRL_ARGPTR_DEC const void *const SIXTRL_RESTRICT obj_handle_begin,
    NS(buffer_size_t) const obj_handle_size );

SIXTRL_STATIC SIXTRL_FN bool NS(BufferArrayObj_append_num_elements)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const obj_handle_size,
    NS(buffer_size_t) const num_elements );

SIXTRL_STATIC SIXTRL_FN bool NS(BufferArrayObj_remove_last_element)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array );

SIXTRL_STATIC SIXTRL_FN bool NS(BufferArrayObj_remove_last_num_elements)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const num_elements );

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)*
NS(BufferArrayObj_preset_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferArrayObj)* SIXTRL_RESTRICT array );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_addr_t)
NS(BufferArrayObj_get_begin_addr_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferArrayObj) *const SIXTRL_RESTRICT array );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_addr_t)
NS(BufferArrayObj_get_element_offset_list_begin_addr_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array );


SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC void const*
NS(BufferArrayObj_get_const_data_begin_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferArrayObj) *const SIXTRL_RESTRICT array );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC void const*
NS(BufferArrayObj_get_const_data_end_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferArrayObj) *const SIXTRL_RESTRICT array );


SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC void*
NS(BufferArrayObj_get_data_begin_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC void*
NS(BufferArrayObj_get_data_end_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferArrayObj)* SIXTRL_RESTRICT array );


SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_size_t) const*
NS(BufferArrayObj_get_element_offset_list_begin_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_size_t) const*
NS(BufferArrayObj_get_element_offset_list_end_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferArrayObj_get_element_offset_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferArrayObj) *const SIXTRL_RESTRICT array,
    NS(buffer_size_t) const index );


SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferArrayObj_get_num_elements_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferArrayObj) *const SIXTRL_RESTRICT array );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferArrayObj_get_max_num_elements_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferArrayObj) *const SIXTRL_RESTRICT array );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferArrayObj_get_length_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferArrayObj) *const SIXTRL_RESTRICT array );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferArrayObj_get_capacity_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferArrayObj) *const SIXTRL_RESTRICT array );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferArrayObj_get_slot_size_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferArrayObj) *const SIXTRL_RESTRICT array );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
NS(BufferArrayObj_get_type_id_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferArrayObj) *const SIXTRL_RESTRICT array );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_addr_t)
NS(BufferArrayObj_get_element_begin_addr_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferArrayObj) *const SIXTRL_RESTRICT array,
    NS(buffer_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferArrayObj_get_element_length_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferArrayObj) *const SIXTRL_RESTRICT array,
    NS(buffer_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_addr_t)
NS(BufferArrayObj_get_element_end_addr_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferArrayObj) *const SIXTRL_RESTRICT array,
    NS(buffer_size_t) const index );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(BufferArrayObj_clear_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(BufferArrayObj_append_element_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    SIXTRL_ARGPTR_DEC const void *const SIXTRL_RESTRICT obj_handle_begin,
    NS(buffer_size_t) const obj_handle_size );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(BufferArrayObj_append_num_elements_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const obj_handle_size,
    NS(buffer_size_t) const num_elements );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(BufferArrayObj_remove_last_element_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(BufferArrayObj_remove_last_num_elements_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const num_elements );

#endif /* !defiend( _GPUcODE ) */

/* ------------------------------------------------------------------------ */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferArrayObj_get_required_num_slots_on_managed_buffer)(
    NS(buffer_size_t) const max_num_elements,
    NS(buffer_size_t) const capacity,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferArrayObj_get_required_num_dataptrs_on_managed_buffer)(
    NS(buffer_size_t) const max_num_elements,
    NS(buffer_size_t) const capacity,
    NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferArrayObj_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferArrayObj_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity );

SIXTRL_STATIC SIXTRL_FN bool NS(BufferArrayObj_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptr );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)*
NS(BufferArrayObj_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity,
    NS(object_type_id_t) const base_type_id );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)*
NS(BufferArrayObj_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_addr_t) const data_begin_addr,
    NS(buffer_addr_t) const element_offset_list_begin_addr,
    NS(buffer_size_t) const nelements, NS(buffer_size_t) const max_nelements,
    NS(buffer_size_t) const capacity, NS(buffer_size_t) const slot_size,
    NS(object_type_id_t) const base_type_id );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)*
NS(BufferArrayObj_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(BufferArrayObj) *const
        SIXTRL_RESTRICT other );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferArrayObj_get_required_num_slots_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferArrayObj_get_required_num_dataptrs_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(BufferArrayObj_can_be_added_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptr );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)*
NS(BufferArrayObj_new_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity,
    NS(object_type_id_t) const base_type_id );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)*
NS(BufferArrayObj_add_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_addr_t) const data_begin_addr,
    NS(buffer_addr_t) const element_offset_list_begin_addr,
    NS(buffer_size_t) const nelements, NS(buffer_size_t) const max_nelements,
    NS(buffer_size_t) const capacity, NS(buffer_size_t) const slot_size,
    NS(object_type_id_t) const base_type_id );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)*
NS(BufferArrayObj_add_copy_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(BufferArrayObj) *const
        SIXTRL_RESTRICT other );

#endif /* !defined( _GPUCODE ) */

/* ************************************************************************* */
/* *********         Implementation of Inline Functions          *********** */
/* ************************************************************************* */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)*
NS(BufferArrayObj_preset)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferArrayObj)* SIXTRL_RESTRICT array )
{
    if( array != SIXTRL_NULLPTR )
    {
        array->begin_addr       = ( NS(buffer_addr_t) )0u;
        array->offset_addr      = ( NS(buffer_addr_t) )0u;

        array->num_elements     = ( NS(buffer_size_t) )0u;
        array->max_num_elements = ( NS(buffer_size_t) )0u;
        array->capacity         = ( NS(buffer_size_t) )0u;
        array->slot_size        = NS(BUFFER_DEFAULT_SLOT_SIZE);

        array->base_type_id     =
            ( NS(object_type_id_t) )SIXTRL_OBJECT_TYPE_UNDEFINED;
    }

    return array;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_addr_t) NS(BufferArrayObj_get_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array )
{
    return ( array != SIXTRL_NULLPTR )
        ? array->begin_addr : ( NS(buffer_addr_t ) )0u;
}

SIXTRL_INLINE NS(buffer_addr_t)
NS(BufferArrayObj_get_element_offset_list_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj) *const
        SIXTRL_RESTRICT array )
{
    return ( array != SIXTRL_NULLPTR )
        ? array->offset_addr : ( NS(buffer_addr_t ) )0u;
}


SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC void const*
NS(BufferArrayObj_get_const_data_begin)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC void const* ptr_t;
    return ( ptr_t )( uintptr_t )NS(BufferArrayObj_get_begin_addr)( array );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC void const*
NS(BufferArrayObj_get_const_data_end)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC void const* ptr_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* ptr_raw_t;

    ptr_t end_ptr = NS(BufferArrayObj_get_const_data_begin)( array );

    if( end_ptr != SIXTRL_NULLPTR )
    {
        end_ptr = ( ptr_t )( ( ( ptr_raw_t )end_ptr ) +
            NS(BufferArrayObj_get_length)( array ) );
    }

    return end_ptr;
}


SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC void*
NS(BufferArrayObj_get_data_begin)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array )
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC void* )( uintptr_t
        )NS(BufferArrayObj_get_const_data_begin)( array );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC void*
NS(BufferArrayObj_get_data_end)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array )
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC void* )( uintptr_t
        )NS(BufferArrayObj_get_const_data_end)( array );
}


SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_size_t) const*
NS(BufferArrayObj_get_element_offset_list_begin)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_size_t) const* ptr_offset_t;
    SIXTRL_ASSERT( array != SIXTRL_NULLPTR );

    return ( ptr_offset_t )( uintptr_t )array->offset_addr;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_size_t) const*
NS(BufferArrayObj_get_element_offset_list_end)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_size_t) const* ptr_offset_t;

    ptr_offset_t end_ptr =
        NS(BufferArrayObj_get_element_offset_list_begin)( array );

    if( end_ptr != SIXTRL_NULLPTR )
    {
        end_ptr = end_ptr + NS(BufferArrayObj_get_num_elements)( array );
    }

    return end_ptr;
}

SIXTRL_INLINE NS(buffer_size_t) NS(BufferArrayObj_get_element_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array,
    NS(buffer_size_t) const index )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_size_t) const* ptr_offset_t;

    ptr_offset_t offset_list =
        NS(BufferArrayObj_get_element_offset_list_begin)( array );

    return ( ( offset_list != SIXTRL_NULLPTR ) &&
             ( index <= NS(BufferArrayObj_get_num_elements)( array ) ) )
        ? offset_list[ index ] : ( NS(buffer_size_t) )0u;
}


SIXTRL_INLINE NS(buffer_size_t) NS(BufferArrayObj_get_num_elements)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array )
{
    return ( array != SIXTRL_NULLPTR )
        ? array->num_elements : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(BufferArrayObj_get_max_num_elements)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array )
{
    return ( array != SIXTRL_NULLPTR )
        ? array->max_num_elements : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(BufferArrayObj_get_length)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_get_element_offset)(
        array, NS(BufferArrayObj_get_num_elements)( array ) );
}

SIXTRL_INLINE NS(buffer_size_t) NS(BufferArrayObj_get_capacity)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array )
{
    return ( array != SIXTRL_NULLPTR )
        ? array->capacity : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(BufferArrayObj_get_slot_size)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array )
{
    return ( array != SIXTRL_NULLPTR )
        ? array->slot_size : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(object_type_id_t) NS(BufferArrayObj_get_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array )
{
    return ( array != SIXTRL_NULLPTR ) ? array->base_type_id
        : ( NS(object_type_id_t) )SIXTRL_OBJECT_TYPE_UNDEFINED;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_addr_t) NS(BufferArrayObj_get_element_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array,
    NS(buffer_size_t) const index )
{
    typedef NS(buffer_addr_t) addr_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_size_t) const* ptr_offset_t;

    addr_t const base_addr = NS(BufferArrayObj_get_begin_addr)( array );

    ptr_offset_t offset_list =
        NS(BufferArrayObj_get_element_offset_list_begin)( array );

    SIXTRL_ASSERT( base_addr > ( NS(buffer_addr_t) )0u );
    SIXTRL_ASSERT( offset_list != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( NS(BufferArrayObj_get_slot_size)( array ) >
                   ( NS(buffer_size_t) )0u );

    SIXTRL_ASSERT( index < NS(BufferArrayObj_get_num_elements)( array ) );

    SIXTRL_ASSERT( NS(BufferArrayObj_get_num_elements)( array ) <=
                   NS(BufferArrayObj_get_max_num_elements)( array ) );

    SIXTRL_ASSERT( NS(BufferArrayObj_get_length)( array ) <=
                   NS(BufferArrayObj_get_capacity)( array ) );

    SIXTRL_ASSERT( offset_list[ index ] <
                   NS(BufferArrayObj_get_length)( array ) );

    SIXTRL_ASSERT( ( ( NS(buffer_size_t) )0u ) == ( offset_list[ index ] %
        NS(BufferArrayObj_get_slot_size)( array ) ) );

    return base_addr + offset_list[ index ];
}

SIXTRL_INLINE NS(buffer_size_t) NS(BufferArrayObj_get_element_length)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array,
    NS(buffer_size_t) const index )
{
    typedef NS(buffer_size_t) size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_size_t) const* ptr_offset_t;

    SIXTRL_STATIC_VAR size_t const ZERO = ( size_t )0u;

    ptr_offset_t offset_list =
        NS(BufferArrayObj_get_element_offset_list_begin)( array );

    size_t const end_index = index + ( size_t )1u;
    size_t length = ZERO;

    SIXTRL_ASSERT( NS(BufferArrayObj_get_slot_size)( array ) > ZERO );

    SIXTRL_ASSERT( offset_list != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(BufferArrayObj_get_num_elements)( array ) >=
                   end_index );

    SIXTRL_ASSERT( NS(BufferArrayObj_get_num_elements)( array ) <=
                   NS(BufferArrayObj_get_max_num_elements)( array ) );

    SIXTRL_ASSERT( NS(BufferArrayObj_get_length)( array ) <=
                   NS(BufferArrayObj_get_capacity)( array ) );

    SIXTRL_ASSERT( offset_list[ end_index ] <=
                   NS(BufferArrayObj_get_length)( array ) );

    length = ( offset_list[ end_index ] >= offset_list[ index ] )
        ? ( offset_list[ end_index ] - offset_list[ index ] ) : ZERO;

    SIXTRL_ASSERT( ZERO == ( length % NS(BufferArrayObj_get_slot_size)(
        array ) ) );

    return length;
}

SIXTRL_INLINE NS(buffer_addr_t) NS(BufferArrayObj_get_element_end_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array,
    NS(buffer_size_t) const index )
{
    typedef NS(buffer_addr_t) addr_t;

    addr_t element_end_addr =
        NS(BufferArrayObj_get_element_begin_addr)( array, index );

    if( element_end_addr > ( addr_t )0u )
    {
        element_end_addr += NS(BufferArrayObj_get_element_length)(
            array, index );
    }

    return element_end_addr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(BufferArrayObj_set_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_addr_t) const begin_addr )
{
    SIXTRL_ASSERT( array != SIXTRL_NULLPTR );
    array->begin_addr = begin_addr;
}

SIXTRL_INLINE void NS(BufferArrayObj_set_element_offset_list_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_addr_t) const offset_list_begin_addr )
{
    SIXTRL_ASSERT( array != SIXTRL_NULLPTR );
    array->offset_addr = offset_list_begin_addr;
}

SIXTRL_INLINE void NS(BufferArrayObj_set_num_elements)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const num_elements )
{
    SIXTRL_ASSERT( array != SIXTRL_NULLPTR );
    array->num_elements = num_elements;
}

SIXTRL_INLINE void NS(BufferArrayObj_set_max_num_elements)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const max_num_elements )
{
    SIXTRL_ASSERT( array != SIXTRL_NULLPTR );
    array->max_num_elements = max_num_elements;
}

SIXTRL_INLINE void NS(BufferArrayObj_set_length)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const length )
{
    SIXTRL_ASSERT( array != SIXTRL_NULLPTR );
    NS(BufferArrayObj_set_element_offset)(
        array, NS(BufferArrayObj_get_num_elements)( array ), length );
}

SIXTRL_INLINE void NS(BufferArrayObj_set_capacity)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const capacity )
{
    SIXTRL_ASSERT( array != SIXTRL_NULLPTR );
    array->capacity = capacity;
}

SIXTRL_INLINE void NS(BufferArrayObj_set_slot_size)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_ASSERT( array != SIXTRL_NULLPTR );
    array->slot_size = slot_size;
}

SIXTRL_INLINE void NS(BufferArrayObj_set_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(object_type_id_t) const base_type_id )
{
    SIXTRL_ASSERT( array != SIXTRL_NULLPTR );
    array->base_type_id = base_type_id;
}

SIXTRL_INLINE void NS(BufferArrayObj_set_element_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const index, NS(buffer_size_t) const offset )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_size_t)* ptr_offset_t;

    ptr_offset_t offset_list = ( ptr_offset_t )( uintptr_t
        )NS(BufferArrayObj_get_element_offset_list_begin_addr)( array );

    SIXTRL_ASSERT( offset_list != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( index <= NS(BufferArrayObj_get_num_elements)( array ) );
    SIXTRL_ASSERT( NS(BufferArrayObj_get_max_num_elements)( array ) >=
                   NS(BufferArrayObj_get_num_elements)( array ) );

    offset_list[ index ] = offset;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(BufferArrayObj_clear)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array )
{
    typedef NS(buffer_size_t) size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_t;

    SIXTRL_STATIC_VAR size_t const ZERO = ( size_t )0u;
    size_t const nelem = NS(BufferArrayObj_get_num_elements)( array );

    if( nelem > ZERO )
    {
        ptr_t begin = ( ptr_t )NS(BufferArrayObj_get_data_begin)( array );
        size_t const bin_length = NS(BufferArrayObj_get_length)( array );

        size_t ii = ZERO;

        for( ; ii <= nelem ; ++ii )
        {
            NS(BufferArrayObj_set_element_offset)( array, ii, ZERO );
        }

        if( begin != SIXTRL_NULLPTR )
        {
            SIXTRL_STATIC_VAR unsigned char const CZERO = ( unsigned char )0u;
            SIXTRACKLIB_SET_VALUES( unsigned char, begin, bin_length, CZERO );
        }

        NS(BufferArrayObj_set_num_elements)( array, ZERO );
    }

    return;
}

SIXTRL_INLINE bool NS(BufferArrayObj_append_element)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    SIXTRL_ARGPTR_DEC const void *const SIXTRL_RESTRICT obj_handle_begin,
    NS(buffer_size_t) const obj_handle_size )
{
    typedef NS(buffer_size_t) size_t;

    SIXTRL_STATIC_VAR size_t const ZERO = ( size_t )0u;

    bool success = false;
    size_t const slot_size = NS(BufferArrayObj_get_slot_size)( array );

    if( ( array != SIXTRL_NULLPTR ) && ( obj_handle_size > ZERO ) &&
        ( slot_size > ZERO ) )
    {
        size_t const nelem = NS(BufferArrayObj_get_num_elements)( array );
        size_t const length = NS(BufferArrayObj_get_length)( array );
        size_t const capacity = NS(BufferArrayObj_get_capacity)( array );

        size_t const elem_length = NS(ManagedBuffer_get_slot_based_length)(
            obj_handle_size, slot_size );

        size_t const new_length = elem_length + length;

        SIXTRL_ASSERT( ( length % slot_size ) == ZERO );

        if( ( ( elem_length % slot_size ) == ZERO ) &&
            ( new_length <= capacity ) &&
            ( nelem < NS(BufferArrayObj_get_max_num_elements)( array ) ) )
        {
            if( obj_handle_begin != SIXTRL_NULLPTR )
            {
                SIXTRL_ASSERT( NS(BufferArrayObj_get_data_begin)( array ) !=
                               SIXTRL_NULLPTR );

                SIXTRACKLIB_COPY_VALUES( unsigned char,
                    NS(BufferArrayObj_get_data_begin)( array ),
                    obj_handle_begin, obj_handle_size );
            }

            NS(BufferArrayObj_set_element_offset)(
                array, nelem, new_length );

            NS(BufferArrayObj_set_num_elements)(
                array, nelem + ( size_t )1u );

            success = true;
        }
    }

    return success;
}

SIXTRL_INLINE bool NS(BufferArrayObj_append_num_elements)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const obj_handle_size,
    NS(buffer_size_t) const num_elements )
{
    bool success = false;

    typedef NS(buffer_size_t) buf_size_t;
    typedef unsigned char raw_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t const current_nelems =
        NS(BufferArrayObj_get_num_elements)( array );

    buf_size_t const max_nelems =
        NS(BufferArrayObj_get_max_num_elements)( array );

    if( ( array != SIXTRL_NULLPTR ) &&
        ( ( current_nelems + num_elements ) <= max_nelems ) )
    {
        typedef SIXTRL_BUFFER_DATAPTR_DEC raw_t* ptr_data_t;

        buf_size_t const current_length =
            NS(BufferArrayObj_get_length)( array );

        ptr_data_t current_end_ptr =
            NS(BufferArrayObj_get_data_end)( array );

        buf_size_t ii = ZERO;
        success = true;

        for( ; ii < num_elements ; ++ii )
        {
            success &= NS(BufferArrayObj_append_element)(
                array, SIXTRL_NULLPTR, obj_handle_size );

            if( !success ) break;
        }

        if( !success )
        {
            buf_size_t const new_length =
                NS(BufferArrayObj_get_length)( array );

            buf_size_t const new_nelems =
                NS(BufferArrayObj_get_num_elements)( array );

            if( current_nelems < new_nelems )
            {
                buf_size_t ii = current_nelems + ( buf_size_t )1u;

                while( ii <= new_nelems )
                {
                    NS(BufferArrayObj_set_element_offset)(
                        array, ii++, ZERO );
                }
            }

            if( current_length < new_length )
            {
                SIXTRL_ASSERT( current_end_ptr != SIXTRL_NULLPTR );
                SIXTRL_STATIC_VAR raw_t const CZERO = ( raw_t )0u;

                SIXTRACKLIB_SET_VALUES( raw_t, current_end_ptr,
                        ( new_length - current_length ), CZERO );
            }

            if( current_nelems < new_nelems )
            {
                NS(BufferArrayObj_set_num_elements)(
                    array, current_nelems );

                NS(BufferArrayObj_set_element_offset)(
                    array, current_nelems, current_length );
            }
        }
    }

    return success;
}

SIXTRL_INLINE bool NS(BufferArrayObj_remove_last_element)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array )
{
    typedef NS(buffer_size_t) size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_data_t;

    bool success = false;

    SIXTRL_STATIC_VAR size_t const ZERO = ( size_t )0u;

    size_t const current_nelems =
        NS(BufferArrayObj_get_num_elements)( array );

    if( ( array != SIXTRL_NULLPTR ) && ( current_nelems > ZERO ) )
    {
        size_t const last_index = current_nelems - ( size_t )1u;

        ptr_data_t elem_begin = ( ptr_data_t )( uintptr_t
            )NS(BufferArrayObj_get_element_begin_addr)( array, last_index );

        size_t const elem_length =
            NS(BufferArrayObj_get_element_length)( array, last_index );

        if( ( elem_length > ZERO ) && ( elem_begin != SIXTRL_NULLPTR ) )
        {
            SIXTRL_STATIC_VAR unsigned char CZERO = ( unsigned char )0u;

            SIXTRACKLIB_SET_VALUES(
                unsigned char, elem_begin, elem_length, CZERO );
        }

        NS(BufferArrayObj_set_element_offset)(
            array, current_nelems, ZERO );

        NS(BufferArrayObj_set_num_elements)(
            array, current_nelems - ( size_t )1u );

        success = true;
    }

    return success;
}

SIXTRL_INLINE bool NS(BufferArrayObj_remove_last_num_elements)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const num_elements_to_remove )
{
    typedef NS(buffer_size_t) size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_data_t;

    bool success = false;

    SIXTRL_STATIC_VAR size_t const ZERO = ( size_t )0u;

    size_t const current_nelems =
        NS(BufferArrayObj_get_num_elements)( array );

    if( ( array != SIXTRL_NULLPTR ) && ( num_elements_to_remove > ZERO ) &&
        ( current_nelems >= num_elements_to_remove ) )
    {
        size_t const new_nelems = current_nelems - num_elements_to_remove;

        if( new_nelems > ZERO )
        {
            size_t ii = new_nelems + ( size_t )1u;

            size_t const current_length =
                NS(BufferArrayObj_get_length)( array );

            size_t const new_length = NS(BufferArrayObj_get_element_offset)(
                array, new_nelems - ( size_t )1u );

            if( current_length > new_length )
            {
                SIXTRL_STATIC_VAR unsigned char CZERO = ( unsigned char )0u;

                ptr_data_t removed_elem_begin = ( ptr_data_t )( uintptr_t
                    )NS(BufferArrayObj_get_element_begin_addr)(
                        array, new_nelems );

                SIXTRACKLIB_SET_VALUES( unsigned char, removed_elem_begin,
                                        current_length - new_length, CZERO );
            }

            for( ; ii <= current_nelems ; ++ii )
            {
                NS(BufferArrayObj_set_element_offset)( array, ii, ZERO );
            }
        }
        else
        {
            NS(BufferArrayObj_clear)( array );
        }

        success = true;
    }

    return success;
}

/* ------------------------------------------------------------------------- */


SIXTRL_INLINE NS(buffer_size_t)
NS(BufferArrayObj_get_required_num_slots_on_managed_buffer)(
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    buf_size_t num_slots = ( buf_size_t )0u;

    if( slot_size > ( buf_size_t )0u )
    {
        buf_size_t const offset_list_length =
            max_nelements + ( buf_size_t )1u;

        buf_size_t required_size = NS(ManagedBuffer_get_slot_based_length)(
            sizeof( NS(BufferArrayObj) ), slot_size );

        if( capacity > ( buf_size_t )0u )
        {
            required_size += NS(ManagedBuffer_get_slot_based_length)(
                capacity, slot_size );
        }

        required_size += NS(ManagedBuffer_get_slot_based_length)(
            sizeof( buf_size_t ) * offset_list_length, slot_size );

        SIXTRL_ASSERT( ( required_size % slot_size ) == ( buf_size_t )0u );
        num_slots = required_size / slot_size;
    }

    return num_slots;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(BufferArrayObj_get_required_num_dataptrs_on_managed_buffer)(
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity,
    NS(buffer_size_t) const slot_size )
{
    ( void )capacity;
    ( void )max_nelements;
    ( void )slot_size;

    return ( NS(buffer_size_t) )2u;
}

#if !defined( _GPUCODE )

SIXTRL_INLINE NS(buffer_size_t) NS(BufferArrayObj_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity )
{
    return NS(BufferArrayObj_get_required_num_slots_on_managed_buffer)(
        max_nelements, capacity, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE NS(buffer_size_t)
NS(BufferArrayObj_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity )
{
    return NS(BufferArrayObj_get_required_num_dataptrs_on_managed_buffer)(
        max_nelements, capacity, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE bool NS(BufferArrayObj_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_ptrs )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const nptrs = NS(BufferArrayObj_get_required_num_dataptrs)(
        buffer, max_nelements, capacity );

    buf_size_t sizes[]      = { ( buf_size_t )1u, sizeof( buf_size_t ) };
    buf_size_t counts[]     = { capacity, max_nelements + ( buf_size_t )1u };

    return NS(Buffer_can_add_object)( buffer, sizeof( NS(BufferArrayObj) ),
        nptrs, &sizes[ 0 ], &counts[ 0 ], requ_objects, requ_slots, requ_ptrs );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)*
NS(BufferArrayObj_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity,
    NS(object_type_id_t) const base_type_id )
{
    typedef NS(BufferArrayObj) array_t;
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC array_t* ptr_array_t;

    buf_size_t const nptrs = NS(BufferArrayObj_get_required_num_dataptrs)(
        buffer, max_nelements, capacity );

    buf_size_t offsets[] =
    {
        offsetof( array_t, begin_addr ),
        offsetof( array_t, offset_addr )
    };

    buf_size_t sizes[]     = { ( buf_size_t )1u, sizeof( buf_size_t ) };
    buf_size_t counts[]    = { capacity, max_nelements + ( buf_size_t )1u };

    array_t array;
    array.begin_addr       = ( NS(buffer_addr_t) )0u;
    array.offset_addr      = ( NS(buffer_addr_t) )0u;
    array.num_elements     = ( buf_size_t )0u;
    array.capacity         = capacity;
    array.base_type_id     = NS(OBJECT_TYPE_ARRAY);
    array.max_num_elements = max_nelements;
    array.slot_size        = NS(BUFFER_DEFAULT_SLOT_SIZE);
    array.base_type_id     = base_type_id;

    return ( ptr_array_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &array, sizeof( array_t ),
            NS(OBJECT_TYPE_ARRAY), nptrs,
                &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)*
NS(BufferArrayObj_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_addr_t) const data_begin_addr,
    NS(buffer_addr_t) const element_offset_list_begin_addr,
    NS(buffer_size_t) const num_elements, NS(buffer_size_t) const max_nelem,
    NS(buffer_size_t) const capacity, NS(buffer_size_t) const slot_size,
    NS(object_type_id_t) const base_type_id )
{
    typedef NS(BufferArrayObj) array_t;
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) addr_t;

    typedef SIXTRL_BUFFER_DATAPTR_DEC array_t*    ptr_array_t;
    typedef SIXTRL_ARGPTR_DEC buf_size_t const*   ptr_const_offset_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC buf_size_t* ptr_offset_t;
    typedef unsigned char raw_t;
    typedef SIXTRL_ARGPTR_DEC raw_t const* ptr_const_raw_t;
    typedef SIXTRL_ARGPTR_DEC raw_t*       ptr_raw_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR addr_t    const ZADDR = ( addr_t )0u;

    ptr_array_t a = NS(BufferArrayObj_new)( buffer, max_nelem, capacity );

    if( a != SIXTRL_NULLPTR )
    {
        NS(BufferArrayObj_set_max_num_elements)( a, max_nelem );
        NS(BufferArrayObj_set_capacity)( a, capacity );
        NS(BufferArrayObj_set_slot_size)( a, slot_size );
        NS(BufferArrayObj_set_type_id)( a, base_type_id );

        if( ( element_offset_list_begin_addr > ZADDR ) &&
            ( num_elements <= max_nelem ) && ( num_elements > ZERO ) )
        {
            ptr_offset_t out = ( ptr_offset_t )( uintptr_t
                )NS(BufferArrayObj_get_element_offset_list_begin_addr)( a );

            ptr_const_offset_t in = ( ptr_const_offset_t )( uintptr_t
                )element_offset_list_begin_addr;

            buf_size_t const nlist_elem = num_elements + ( buf_size_t )1u;
            buf_size_t const in_length = in[ num_elements ];

            SIXTRACKLIB_COPY_VALUES( buf_size_t, out, in, nlist_elem );

            if( ( data_begin_addr > ZADDR ) && ( capacity >= in_length ) )
            {
                ptr_const_raw_t raw_in =
                    ( ptr_const_raw_t )( uintptr_t )data_begin_addr;

                ptr_raw_t raw_out = ( ptr_raw_t )( uintptr_t
                    )NS(BufferArrayObj_get_begin_addr)( a );

                SIXTRACKLIB_COPY_VALUES( raw_t, raw_out, raw_in, in_length );

                if( in_length < capacity )
                {
                    SIXTRL_STATIC_VAR raw_t const CZERO = ( raw_t )0u;

                    SIXTRACKLIB_SET_VALUES( raw_t, raw_out + in_length,
                        capacity - in_length, CZERO );
                }
            }

            NS(BufferArrayObj_set_num_elements)( a, num_elements );
        }
        else if( ( data_begin_addr > ZADDR ) && ( capacity > ZERO ) )
        {
            ptr_const_raw_t raw_in =
                    ( ptr_const_raw_t )( uintptr_t )data_begin_addr;

            ptr_raw_t raw_out = ( ptr_raw_t )( uintptr_t
                    )NS(BufferArrayObj_get_begin_addr)( a );

            SIXTRACKLIB_COPY_VALUES( raw_t, raw_out, raw_in, capacity );
        }
    }

    return a;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)*
NS(BufferArrayObj_add_copy)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
    SIXTRL_RESTRICT buffer, SIXTRL_BUFFER_ARGPTR_DEC const
        NS(BufferArrayObj) *const SIXTRL_RESTRICT other )
{
    return NS(BufferArrayObj_add)( buffer,
        NS(BufferArrayObj_get_begin_addr)( other ),
        NS(BufferArrayObj_get_element_offset_list_begin_addr)( other ),
        NS(BufferArrayObj_get_num_elements)( other ),
        NS(BufferArrayObj_get_max_num_elements)( other ),
        NS(BufferArrayObj_get_capacity)( other ),
        NS(BufferArrayObj_get_slot_size)( other ),
        NS(BufferArrayObj_get_type_id)( other ) );
}

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_BUFFER_BUFFER_ARRAY_OBJECT_H__ */

/* end: sixtracklib/common/buffer/managed_buffer_handle.h */
