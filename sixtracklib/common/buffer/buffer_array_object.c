#include "sixtracklib/common/buffer/buffer_array_object.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer/buffer_type.h"
#include "sixtracklib/common/buffer.h"


SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)*
NS(BufferArrayObj_preset_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferArrayObj)* SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_preset)( array );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(buffer_addr_t) NS(BufferArrayObj_get_begin_addr_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj) *const SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_get_begin_addr)( array );
}

NS(buffer_addr_t) NS(BufferArrayObj_get_element_offset_list_begin_addr_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_get_element_offset_list_begin_addr)( array );
}


SIXTRL_BUFFER_DATAPTR_DEC void const*
NS(BufferArrayObj_get_const_data_begin_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferArrayObj) *const SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_get_const_data_begin)( array );
}

SIXTRL_BUFFER_DATAPTR_DEC void const*
NS(BufferArrayObj_get_const_data_end_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferArrayObj) *const SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_get_const_data_end)( array );
}


SIXTRL_BUFFER_DATAPTR_DEC void* NS(BufferArrayObj_get_data_begin_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_get_data_begin_ext)( array );
}

SIXTRL_BUFFER_DATAPTR_DEC void* NS(BufferArrayObj_get_data_end_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_get_data_end)( array );
}


SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_size_t) const*
NS(BufferArrayObj_get_element_offset_list_begin_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_get_element_offset_list_begin_ext)( array );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(buffer_size_t) const*
NS(BufferArrayObj_get_element_offset_list_end_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_get_element_offset_list_end)( array );
}

NS(buffer_size_t) NS(BufferArrayObj_get_element_offset_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj) *const
        SIXTRL_RESTRICT array, NS(buffer_size_t) const index )
{
    return NS(BufferArrayObj_get_element_offset_ext)( array, index );
}


NS(buffer_size_t) NS(BufferArrayObj_get_num_elements_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj) *const
        SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_get_num_elements)( array );
}

NS(buffer_size_t) NS(BufferArrayObj_get_max_num_elements_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_get_max_num_elements)( array );
}

NS(buffer_size_t) NS(BufferArrayObj_get_length_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj)
        *const SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_get_length)( array );
}

NS(buffer_size_t) NS(BufferArrayObj_get_capacity_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj) *const
        SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_get_capacity)( array );
}

NS(buffer_size_t) NS(BufferArrayObj_get_slot_size_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj) *const
        SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_get_slot_size)( array );
}

NS(object_type_id_t) NS(BufferArrayObj_get_type_id_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj) *const
        SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_get_type_id)( array );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(buffer_addr_t) NS(BufferArrayObj_get_element_begin_addr_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj) *const
        SIXTRL_RESTRICT array, NS(buffer_size_t) const index )
{
    return NS(BufferArrayObj_get_element_begin_addr)( array, index );
}

NS(buffer_size_t) NS(BufferArrayObj_get_element_length_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj) *const
        SIXTRL_RESTRICT array, NS(buffer_size_t) const index )
{
    return NS(BufferArrayObj_get_element_length)( array, index );
}

NS(buffer_addr_t) NS(BufferArrayObj_get_element_end_addr_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferArrayObj) *const
        SIXTRL_RESTRICT array, NS(buffer_size_t) const index )
{
    return NS(BufferArrayObj_get_element_end_addr)( array, index );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void NS(BufferArrayObj_clear_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array )
{
    NS(BufferArrayObj_clear)( array );
}


bool NS(BufferArrayObj_append_element_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    SIXTRL_ARGPTR_DEC const void *const SIXTRL_RESTRICT obj_handle_begin,
    NS(buffer_size_t) const obj_handle_size )
{
    return NS(BufferArrayObj_append_element)(
        array, obj_handle_begin, obj_handle_size );
}


bool NS(BufferArrayObj_append_num_elements_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const obj_handle_size,
    NS(buffer_size_t) const num_elements )
{
    return NS(BufferArrayObj_append_num_elements)(
        array, obj_handle_size, num_elements );
}

bool NS(BufferArrayObj_remove_last_element_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array )
{
    return NS(BufferArrayObj_remove_last_element)( array );
}

bool NS(BufferArrayObj_remove_last_num_elements_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* SIXTRL_RESTRICT array,
    NS(buffer_size_t) const num_elements )
{
    return NS(BufferArrayObj_remove_last_num_elements)(
        array, num_elements );
}

/* ------------------------------------------------------------------------- */


NS(buffer_size_t) NS(BufferArrayObj_get_required_num_slots_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity )
{
    return NS(BufferArrayObj_get_required_num_slots)(
        buffer, max_nelements, capacity );
}

NS(buffer_size_t) NS(BufferArrayObj_get_required_num_dataptrs_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity )
{
    return NS(BufferArrayObj_get_required_num_dataptrs)(
        buffer, max_nelements, capacity );
}

bool NS(BufferArrayObj_can_be_added_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs)
{
    return NS(BufferArrayObj_can_be_added)( buffer, max_nelements, capacity,
        requ_objects, requ_slots, requ_dataptrs );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* NS(BufferArrayObj_new_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_nelements, NS(buffer_size_t) const capacity,
    NS(object_type_id_t) const base_type_id )
{
    return NS(BufferArrayObj_new)(
        buffer, max_nelements, capacity, base_type_id );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)* NS(BufferArrayObj_add_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_addr_t) const data_begin_addr,
    NS(buffer_addr_t) const element_offset_list_begin_addr,
    NS(buffer_size_t) const nelements, NS(buffer_size_t) const max_nelements,
    NS(buffer_size_t) const capacity, NS(buffer_size_t) const slot_size,
    NS(object_type_id_t) const base_type_id )
{
    return NS(BufferArrayObj_add)( buffer, data_begin_addr,
        element_offset_list_begin_addr, nelements, max_nelements, capacity,
            slot_size, base_type_id );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferArrayObj)*
NS(BufferArrayObj_add_copy_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(BufferArrayObj) *const
        SIXTRL_RESTRICT other )
{
    return NS(BufferArrayObj_add_copy)( buffer, other );
}

/* end: sixtracklib/common/buffer/buffer_array_object.c */
