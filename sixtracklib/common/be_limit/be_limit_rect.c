#include "sixtracklib/common/be_limit/be_limit_rect.h"

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_limit/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


NS(buffer_size_t) NS(LimitRect_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT limit )
{
    return NS(LimitRect_get_required_num_dataptrs_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ), limit,
        NS(Buffer_get_slot_size)( buffer ) );
}

NS(buffer_size_t) NS(LimitRect_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC  const NS(LimitRect) *const SIXTRL_RESTRICT limit )
{
    return NS(LimitRect_get_required_num_slots_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ), limit,
        NS(Buffer_get_slot_size)( buffer ) );
}

bool NS(LimitRect_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    typedef NS(buffer_size_t)  buf_size_t;
    typedef NS(LimitRect) elem_t;

    buf_size_t const num_dataptrs =
        NS(LimitRect_get_required_num_dataptrs)( buffer, SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes  = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts = SIXTRL_NULLPTR;

    return NS(Buffer_can_add_object)( buffer, sizeof( elem_t ),
        num_dataptrs, sizes, counts, requ_objects, requ_slots, requ_dataptrs );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect)* NS(LimitRect_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(buffer_size_t)   buf_size_t;
    typedef NS(LimitRect)           elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const num_dataptrs =
        NS(LimitRect_get_required_num_dataptrs)( buffer, SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    elem_t temp_obj;
    NS(LimitRect_preset)( &temp_obj );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
        NS(OBJECT_TYPE_LIMIT_RECT), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect)* NS(LimitRect_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(particle_real_t) const min_x, NS(particle_real_t) const max_x,
    NS(particle_real_t) const min_y, NS(particle_real_t) const max_y )
{
    typedef NS(buffer_size_t)   buf_size_t;
    typedef NS(LimitRect)           elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const num_dataptrs =
        NS(LimitRect_get_required_num_dataptrs)( buffer, SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    elem_t temp_obj;
    NS(LimitRect_set_min_x)( &temp_obj, min_x );
    NS(LimitRect_set_max_x)( &temp_obj, max_x );
    NS(LimitRect_set_min_y)( &temp_obj, min_y );
    NS(LimitRect_set_max_y)( &temp_obj, max_y );
    
    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
        NS(OBJECT_TYPE_LIMIT_RECT), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect)* NS(LimitRect_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT limit )
{
    return NS(LimitRect_add)( buffer, 
        NS(LimitRect_get_min_x)( limit ), NS(LimitRect_get_max_x)( limit ),
        NS(LimitRect_get_min_y)( limit ), NS(LimitRect_get_max_y)( limit ) );
}

/* end: sixtracklib/common/be_limit/be_limit.c */
