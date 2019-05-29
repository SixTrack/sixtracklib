#include "sixtracklib/common/be_limit/be_limit.h"

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


NS(buffer_size_t) NS(Limit_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT limit )
{
    return NS(Limit_get_required_num_dataptrs_on_managed_buffer)(
        NS(Buffer_get_data_begin)( buffer ), limit,
        NS(Buffer_get_slot_size)( buffer ) );
}

NS(buffer_size_t) NS(Limit_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC  const NS(Limit) *const SIXTRL_RESTRICT limit )
{
    return NS(Limit_get_required_num_slots_on_managed_buffer)(
        NS(Buffer_get_data_begin)( buffer ), limit,
        NS(Buffer_get_slot_size)( buffer ) );
}

bool NS(Limit_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    typedef NS(buffer_size_t)  buf_size_t;
    typedef NS(Limit) elem_t;

    buf_size_t const num_dataptrs =
        NS(Limit_get_required_num_dataptrs)( SIXTRL_NULLPTR, SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    return NS(Buffer_can_add_object)( buffer, sizeof( elem_t ),
        num_dataptrs, sizes, counts, requ_objects, requ_slots, requ_dataptrs );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(Limit)* NS(Limit_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(buffer_size_t)   buf_size_t;
    typedef NS(particle_real_t) real_t;
    typedef NS(Limit)           elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const num_dataptrs =
        NS(Limit_get_required_num_dataptrs)( SIXTRL_NULLPTR, SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    elem_t temp_obj;
    NS(Limit_set_x_limit)( &temp_obj, NS(BE_LIMIT_DEFAULT_X_LIMIT) );
    NS(Limit_set_y_limit)( &temp_obj, NS(BE_LIMIT_DEFAULT_Y_LIMIT) );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_LIMIT), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(Limit)* NS(Limit_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(particle_real_t) const x_limit, NS(particle_real_t) const y_limit )
{
    typedef NS(buffer_size_t)   buf_size_t;
    typedef NS(particle_real_t) real_t;
    typedef NS(Limit)           elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const num_dataptrs =
        NS(Limit_get_required_num_dataptrs)( SIXTRL_NULLPTR, SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    elem_t temp_obj;
    NS(Limit_preset)( &temp_obj );
    NS(Limit_set_x_limit)( &temp_obj, x_limit );
    NS(Limit_set_y_limit)( &temp_obj, y_limit );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_LIMIT), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(Limit)* NS(Limit_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT limit )
{
    return NS(Limit_add)( buffer, NS(Limit_get_x_limit)( limit ),
                          NS(Limit_get_y_limit)( limit ) );
}

/* end: sixtracklib/common/be_limit/be_limit.c */
