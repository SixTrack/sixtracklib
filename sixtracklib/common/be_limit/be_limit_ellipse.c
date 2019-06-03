#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_limit/be_limit_ellipse.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
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

#if !defined( _GPUCODE )

NS(buffer_size_t) NS(LimitEllipse_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit )
{
    return NS(LimitEllipse_get_required_num_dataptrs_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ), limit, 
        NS(Buffer_get_slot_size)( buffer ) );
}

NS(buffer_size_t) NS(LimitEllipse_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC  const NS(LimitEllipse) *const SIXTRL_RESTRICT limit)
{
    return NS(LimitEllipse_get_required_num_slots_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ), limit, 
        NS(Buffer_get_slot_size)( buffer ) );
}

bool NS(LimitEllipse_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT req_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT req_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT req_dataptrs )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(LimitEllipse) elem_t;

    buf_size_t const num_dataptrs =
        NS(LimitEllipse_get_required_num_dataptrs)( buffer, SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes  = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts = SIXTRL_NULLPTR;
    
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return NS(Buffer_can_add_object)( buffer, sizeof( elem_t ),
        num_dataptrs, sizes, counts, req_objects, req_slots, req_dataptrs );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse)* NS(LimitEllipse_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(LimitEllipse) elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const num_dataptrs =
        NS(LimitEllipse_get_required_num_dataptrs)( buffer, SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;
    
    elem_t temp_obj;
    NS(LimitEllipse_preset)( &temp_obj );

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );
    
    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
        NS(OBJECT_TYPE_LIMIT_ELLIPSE), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse)* NS(LimitEllipse_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(particle_real_t) const x_origin, 
    NS(particle_real_t) const y_origin,
    NS(particle_real_t) const x_semi_axis, 
    NS(particle_real_t) const y_semi_axis )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(LimitEllipse) elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const num_dataptrs =
        NS(LimitEllipse_get_required_num_dataptrs)( buffer, SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    elem_t temp_obj;
    
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );
    
    NS(LimitEllipse_set_x_origin)( &temp_obj, x_origin );
    NS(LimitEllipse_set_y_origin)( &temp_obj, y_origin );
    NS(LimitEllipse_set_half_axes)( &temp_obj, x_semi_axis, y_semi_axis );
    
    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_LIMIT_ELLIPSE), num_dataptrs, offsets, 
                sizes, counts ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse)* NS(LimitEllipse_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit )
{
    return NS(LimitEllipse_add)( buffer, 
        NS(LimitEllipse_get_x_origin)( limit ),
        NS(LimitEllipse_get_y_origin)( limit ), 
        NS(LimitEllipse_get_x_half_axis)( limit ),
        NS(LimitEllipse_get_y_half_axis)( limit ) );
}

#endif /* !defined( _GPUCODE )*/

/* end: sixtracklib/common/be_limit/be_limit_ellipse.c */
