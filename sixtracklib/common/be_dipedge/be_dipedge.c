#include "sixtracklib/common/be_dipedge/be_dipedge.h"


#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/constants.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

NS(buffer_size_t) NS(DipoleEdge_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    return NS(DipoleEdge_get_required_num_dataptrs_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ), dipedge, 
        NS(Buffer_get_slot_size)( buffer ) );
}

NS(buffer_size_t) NS(DipoleEdge_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC  const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    return NS(DipoleEdge_get_required_num_slots_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ), dipedge, 
        NS(Buffer_get_slot_size)( buffer ) );
}

bool NS(DipoleEdge_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    typedef NS(buffer_size_t)  buf_size_t;
    typedef NS(DipoleEdge) elem_t;

    buf_size_t const num_dataptrs =
        NS(DipoleEdge_get_required_num_dataptrs)( buffer, SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;
    
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return NS(Buffer_can_add_object)( buffer, sizeof( elem_t ),
        num_dataptrs, sizes, counts, requ_objects, requ_slots, requ_dataptrs );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(DipoleEdge)* NS(DipoleEdge_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(DipoleEdge) elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const num_dataptrs =
        NS(DipoleEdge_get_required_num_dataptrs)( buffer, SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;
    
    elem_t temp_obj;
    
    NS(DipoleEdge_preset)( &temp_obj );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_DIPEDGE), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(DipoleEdge)* NS(DipoleEdge_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(dipedge_real_t) const inv_rho, 
    NS(dipedge_real_t) const rot_angle_deg, 
    NS(dipedge_real_t) const b, 
    NS(dipedge_real_t) const tilt_angle_deg )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(DipoleEdge) elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const num_dataptrs =
        NS(DipoleEdge_get_required_num_dataptrs)( buffer, SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;
    
    elem_t temp_obj;
    
    NS(DipoleEdge_set_inv_rho)( &temp_obj, inv_rho );
    NS(DipoleEdge_set_rot_angle)( &temp_obj, rot_angle_deg );
    NS(DipoleEdge_set_b)( &temp_obj, b );
    NS(DipoleEdge_set_tilt_angle)( &temp_obj, tilt_angle_deg );
    
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_DIPEDGE), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(DipoleEdge)* NS(DipoleEdge_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(DipoleEdge) elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const num_dataptrs =
        NS(DipoleEdge_get_required_num_dataptrs)( buffer, SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;
    
    SIXTRL_ASSERT( dipedge != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, dipedge, sizeof( elem_t ),
            NS(OBJECT_TYPE_DIPEDGE), num_dataptrs, offsets, sizes, counts ) );
}

/* end: sixtracklib/common/be_dipedge/be_dipedge.c */
