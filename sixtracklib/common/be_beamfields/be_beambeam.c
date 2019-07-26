#include "sixtracklib/common/be_beambeam/be_beambeam.h"

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/be_beambeam/gauss_fields.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

/* ************************************************************************* */
/* BeamBeam4D: */

NS(buffer_size_t) NS(BeamBeam4D_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const beam_beam )
{
    return NS(BeamBeam4D_get_required_num_dataptrs_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ),
        beam_beam, NS(Buffer_get_slot_size)( buffer ) );
}

NS(buffer_size_t)
NS(BeamBeam4D_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const beam_beam )
{
    return NS(BeamBeam4D_get_required_num_slots_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ),
        beam_beam, NS(Buffer_get_slot_size)( buffer ) );
}

bool NS(BeamBeam4D_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_size,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    typedef NS(buffer_size_t)  buf_size_t;
    typedef NS(BeamBeam4D) elem_t;

    buf_size_t const sizes[]  = { sizeof( SIXTRL_REAL_T ) };
    buf_size_t const counts[] = { data_size };
    buf_size_t num_dataptrs   = ( buf_size_t )0u;

    elem_t temp_obj;
    NS(BeamBeam4D_preset)( &temp_obj );
    NS(BeamBeam4D_set_data_size)( &temp_obj, data_size );

    num_dataptrs = NS(BeamBeam4D_get_required_num_dataptrs)( buffer, &temp_obj );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

    return NS(Buffer_can_add_object)( buffer, sizeof( elem_t ),
        num_dataptrs, &sizes[ 0 ], &counts[ 0 ], ptr_requ_objects,
            ptr_requ_slots, ptr_requ_dataptrs );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam4D)* NS(BeamBeam4D_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const data_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(BeamBeam4D) elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const offsets[] = { offsetof( elem_t, data ) };
    buf_size_t const sizes[]   = { sizeof( SIXTRL_REAL_T ) };
    buf_size_t const counts[]  = { data_size };
    buf_size_t num_dataptrs    = ( buf_size_t )0u;

    elem_t temp_obj;
    NS(BeamBeam4D_preset)( &temp_obj );
    NS(BeamBeam4D_set_data_size)( &temp_obj, data_size );

    num_dataptrs = NS(BeamBeam4D_get_required_num_dataptrs)( buffer, &temp_obj );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_BEAM_BEAM_4D), num_dataptrs,
                &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam4D)* NS(BeamBeam4D_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const  data_size,
    NS(beambeam4d_real_ptr_t) SIXTRL_RESTRICT input_data )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(BeamBeam4D) elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const offsets[] = { offsetof( elem_t, data ) };
    buf_size_t const sizes[]   = { sizeof( SIXTRL_REAL_T ) };
    buf_size_t const counts[]  = { data_size };
    buf_size_t num_dataptrs    = ( buf_size_t )0u;

    elem_t temp_obj;
    NS(BeamBeam4D_preset)( &temp_obj );
    NS(BeamBeam4D_set_data_size)( &temp_obj, data_size );
    NS(BeamBeam4D_assign_data_ptr)( &temp_obj, input_data );
    num_dataptrs = NS(BeamBeam4D_get_required_num_dataptrs)(
        buffer, SIXTRL_NULLPTR );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_BEAM_BEAM_4D), num_dataptrs,
                &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam4D)* NS(BeamBeam4D_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT orig )
{
    return NS(BeamBeam4D_add)( buffer, NS(BeamBeam4D_get_data_size)( orig ),
        NS(BeamBeam4D_get_data)( ( NS(BeamBeam4D)* )orig ) );
}

/* ************************************************************************* */
/* SpaceChargeCoasting: */


NS(buffer_size_t) NS(SpaceChargeCoasting_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const beam_beam )
{
    return NS(SpaceChargeCoasting_get_required_num_dataptrs_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ),
        beam_beam, NS(Buffer_get_slot_size)( buffer ) );
}

NS(buffer_size_t)
NS(SpaceChargeCoasting_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const beam_beam )
{
    return NS(SpaceChargeCoasting_get_required_num_slots_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ),
        beam_beam, NS(Buffer_get_slot_size)( buffer ) );
}

bool NS(SpaceChargeCoasting_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_size,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    typedef NS(buffer_size_t)  buf_size_t;
    typedef NS(SpaceChargeCoasting) elem_t;

    buf_size_t const sizes[]  = { sizeof( SIXTRL_REAL_T ) };
    buf_size_t const counts[] = { data_size };
    buf_size_t num_dataptrs   = ( buf_size_t )0u;

    elem_t temp_obj;
    NS(SpaceChargeCoasting_preset)( &temp_obj );
    NS(SpaceChargeCoasting_set_data_size)( &temp_obj, data_size );

    num_dataptrs = NS(SpaceChargeCoasting_get_required_num_dataptrs)(
        buffer, &temp_obj );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

    return NS(Buffer_can_add_object)( buffer, sizeof( elem_t ),
        num_dataptrs, &sizes[ 0 ], &counts[ 0 ], ptr_requ_objects,
            ptr_requ_slots, ptr_requ_dataptrs );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(SpaceChargeCoasting)* NS(SpaceChargeCoasting_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const data_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(SpaceChargeCoasting) elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const offsets[] = { offsetof( elem_t, data ) };
    buf_size_t const sizes[]   = { sizeof( SIXTRL_REAL_T ) };
    buf_size_t const counts[]  = { data_size };
    buf_size_t num_dataptrs    = ( buf_size_t )0u;

    elem_t temp_obj;
    NS(SpaceChargeCoasting_preset)( &temp_obj );
    NS(SpaceChargeCoasting_set_data_size)( &temp_obj, data_size );

    num_dataptrs = NS(SpaceChargeCoasting_get_required_num_dataptrs)( buffer, &temp_obj );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_SPACE_CHARGE_COASTING), num_dataptrs,
                &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(SpaceChargeCoasting)* NS(SpaceChargeCoasting_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const  data_size,
    NS(beambeam4d_real_ptr_t) SIXTRL_RESTRICT input_data )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(SpaceChargeCoasting) elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const offsets[] = { offsetof( elem_t, data ) };
    buf_size_t const sizes[]   = { sizeof( SIXTRL_REAL_T ) };
    buf_size_t const counts[]  = { data_size };
    buf_size_t num_dataptrs    = ( buf_size_t )0u;

    elem_t temp_obj;
    NS(SpaceChargeCoasting_preset)( &temp_obj );
    NS(SpaceChargeCoasting_set_data_size)( &temp_obj, data_size );
    NS(SpaceChargeCoasting_assign_data_ptr)( &temp_obj, input_data );
    num_dataptrs = NS(SpaceChargeCoasting_get_required_num_dataptrs)(
        buffer, SIXTRL_NULLPTR );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_SPACE_CHARGE_COASTING), num_dataptrs,
                &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(SpaceChargeCoasting)* NS(SpaceChargeCoasting_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT orig )
{
    return NS(SpaceChargeCoasting_add)( buffer,
            NS(SpaceChargeCoasting_get_data_size)( orig ),
            NS(SpaceChargeCoasting_get_data)(
                ( NS(SpaceChargeCoasting)* )orig ) );
}

/* ************************************************************************* */
/* SpaceChargeBunched: */

NS(buffer_size_t) NS(SpaceChargeBunched_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const beam_beam )
{
    return NS(SpaceChargeBunched_get_required_num_dataptrs_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ),
        beam_beam, NS(Buffer_get_slot_size)( buffer ) );
}

NS(buffer_size_t)
NS(SpaceChargeBunched_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const beam_beam )
{
    return NS(SpaceChargeBunched_get_required_num_slots_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ),
        beam_beam, NS(Buffer_get_slot_size)( buffer ) );
}

bool NS(SpaceChargeBunched_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_size,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    typedef NS(buffer_size_t)  buf_size_t;
    typedef NS(SpaceChargeBunched) elem_t;

    buf_size_t const sizes[]  = { sizeof( SIXTRL_REAL_T ) };
    buf_size_t const counts[] = { data_size };
    buf_size_t num_dataptrs   = ( buf_size_t )0u;

    elem_t temp_obj;
    NS(SpaceChargeBunched_preset)( &temp_obj );
    NS(SpaceChargeBunched_set_data_size)( &temp_obj, data_size );

    num_dataptrs = NS(SpaceChargeBunched_get_required_num_dataptrs)(
        buffer, &temp_obj );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

    return NS(Buffer_can_add_object)( buffer, sizeof( elem_t ),
        num_dataptrs, &sizes[ 0 ], &counts[ 0 ], ptr_requ_objects,
            ptr_requ_slots, ptr_requ_dataptrs );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(SpaceChargeBunched)* NS(SpaceChargeBunched_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const data_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(SpaceChargeBunched) elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const offsets[] = { offsetof( elem_t, data ) };
    buf_size_t const sizes[]   = { sizeof( SIXTRL_REAL_T ) };
    buf_size_t const counts[]  = { data_size };
    buf_size_t num_dataptrs    = ( buf_size_t )0u;

    elem_t temp_obj;
    NS(SpaceChargeBunched_preset)( &temp_obj );
    NS(SpaceChargeBunched_set_data_size)( &temp_obj, data_size );

    num_dataptrs = NS(SpaceChargeBunched_get_required_num_dataptrs)(
        buffer, &temp_obj );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_SPACE_CHARGE_BUNCHED), num_dataptrs,
                &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(SpaceChargeBunched)* NS(SpaceChargeBunched_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const  data_size,
    NS(beambeam4d_real_ptr_t) SIXTRL_RESTRICT input_data )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(SpaceChargeBunched) elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const offsets[] = { offsetof( elem_t, data ) };
    buf_size_t const sizes[]   = { sizeof( SIXTRL_REAL_T ) };
    buf_size_t const counts[]  = { data_size };
    buf_size_t num_dataptrs    = ( buf_size_t )0u;

    elem_t temp_obj;
    NS(SpaceChargeBunched_preset)( &temp_obj );
    NS(SpaceChargeBunched_set_data_size)( &temp_obj, data_size );
    NS(SpaceChargeBunched_assign_data_ptr)( &temp_obj, input_data );
    num_dataptrs = NS(SpaceChargeBunched_get_required_num_dataptrs)(
        buffer, SIXTRL_NULLPTR );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_SPACE_CHARGE_BUNCHED), num_dataptrs,
                &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(SpaceChargeBunched)* NS(SpaceChargeBunched_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT orig )
{
    return NS(SpaceChargeBunched_add)( buffer,
       NS(SpaceChargeBunched_get_data_size)( orig ),
       NS(SpaceChargeBunched_get_data)( ( NS(SpaceChargeBunched)* )orig ) );
}

/* ************************************************************************* */
/* BeamBeam6D: */

NS(buffer_size_t) NS(BeamBeam6D_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const beam_beam )
{
    return NS(BeamBeam6D_get_required_num_dataptrs_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ),
        beam_beam, NS(Buffer_get_slot_size)( buffer ) );
}

NS(buffer_size_t)
NS(BeamBeam6D_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const beam_beam )
{
    return NS(BeamBeam6D_get_required_num_slots_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ),
        beam_beam, NS(Buffer_get_slot_size)( buffer ) );
}

bool NS(BeamBeam6D_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_size,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    typedef NS(buffer_size_t)  buf_size_t;
    typedef NS(BeamBeam6D) elem_t;

    buf_size_t const sizes[]  = { sizeof( SIXTRL_REAL_T ) };
    buf_size_t const counts[] = { data_size };
    buf_size_t num_dataptrs   = ( buf_size_t )0u;

    elem_t temp_obj;
    NS(BeamBeam6D_preset)( &temp_obj );
    NS(BeamBeam6D_set_data_size)( &temp_obj, data_size );

    num_dataptrs = NS(BeamBeam6D_get_required_num_dataptrs)( buffer, &temp_obj );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

    return NS(Buffer_can_add_object)( buffer, sizeof( elem_t ),
        num_dataptrs, &sizes[ 0 ], &counts[ 0 ], ptr_requ_objects,
            ptr_requ_slots, ptr_requ_dataptrs );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam6D)* NS(BeamBeam6D_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const data_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(BeamBeam6D) elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const offsets[] = { offsetof( elem_t, data ) };
    buf_size_t const sizes[]   = { sizeof( SIXTRL_REAL_T ) };
    buf_size_t const counts[]  = { data_size };
    buf_size_t num_dataptrs    = ( buf_size_t )0u;

    elem_t temp_obj;
    NS(BeamBeam6D_preset)( &temp_obj );
    NS(BeamBeam6D_set_data_size)( &temp_obj, data_size );

    num_dataptrs = NS(BeamBeam6D_get_required_num_dataptrs)( buffer, &temp_obj );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_BEAM_BEAM_6D), num_dataptrs,
                &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam6D)* NS(BeamBeam6D_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const  data_size,
    NS(beambeam4d_real_ptr_t) SIXTRL_RESTRICT input_data )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(BeamBeam6D) elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    buf_size_t const offsets[] = { offsetof( elem_t, data ) };
    buf_size_t const sizes[]   = { sizeof( SIXTRL_REAL_T ) };
    buf_size_t const counts[]  = { data_size };
    buf_size_t num_dataptrs    = ( buf_size_t )0u;

    elem_t temp_obj;
    NS(BeamBeam6D_preset)( &temp_obj );
    NS(BeamBeam6D_set_data_size)( &temp_obj, data_size );
    NS(BeamBeam6D_assign_data_ptr)( &temp_obj, input_data );
    num_dataptrs = NS(BeamBeam6D_get_required_num_dataptrs)(
        buffer, SIXTRL_NULLPTR );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_BEAM_BEAM_6D), num_dataptrs,
                &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam6D)* NS(BeamBeam6D_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT orig )
{
    return NS(BeamBeam6D_add)( buffer,
        NS(BeamBeam6D_get_data_size)( orig ),
        NS(BeamBeam6D_get_data)( ( NS(BeamBeam6D)* )orig ) );
}

/* end: sixtracklib/common/be_beambeam/be_beambeam.c */
