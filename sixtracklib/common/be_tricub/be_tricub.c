#include "sixtracklib/common/be_tricub/be_tricub.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"

NS(buffer_size_t) NS(TriCub_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const tricub )
{
    return NS(TriCub_get_required_num_dataptrs_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ), tricub,
        NS(Buffer_get_slot_size)( buffer ) );
}

NS(buffer_size_t) NS(TriCub_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const tricub )
{
    return NS(TriCub_get_required_num_slots_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ), tricub,
        NS(Buffer_get_slot_size)( buffer ) );
}

bool NS(TriCub_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(be_tricub_int_t) const nx, NS(be_tricub_int_t) const ny,
    NS(be_tricub_int_t) const nz,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    typedef NS(buffer_size_t)  buf_size_t;
    typedef NS(TriCub)      elem_t;

    bool can_be_added = false;
    NS(be_tricub_int_t) const temp_data_size =
        nx * ny * nz * SIXTRL_BE_TRICUBMAP_MATRIX_DIM;

    if( temp_data_size > ( NS(be_tricub_int_t) )0u )
    {
        buf_size_t const data_size = ( buf_size_t )temp_data_size;
        buf_size_t const sizes[]   = { sizeof( SIXTRL_REAL_T ) };
        buf_size_t const counts[]  = { data_size };
        buf_size_t num_dataptrs    = ( buf_size_t )0u;

        elem_t temp_obj;
        NS(TriCub_preset)( &temp_obj );
        NS(TriCub_init)( &temp_obj, nx, ny, nz );

        num_dataptrs = NS(TriCub_get_required_num_dataptrs)(
            buffer, &temp_obj );

        SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

        can_be_added = NS(Buffer_can_add_object)( buffer, sizeof( elem_t ),
            num_dataptrs, &sizes[ 0 ], &counts[ 0 ], ptr_requ_objects,
                ptr_requ_slots, ptr_requ_dataptrs );
    }

    return can_be_added;
}

SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)* NS(TriCub_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_tricub_int_t) const nx, NS(be_tricub_int_t) const ny,
    NS(be_tricub_int_t) const nz )
{
    typedef NS(buffer_size_t)  buf_size_t;
    typedef NS(TriCub)  elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    ptr_to_elem_t ptr_elem = SIXTRL_NULLPTR;

    NS(be_tricub_int_t) const temp_data_size =
        nx * ny * nz * SIXTRL_BE_TRICUBMAP_MATRIX_DIM;

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( temp_data_size > ( NS(be_tricub_int_t) )0u ) )
    {
        buf_size_t const data_size = ( buf_size_t )temp_data_size;
        buf_size_t const sizes[]   = { sizeof( SIXTRL_REAL_T ) };
        buf_size_t const counts[]  = { data_size };
        buf_size_t const offsets[] = { offsetof( elem_t, phi ) };
        buf_size_t num_dataptrs    = ( buf_size_t )0u;

        elem_t temp_obj;
        NS(TriCub_preset)( &temp_obj );
        NS(TriCub_init)( &temp_obj, nx, ny, nz );

        num_dataptrs = NS(TriCub_get_required_num_dataptrs)(
            buffer, &temp_obj );

        SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

        ptr_elem = ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_TRICUB), num_dataptrs,
                &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
    }

    return ptr_elem;
}

SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)* NS(TriCub_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_tricub_int_t) const nx, NS(be_tricub_int_t) const ny,
    NS(be_tricub_int_t) const nz,
    NS(be_tricub_real_t) const x0,
    NS(be_tricub_real_t) const y0, NS(be_tricub_real_t) const z0,
    NS(be_tricub_real_t) const dx, NS(be_tricub_real_t) const dy,
    NS(be_tricub_real_t) const dz,
    NS(be_tricub_ptr_real_t) ptr_to_phi_data )
{
    typedef NS(buffer_size_t)  buf_size_t;
    typedef NS(TriCub)  elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_to_elem_t;

    ptr_to_elem_t ptr_elem = SIXTRL_NULLPTR;

    NS(be_tricub_int_t) const temp_data_size =
        nx * ny * nz * SIXTRL_BE_TRICUBMAP_MATRIX_DIM;

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( temp_data_size > ( NS(be_tricub_int_t) )0u ) )
    {
        buf_size_t const data_size = ( buf_size_t )temp_data_size;
        buf_size_t const sizes[]   = { sizeof( SIXTRL_REAL_T ) };
        buf_size_t const counts[]  = { data_size };
        buf_size_t const offsets[] = { offsetof( elem_t, phi ) };
        buf_size_t num_dataptrs    = ( buf_size_t )0u;

        elem_t temp_obj;
        NS(TriCub_preset)( &temp_obj );
        NS(TriCub_init)( &temp_obj, nx, ny, nz );

        NS(TriCub_set_x0)( &temp_obj, x0 );
        NS(TriCub_set_y0)( &temp_obj, y0 );
        NS(TriCub_set_z0)( &temp_obj, z0 );

        NS(TriCub_set_dx)( &temp_obj, dx );
        NS(TriCub_set_dy)( &temp_obj, dy );
        NS(TriCub_set_dz)( &temp_obj, dz );

        NS(TriCub_assign_ptr_phi)( &temp_obj, ptr_to_phi_data );

        num_dataptrs = NS(TriCub_get_required_num_dataptrs)(
            buffer, &temp_obj );

        SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

        ptr_elem = ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_TRICUB), num_dataptrs,
                &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
    }

    return ptr_elem;
}

SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)* NS(TriCub_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT orig )
{
    return NS(TriCub_add)( buffer,
        NS(TriCub_get_nx)( orig ), NS(TriCub_get_ny)( orig ),
        NS(TriCub_get_nz)( orig ),
        NS(TriCub_get_x0)( orig ), NS(TriCub_get_y0)( orig ),
        NS(TriCub_get_z0)( orig ),
        NS(TriCub_get_dx)( orig ), NS(TriCub_get_dy)( orig ),
        NS(TriCub_get_dz)( orig ),
        ( NS(be_tricub_ptr_real_t) )NS(TriCub_get_ptr_const_phi)( orig ) );
}

/* end: sixtracklib/common/be_tricub/be_tricub.c */
