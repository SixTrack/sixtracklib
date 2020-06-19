#include "sixtracklib/common/be_tricub/be_tricub.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/buffer/assign_address_item.h"
#include "sixtracklib/common/buffer.h"

NS(arch_status_t) NS(TriCubData_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t const num_dataptrs = NS(TriCubData_num_dataptrs)( data );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );
    ( void )slot_size;

    if( ( offsets != SIXTRL_NULLPTR ) && ( max_num_offsets >= num_dataptrs ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, offsets, max_num_offsets, ZERO );
        offsets[ 0 ] = NS(TriCubData_ptr_offset)( data );

        SIXTRL_ASSERT( slot_size > ZERO );
        SIXTRL_ASSERT( ( offsets[ 0 ] % slot_size ) == ZERO );
        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

NS(arch_status_t) NS(TriCubData_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    typedef NS(buffer_size_t) buf_size_t;

    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    buf_size_t const num_dataptrs = NS(TriCubData_num_dataptrs)( data );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

    if( ( counts != SIXTRL_NULLPTR ) && ( max_num_counts >= num_dataptrs ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, counts, max_num_counts, ZERO );
        counts[ 0 ] = NS(TriCubData_table_size)( data );
        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

NS(arch_status_t) NS(TriCubData_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t const num_dataptrs = NS(TriCubData_num_dataptrs)( data );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );
    ( void )slot_size;

    if( ( sizes != SIXTRL_NULLPTR ) && ( max_num_sizes >= num_dataptrs ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, sizes, max_num_sizes, ZERO );
        sizes[ 0 ] = sizeof( NS(be_tricub_real_t) );
        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}


SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const*
NS(TriCubData_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    return NS(TriCubData_const_from_obj_index)(
        NS(Buffer_get_const_object)( buffer, index ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* NS(TriCubData_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    return NS(TriCubData_from_obj_index)(
        NS(Buffer_get_object)( buffer, index ) );
}

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

NS(object_type_id_t) NS(TriCubData_type_id_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    return NS(TriCubData_type_id)( data );
}

NS(arch_size_t) NS(TriCubData_ptr_offset)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    ( void )data;
    return ( NS(arch_size_t) )offsetof( NS(TriCubData), table_addr );
}

/* ------------------------------------------------------------------------- */

bool NS(TriCubData_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(be_tricub_int_t) const nx, NS(be_tricub_int_t) const ny,
    NS(be_tricub_int_t) const nz,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    typedef NS(buffer_size_t) buf_size_t;

    bool can_be_added = false;
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    NS(TriCubData) tricub_data;
    buf_size_t const num_dataptrs = NS(TriCubData_num_dataptrs)( &tricub_data );
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    buf_size_t sizes[]  = { ( buf_size_t )0u };
    buf_size_t counts[] = { ( buf_size_t )0u };

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

    NS(TriCubData_preset)( &tricub_data );
    NS(TriCubData_set_nx)( &tricub_data, nx );
    NS(TriCubData_set_ny)( &tricub_data, ny );
    NS(TriCubData_set_nz)( &tricub_data, nz );

    status = NS(TriCubData_attributes_sizes)(
        &sizes[ 0 ], num_dataptrs, &tricub_data, slot_size );

    if( status == NS(ARCH_STATUS_SUCCESS) )
    {
        status = NS(TriCubData_attributes_counts)(
            &counts[ 0 ], num_dataptrs, &tricub_data );
    }

    if( status == NS(ARCH_STATUS_SUCCESS) )
    {
        can_be_added = NS(Buffer_can_add_object)( buffer,
            sizeof( NS(TriCubData) ), num_dataptrs, sizes, counts,
                ptr_requ_objects, ptr_requ_slots, ptr_requ_dataptrs );
    }

    return can_be_added;
}

SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* NS(TriCubData_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_tricub_int_t) const nx, NS(be_tricub_int_t) const ny,
    NS(be_tricub_int_t) const nz )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* ptr_tricub_data = SIXTRL_NULLPTR;
    NS(arch_size_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    NS(TriCubData) tricub_data;
    buf_size_t const num_dataptrs = NS(TriCubData_num_dataptrs)( &tricub_data );

    buf_size_t offsets[] = { ( buf_size_t )0u };
    buf_size_t sizes[]   = { ( buf_size_t )0u };
    buf_size_t counts[]  = { ( buf_size_t )0u };
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

    NS(TriCubData_preset)( &tricub_data );
    NS(TriCubData_set_nx)( &tricub_data, nx );
    NS(TriCubData_set_ny)( &tricub_data, ny );
    NS(TriCubData_set_nz)( &tricub_data, nz );

    status = NS(TriCubData_attributes_offsets)(
        &offsets[ 0 ], num_dataptrs, &tricub_data, slot_size );

    if( status == NS(ARCH_STATUS_SUCCESS) )
    {
        status = NS(TriCubData_attributes_counts)(
            &counts[ 0 ], num_dataptrs, &tricub_data );

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(TriCubData_attributes_sizes)(
                &sizes[ 0 ], num_dataptrs, &tricub_data, slot_size );
        }
    }

    if( status == NS(ARCH_STATUS_SUCCESS) )
    {
        ptr_tricub_data = ( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* )(
            uintptr_t )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
                buffer, &tricub_data, sizeof( tricub_data ),
                    NS(TriCubData_type_id)( &tricub_data ), num_dataptrs,
                        &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
    }

    return ptr_tricub_data;
}

SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_tricub_real_t) const x0, NS(be_tricub_real_t) const dx,
    NS(be_tricub_int_t)  const nx,
    NS(be_tricub_real_t) const y0, NS(be_tricub_real_t) const dy,
    NS(be_tricub_int_t)  const ny,
    NS(be_tricub_real_t) const z0, NS(be_tricub_real_t) const dz,
    NS(be_tricub_int_t)  const nz,
    NS(be_tricub_int_t)  const mirror_x, NS(be_tricub_int_t)  const mirror_y,
    NS(be_tricub_int_t)  const mirror_z, NS(buffer_addr_t) const table_addr )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* ptr_tricub_data = SIXTRL_NULLPTR;
    NS(arch_size_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    NS(TriCubData) tricub_data;
    buf_size_t const num_dataptrs = NS(TriCubData_num_dataptrs)( &tricub_data );

    buf_size_t offsets[] = { ( buf_size_t )0u };
    buf_size_t sizes[]   = { ( buf_size_t )0u };
    buf_size_t counts[]  = { ( buf_size_t )0u };
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );
    NS(TriCubData_preset)( &tricub_data );

    NS(TriCubData_set_x0)( &tricub_data, x0 );
    NS(TriCubData_set_dx)( &tricub_data, dx );
    NS(TriCubData_set_nx)( &tricub_data, nx );

    NS(TriCubData_set_y0)( &tricub_data, y0 );
    NS(TriCubData_set_dy)( &tricub_data, dy );
    NS(TriCubData_set_ny)( &tricub_data, ny );

    NS(TriCubData_set_z0)( &tricub_data, z0 );
    NS(TriCubData_set_dz)( &tricub_data, dz );
    NS(TriCubData_set_nz)( &tricub_data, nz );

    NS(TriCubData_set_mirror_x)( &tricub_data, mirror_x );
    NS(TriCubData_set_mirror_y)( &tricub_data, mirror_y );
    NS(TriCubData_set_mirror_z)( &tricub_data, mirror_z );

    NS(TriCubData_set_table_addr)( &tricub_data, table_addr );

    status = NS(TriCubData_attributes_offsets)(
        &offsets[ 0 ], num_dataptrs, &tricub_data, slot_size );

    if( status == NS(ARCH_STATUS_SUCCESS) )
    {
        status = NS(TriCubData_attributes_counts)(
            &counts[ 0 ], num_dataptrs, &tricub_data );

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(TriCubData_attributes_sizes)(
                &sizes[ 0 ], num_dataptrs, &tricub_data, slot_size );
        }
    }

    if( status == NS(ARCH_STATUS_SUCCESS) )
    {
        ptr_tricub_data = ( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* )(
            uintptr_t )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
                buffer, &tricub_data, sizeof( tricub_data ),
                    NS(TriCubData_type_id)( &tricub_data ), num_dataptrs,
                        &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
    }

    return ptr_tricub_data;
}

SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* NS(TriCubData_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    return NS(TriCubData_add)( buffer,
        NS(TriCubData_x0)( data ), NS(TriCubData_dx)( data ),
        NS(TriCubData_nx)( data ), NS(TriCubData_y0)( data ),
        NS(TriCubData_dy)( data ), NS(TriCubData_ny)( data ),
        NS(TriCubData_z0)( data ), NS(TriCubData_dz)( data ),
        NS(TriCubData_nz)( data ), NS(TriCubData_mirror_x)( data ),
        NS(TriCubData_mirror_y)( data ), NS(TriCubData_mirror_z)( data ),
        NS(TriCubData_table_addr)( data ) );
}

/* ========================================================================= */

SIXTRL_BE_ARGPTR_DEC NS(TriCub) const* NS(TriCub_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    return NS(TriCub_const_from_obj_index)(
        NS(Buffer_get_const_object)( buffer, index ) );
}

SIXTRL_BE_ARGPTR_DEC NS(TriCub)* NS(TriCub_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    return NS(TriCub_from_obj_index)(
        NS(Buffer_get_object)( buffer, index ) );
}

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

NS(object_type_id_t) NS(TriCub_type_id_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    return NS(TriCub_type_id)( tricub );
}

NS(arch_size_t) NS(TriCub_data_addr_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    return ( NS(arch_size_t) )offsetof( NS(TriCub), data_addr );
}

/* ------------------------------------------------------------------------- */

bool NS(TriCub_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    NS(TriCub) tricub;
    NS(TriCub_preset)( &tricub );

    SIXTRL_ASSERT( ( NS(buffer_size_t) )0u ==
        NS(TriCub_num_dataptrs)( &tricub ) );
    ( void )ptr_requ_dataptrs;

    return NS(Buffer_can_add_trivial_object)( buffer, sizeof( tricub ),
        ptr_requ_objects, ptr_requ_slots );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)* NS(TriCub_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    NS(TriCub) tricub;
    NS(TriCub_preset)( &tricub );

    SIXTRL_ASSERT( ( NS(buffer_size_t) )0u ==
        NS(TriCub_num_dataptrs)( &tricub ) );

    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_trivial_object)( buffer,
            &tricub, sizeof( tricub ), NS(TriCub_type_id)( &tricub ) ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)* NS(TriCub_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_tricub_real_t) const x_shift, NS(be_tricub_real_t) const y_shift,
    NS(be_tricub_real_t) const tau_shift,
    NS(be_tricub_real_t) const dipolar_kick_px,
    NS(be_tricub_real_t) const dipolar_kick_py,
    NS(be_tricub_real_t) const dipolar_kick_ptau,
    NS(be_tricub_real_t) const length,
    NS(buffer_addr_t) const data_addr )
{
    NS(TriCub) tricub;
    NS(TriCub_preset)( &tricub );
    NS(TriCub_set_x_shift)( &tricub, x_shift );
    NS(TriCub_set_y_shift)( &tricub, y_shift );
    NS(TriCub_set_tau_shift)( &tricub, tau_shift );
    NS(TriCub_set_dipolar_kick_px)( &tricub, dipolar_kick_px );
    NS(TriCub_set_dipolar_kick_py)( &tricub, dipolar_kick_py );
    NS(TriCub_set_dipolar_kick_ptau)( &tricub, dipolar_kick_ptau );
    NS(TriCub_set_length)( &tricub, length );
    NS(TriCub_set_data_addr)( &tricub, data_addr );

    SIXTRL_ASSERT( ( NS(buffer_size_t) )0u ==
        NS(TriCub_num_dataptrs)( &tricub ) );

    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_trivial_object)( buffer,
            &tricub, sizeof( tricub ), NS(TriCub_type_id)( &tricub ) ) );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)* NS(TriCub_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( ( NS(buffer_size_t) )0u ==
        NS(TriCub_num_dataptrs)( tricub ) );

    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_trivial_object)( buffer,
            tricub, sizeof( NS(TriCub) ), NS(TriCub_type_id)( tricub ) ) );
}

/* ------------------------------------------------------------------------ */

SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
NS(TriCub_buffer_create_assign_address_item)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT assign_items_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const belements_index,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT tricub_data_buffer,
    NS(buffer_size_t) const tricub_data_index )
{
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
        assign_item = SIXTRL_NULLPTR;

    if( ( assign_items_buffer != SIXTRL_NULLPTR ) &&
        ( NS(TriCub_const_from_buffer)(
            belements, belements_index ) != SIXTRL_NULLPTR ) &&
        ( NS(TriCubData_const_from_buffer)(
            tricub_data_buffer, tricub_data_index ) != SIXTRL_NULLPTR ) )
    {
        assign_item = NS(AssignAddressItem_add)( assign_items_buffer,
            NS(OBJECT_TYPE_TRICUB), NS(ARCH_BEAM_ELEMENTS_BUFFER_ID),
                belements_index, offsetof( NS(TriCub), data_addr ),
            NS(OBJECT_TYPE_TRICUB_DATA), NS(ARCH_MIN_USER_DEFINED_BUFFER_ID),
                tricub_data_index, ( NS(buffer_size_t) )0u );
    }

    return assign_item;
}

/* end: sixtracklib/common/be_tricub/be_tricub.c */
