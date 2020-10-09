#include "sixtracklib/common/be_tricub/be_tricub.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/internal/objects_type_id.h"
#include "sixtracklib/common/buffer/assign_address_item.h"
#include "sixtracklib/common/buffer.h"

NS(object_type_id_t) NS(TriCubData_type_id_ext)( void ) SIXTRL_NOEXCEPT
{
    return NS(TriCubData_type_id)();
}

NS(arch_size_t) NS(TriCubData_ptr_offset_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(TriCubData) *const SIXTRL_RESTRICT d ) SIXTRL_NOEXCEPT
{
    return NS(TriCubData_ptr_offset)( d );
}

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

NS(arch_status_t) NS(TriCubData_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    buf_size_t const num_dataptrs = NS(TriCubData_num_dataptrs)( data );

    if( ( offsets != SIXTRL_NULLPTR ) && ( max_num_offsets >= num_dataptrs ) &&
        ( num_dataptrs == ( buf_size_t )1u ) &&
        ( slot_size > ( buf_size_t )0 ) )
    {
        buf_size_t ii = ( buf_size_t )1u;
        offsets[ 0 ] = NS(TriCubData_ptr_offset)( data );
        SIXTRL_ASSERT( offsets[ 0 ] % slot_size == ( buf_size_t )0 );

        for( ; ii < max_num_offsets ; ++ii ) offsets[ ii ] = ( buf_size_t )0u;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

NS(arch_status_t) NS(TriCubData_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    buf_size_t const num_dataptrs = NS(TriCubData_num_dataptrs)( data );

    if( ( counts != SIXTRL_NULLPTR ) && ( max_num_counts >= num_dataptrs ) &&
        ( num_dataptrs == ( buf_size_t )1u ) &&
        ( slot_size > ( buf_size_t )0 ) )
    {
        buf_size_t ii = ( buf_size_t )1u;
        counts[ 0 ] = NS(TriCubData_table_size)( data );

        for( ; ii < max_num_counts ; ++ii ) counts[ ii ] = ( buf_size_t )0u;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

NS(arch_status_t) NS(TriCubData_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    buf_size_t const num_dataptrs = NS(TriCubData_num_dataptrs)( data );

    if( ( sizes != SIXTRL_NULLPTR ) && ( max_num_sizes >= num_dataptrs ) &&
        ( num_dataptrs == ( buf_size_t )1u ) &&
        ( slot_size > ( buf_size_t )0 ) )
    {
        buf_size_t ii = ( buf_size_t )1u;
        sizes[ 0 ] = sizeof( NS(be_tricub_real_t) );

        for( ; ii < num_dataptrs ; ++ii ) sizes[ ii ] = ( buf_size_t )0u;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

bool NS(TriCubData_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(be_tricub_int_t) const nx, NS(be_tricub_int_t) const ny,
    NS(be_tricub_int_t) const nz,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_requ_dataptrs ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;

    bool can_be_added = false;
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    buf_size_t num_dataptrs = ( buf_size_t )0u;
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    NS(TriCubData) tricub_data;
    NS(TriCubData_preset)( &tricub_data );
    status  = NS(TriCubData_set_nx)( &tricub_data, nx );
    status |= NS(TriCubData_set_ny)( &tricub_data, ny );
    status |= NS(TriCubData_set_nz)( &tricub_data, nz );
    num_dataptrs = NS(TriCubData_num_dataptrs)( &tricub_data );

    if( ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) &&
        ( num_dataptrs == ( buf_size_t )1u ) &&
        ( slot_size > ( buf_size_t )0u ) && ( buffer != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ARGPTR_DEC buf_size_t sizes[  1 ];
        SIXTRL_ARGPTR_DEC buf_size_t counts[ 1 ];

        status = NS(TriCubData_attributes_sizes)(
            &sizes[ 0 ], num_dataptrs, &tricub_data, slot_size );

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(TriCubData_attributes_counts)(
                &counts[ 0 ], num_dataptrs, &tricub_data, slot_size );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            can_be_added = NS(Buffer_can_add_object)( buffer,
                sizeof( NS(TriCubData) ), num_dataptrs, sizes, counts,
                    ptr_requ_objects, ptr_requ_slots, ptr_requ_dataptrs );
        }
    }

    return can_be_added;
}

SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* NS(TriCubData_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_tricub_int_t) const nx, NS(be_tricub_int_t) const ny,
    NS(be_tricub_int_t) const nz )
{
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* added_elem = SIXTRL_NULLPTR;
    typedef NS(buffer_size_t) buf_size_t;

    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    buf_size_t num_dataptrs = ( buf_size_t )0u;
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    NS(TriCubData) tricub_data;
    NS(TriCubData_preset)( &tricub_data );
    status  = NS(TriCubData_set_nx)( &tricub_data, nx );
    status |= NS(TriCubData_set_ny)( &tricub_data, ny );
    status |= NS(TriCubData_set_nz)( &tricub_data, nz );
    num_dataptrs = NS(TriCubData_num_dataptrs)( &tricub_data );

    if( ( status == NS(ARCH_STATUS_SUCCESS) ) && ( buffer != SIXTRL_NULLPTR ) &&
        ( num_dataptrs == ( buf_size_t )1u ) &&
        ( slot_size > ( buf_size_t )0u ) )
    {
        SIXTRL_ARGPTR_DEC buf_size_t offsets[ 1 ];
        SIXTRL_ARGPTR_DEC buf_size_t sizes[  1 ];
        SIXTRL_ARGPTR_DEC buf_size_t counts[ 1 ];

        status = NS(TriCubData_attributes_sizes)(
            &sizes[ 0 ], num_dataptrs, &tricub_data, slot_size );

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(TriCubData_attributes_counts)(
                &counts[ 0 ], num_dataptrs, &tricub_data, slot_size );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(TriCubData_attributes_offsets)(
                &offsets[ 0 ], num_dataptrs, &tricub_data, slot_size );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            added_elem = ( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* )(
                uintptr_t )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
                    buffer, &tricub_data, sizeof( tricub_data ),
                        NS(TriCubData_type_id)(), num_dataptrs,
                            &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
        }
    }

    return added_elem;
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
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* added_elem = SIXTRL_NULLPTR;
    typedef NS(buffer_size_t) buf_size_t;

    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    buf_size_t num_dataptrs = ( buf_size_t )0u;
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    NS(TriCubData) tricub_data;
    NS(TriCubData_preset)( &tricub_data );
    status  = NS(TriCubData_set_x0)( &tricub_data, x0 );
    status |= NS(TriCubData_set_dx)( &tricub_data, dx );
    status |= NS(TriCubData_set_nx)( &tricub_data, nx );
    status |= NS(TriCubData_set_y0)( &tricub_data, y0 );
    status |= NS(TriCubData_set_dy)( &tricub_data, dy );
    status |= NS(TriCubData_set_ny)( &tricub_data, ny );
    status |= NS(TriCubData_set_z0)( &tricub_data, z0 );
    status |= NS(TriCubData_set_dz)( &tricub_data, dz );
    status |= NS(TriCubData_set_nz)( &tricub_data, nz );
    status |= NS(TriCubData_set_mirror_x)( &tricub_data, mirror_x );
    status |= NS(TriCubData_set_mirror_y)( &tricub_data, mirror_y );
    status |= NS(TriCubData_set_mirror_z)( &tricub_data, mirror_z );
    status |= NS(TriCubData_set_table_addr)( &tricub_data, table_addr );
    num_dataptrs = NS(TriCubData_num_dataptrs)( &tricub_data );

    if( ( status == NS(ARCH_STATUS_SUCCESS) ) && ( buffer != SIXTRL_NULLPTR ) &&
        ( num_dataptrs == ( buf_size_t )1u ) &&
        ( slot_size > ( buf_size_t )0u ) )
    {
        SIXTRL_ARGPTR_DEC buf_size_t offsets[ 1 ];
        SIXTRL_ARGPTR_DEC buf_size_t sizes[  1 ];
        SIXTRL_ARGPTR_DEC buf_size_t counts[ 1 ];

        status = NS(TriCubData_attributes_sizes)(
            &sizes[ 0 ], num_dataptrs, &tricub_data, slot_size );

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(TriCubData_attributes_counts)(
                &counts[ 0 ], num_dataptrs, &tricub_data, slot_size );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(TriCubData_attributes_offsets)(
                &offsets[ 0 ], num_dataptrs, &tricub_data, slot_size );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            added_elem = ( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* )(
                uintptr_t )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
                    buffer, &tricub_data, sizeof( tricub_data ),
                        NS(TriCubData_type_id)(), num_dataptrs,
                            &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
        }
    }

    return added_elem;
}

SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* NS(TriCubData_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT orig )
{
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* added_elem = SIXTRL_NULLPTR;
    typedef NS(buffer_size_t) buf_size_t;

    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    buf_size_t num_dataptrs = ( buf_size_t )0u;
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    if( ( status == NS(ARCH_STATUS_SUCCESS) ) && ( buffer != SIXTRL_NULLPTR ) &&
        ( num_dataptrs == ( buf_size_t )1u ) && ( orig != SIXTRL_NULLPTR ) &&
        ( slot_size > ( buf_size_t )0u ) )
    {
        SIXTRL_ARGPTR_DEC buf_size_t offsets[ 1 ];
        SIXTRL_ARGPTR_DEC buf_size_t sizes[  1 ];
        SIXTRL_ARGPTR_DEC buf_size_t counts[ 1 ];

        status = NS(TriCubData_attributes_sizes)(
            &sizes[ 0 ], num_dataptrs, orig, slot_size );

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(TriCubData_attributes_counts)(
                &counts[ 0 ], num_dataptrs, orig, slot_size );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(TriCubData_attributes_offsets)(
                &offsets[ 0 ], num_dataptrs, orig, slot_size );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            added_elem = ( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* )(
                uintptr_t )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
                    buffer, orig, sizeof( NS(TriCubData) ),
                        NS(TriCubData_type_id)(), num_dataptrs,
                            &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
        }
    }

    return added_elem;
}

/* ========================================================================= */

NS(object_type_id_t) NS(TriCub_type_id_ext)( void ) SIXTRL_NOEXCEPT
{
    return NS(TriCub_type_id)();
}

NS(arch_size_t) NS(TriCub_data_addr_offset_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(TriCub) *const SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT
{
    return NS(TriCub_data_addr_offset)( tricub );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(arch_status_t) NS(TriCub_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( tricub ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( offsets != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) &&
        ( max_num_offsets > ZERO ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, offsets, max_num_offsets, ZERO );
    }

    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

NS(arch_status_t) NS(TriCub_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub)
        *const SIXTRL_RESTRICT SIXTRL_UNUSED( tricub ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( sizes != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) &&
        ( max_num_sizes > ZERO ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, sizes, max_num_sizes, ZERO );
    }

    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

NS(arch_status_t) NS(TriCub_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub)
        *const SIXTRL_RESTRICT SIXTRL_UNUSED( tricub ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( counts != SIXTRL_NULLPTR ) && ( slot_size > ( buf_size_t )0 ) &&
        ( max_num_counts > ZERO ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, counts, max_num_counts, ZERO );
    }

    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(TriCub_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    ( void )ptr_requ_dataptrs;
    return NS(Buffer_can_add_trivial_object)( buffer, sizeof( NS(TriCub) ),
        ptr_requ_objects, ptr_requ_slots );
}

SIXTRL_BE_ARGPTR_DEC NS(TriCub)* NS(TriCub_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    NS(TriCub) tricub;
    NS(TriCub_preset)( &tricub );

    SIXTRL_ASSERT( ( NS(buffer_size_t) )0u ==
        NS(TriCub_num_dataptrs)( &tricub ) );

    return ( SIXTRL_BE_ARGPTR_DEC NS(TriCub)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_trivial_object)( buffer,
            &tricub, sizeof( NS(TriCub) ), NS(TriCub_type_id)() ) );
}

SIXTRL_BE_ARGPTR_DEC NS(TriCub)* NS(TriCub_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_tricub_real_t) const x_shift, NS(be_tricub_real_t) const y_shift,
    NS(be_tricub_real_t) const tau_shift,
    NS(be_tricub_real_t) const dipolar_kick_px,
    NS(be_tricub_real_t) const dipolar_kick_py,
    NS(be_tricub_real_t) const dipolar_kick_ptau,
    NS(be_tricub_real_t) const length,
    NS(buffer_addr_t) const data_addr )
{
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* added_elem = SIXTRL_NULLPTR;

    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);
    NS(TriCub) tricub;
    NS(TriCub_preset)( &tricub );
    status  = NS(TriCub_set_x_shift)( &tricub, x_shift );
    status |= NS(TriCub_set_y_shift)( &tricub, y_shift );
    status |= NS(TriCub_set_tau_shift)( &tricub, tau_shift );
    status |= NS(TriCub_set_dipolar_kick_px)( &tricub, dipolar_kick_px );
    status |= NS(TriCub_set_dipolar_kick_py)( &tricub, dipolar_kick_py );
    status |= NS(TriCub_set_dipolar_kick_ptau)( &tricub, dipolar_kick_ptau );
    status |= NS(TriCub_set_length)( &tricub, length );
    status |= NS(TriCub_set_data_addr)( &tricub, data_addr );

    if( ( status == NS(ARCH_STATUS_SUCCESS) ) &&
        ( NS(buffer_size_t) )0u == NS(TriCub_num_dataptrs)( &tricub ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(TriCub)* )( uintptr_t
            )NS(Object_get_begin_addr)( NS(Buffer_add_trivial_object)( buffer,
                &tricub, sizeof( NS(TriCub) ), NS(TriCub_type_id)() ) );
    }

    return added_elem;
}

SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)* NS(TriCub_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT orig )
{
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* added_elem = SIXTRL_NULLPTR;

    if( ( orig != SIXTRL_NULLPTR ) && ( buffer != SIXTRL_NULLPTR ) &&
        ( NS(buffer_size_t) )0u == NS(TriCub_num_dataptrs)( orig ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(TriCub)* )( uintptr_t
            )NS(Object_get_begin_addr)( NS(Buffer_add_trivial_object)( buffer,
                orig, sizeof( NS(TriCub) ), NS(TriCub_type_id)() ) );
    }

    return added_elem;
}

/* ------------------------------------------------------------------------ */

SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
NS(TriCub_buffer_create_assign_address_item)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT assign_items_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const belements_index,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT tricub_data_buffer,
    NS(buffer_size_t) const tricub_data_buffer_id,
    NS(buffer_size_t) const tricub_data_index )
{
    SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
        assign_item = SIXTRL_NULLPTR;

    SIXTRL_BE_ARGPTR_DEC NS(TriCub) const* tricub =
        NS(TriCub_const_from_buffer)( belements, belements_index );

    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const* tricub_data =
        NS(TriCubData_const_from_buffer)(
            tricub_data_buffer, tricub_data_index );

    if( ( assign_items_buffer != SIXTRL_NULLPTR ) &&
        ( tricub != SIXTRL_NULLPTR ) && ( tricub_data != SIXTRL_NULLPTR ) &&
        ( tricub_data_buffer_id != NS(ARCH_ILLEGAL_BUFFER_ID) ) )
    {
        assign_item = NS(AssignAddressItem_add)( assign_items_buffer,
            NS(OBJECT_TYPE_TRICUB), NS(ARCH_BEAM_ELEMENTS_BUFFER_ID),
                belements_index, NS(TriCub_data_addr_offset)( tricub ),
            NS(OBJECT_TYPE_TRICUB_DATA), tricub_data_buffer_id,
                tricub_data_index, NS(TriCubData_ptr_offset)( tricub_data ) );
    }

    return assign_item;
}

/* end: sixtracklib/common/be_tricub/be_tricub.c */
