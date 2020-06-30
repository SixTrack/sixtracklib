#include "sixtracklib/common/be_rfmultipole/be_rfmultipole.h"
#include "sixtracklib/common/buffer.h"

NS(object_type_id_t) NS(RFMultipole_type_id_ext)( void ) SIXTRL_NOEXCEPT
{
    return NS(RFMultipole_type_id)();
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(arch_status_t) NS(RFMultipole_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);
    NS(buffer_size_t) const ndataptrs = NS(RFMultipole_num_dataptrs)( mpole );

    if( ( mpole != SIXTRL_NULLPTR ) && ( offsets != SIXTRL_NULLPTR ) &&
        ( slot_size > ( NS(buffer_size_t) )0u ) &&
        ( max_num_offsets >= ndataptrs ) )
    {
        SIXTRL_ASSERT( ndataptrs == ( NS(buffer_size_t) )2u );

        offsets[ 0 ] = offsetof( NS(RFMultipole), bal_addr );
        offsets[ 1 ] = offsetof( NS(RFMultipole), phase_addr );

        SIXTRL_ASSERT( offsets[ 0 ] % slot_size == ( NS(buffer_size_t) )0 );
        SIXTRL_ASSERT( offsets[ 1 ] % slot_size == ( NS(buffer_size_t) )0 );
        status = NS(ARCH_STATUS_SUCCESS);
    }

    return status;
}

NS(arch_status_t) NS(RFMultipole_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);
    NS(buffer_size_t) const ndataptrs = NS(RFMultipole_num_dataptrs)( mpole );

    if( ( mpole != SIXTRL_NULLPTR ) && ( sizes != SIXTRL_NULLPTR ) &&
        ( slot_size > ( NS(buffer_size_t) )0u ) &&
        ( max_num_sizes >= ndataptrs ) )
    {
        SIXTRL_ASSERT( ndataptrs == ( NS(buffer_size_t) )2u );
        sizes[ 0 ] = sizeof( NS(rf_multipole_real_t) );
        sizes[ 1 ] = sizeof( NS(rf_multipole_real_t) );

        status = NS(ARCH_STATUS_SUCCESS);
    }

    return status;
}

NS(arch_status_t) NS(RFMultipole_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);
    NS(buffer_size_t) const ndataptrs = NS(RFMultipole_num_dataptrs)( mpole );

    if( ( counts != SIXTRL_NULLPTR ) && ( max_num_counts >= ndataptrs ) &&
        ( slot_size > ( NS(buffer_size_t) )0 ) &&
        ( NS(RFMultipole_order)( mpole ) >= ( NS(rf_multipole_int_t) )0u ) )
    {
        SIXTRL_ASSERT( ndataptrs == ( NS(buffer_size_t) )2u );

        counts[ 0 ] = NS(RFMultipole_bal_length)( mpole );
        counts[ 1 ] = NS(RFMultipole_phase_length)( mpole );
        status = NS(ARCH_STATUS_SUCCESS);
    }

    return status;
}

/* ------------------------------------------------------------------------- */

bool NS(RFMultipole_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    typedef NS(buffer_size_t) buf_size_t;

    bool can_be_added = false;
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );
    buf_size_t ndataptrs = ( buf_size_t )0u;

    NS(RFMultipole) mpole;
    NS(arch_status_t) status = NS(RFMultipole_clear)( &mpole );
    status |= NS(RFMultipole_set_order)( &mpole, order );
    ndataptrs = NS(RFMultipole_num_dataptrs)( &mpole );

    if( ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) &&
        ( order >= ( NS(rf_multipole_int_t) )0 ) &&
        ( ndataptrs == ( buf_size_t )2u ) && ( buffer != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ARGPTR_DEC buf_size_t sizes[ 2 ];
        SIXTRL_ARGPTR_DEC buf_size_t counts[ 2 ];

        status = NS(RFMultipole_attributes_sizes)(
            &sizes[ 0 ], ( buf_size_t )2u, &mpole, slot_size );

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(RFMultipole_attributes_counts)(
                &counts[ 0 ], ( buf_size_t )2u, &mpole, slot_size );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            can_be_added = NS(Buffer_can_add_object)( buffer,
                sizeof( NS(RFMultipole) ), ndataptrs, sizes, counts,
                    ptr_requ_objects, ptr_requ_slots, ptr_requ_dataptrs );
        }
    }

    return can_be_added;
}

SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* NS(RFMultipole_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* added_elem = SIXTRL_NULLPTR;
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );
    buf_size_t ndataptrs = ( buf_size_t )0u;

    NS(RFMultipole) mpole;
    NS(arch_status_t) status = NS(RFMultipole_clear)( &mpole );
    status |= NS(RFMultipole_set_order)( &mpole, order );
    ndataptrs = NS(RFMultipole_num_dataptrs)( &mpole );

    if( ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) &&
        ( order >= ( NS(rf_multipole_int_t) )0 ) &&
        ( ndataptrs == ( buf_size_t )2u ) && ( buffer != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ARGPTR_DEC buf_size_t offsets[ 2 ];
        SIXTRL_ARGPTR_DEC buf_size_t sizes[ 2 ];
        SIXTRL_ARGPTR_DEC buf_size_t counts[ 2 ];

        status = NS(RFMultipole_attributes_offsets)(
            &offsets[ 0 ], ( buf_size_t )2u, &mpole, slot_size );

        if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
        {
            status = NS(RFMultipole_attributes_sizes)(
                &sizes[ 0 ], ( buf_size_t )2u, &mpole, slot_size );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(RFMultipole_attributes_counts)(
                &counts[ 0 ], ( buf_size_t )2u, &mpole, slot_size );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* )(
                uintptr_t )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
                    buffer, &mpole, sizeof( NS(RFMultipole) ),
                    NS(RFMultipole_type_id)(), ndataptrs, &offsets[ 0 ],
                        &sizes[ 0 ], &counts[ 0 ] ) );
        }
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* NS(RFMultipole_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order,
    NS(rf_multipole_real_t) const voltage,
    NS(rf_multipole_real_t) const frequency,
    NS(rf_multipole_real_t) const lag,
    NS(buffer_addr_t) const bal_addr, NS(buffer_addr_t) const phase_addr )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* added_elem = SIXTRL_NULLPTR;
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );
    buf_size_t ndataptrs = ( buf_size_t )0u;

    NS(RFMultipole) mpole;
    NS(arch_status_t) status = NS(RFMultipole_clear)( &mpole );
    status |= NS(RFMultipole_set_order)( &mpole, order );
    status |= NS(RFMultipole_set_voltage)( &mpole, voltage );
    status |= NS(RFMultipole_set_frequency)( &mpole, frequency );
    status |= NS(RFMultipole_set_lag)( &mpole, lag );
    status |= NS(RFMultipole_set_bal_addr)( &mpole, bal_addr );
    status |= NS(RFMultipole_set_phase_addr)( &mpole, phase_addr );
    ndataptrs = NS(RFMultipole_num_dataptrs)( &mpole );

    if( ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) &&
        ( order >= ( NS(rf_multipole_int_t) )0 ) &&
        ( ndataptrs == ( buf_size_t )2u ) && ( buffer != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ARGPTR_DEC buf_size_t offsets[ 2 ];
        SIXTRL_ARGPTR_DEC buf_size_t sizes[ 2 ];
        SIXTRL_ARGPTR_DEC buf_size_t counts[ 2 ];

        status = NS(RFMultipole_attributes_offsets)(
            &offsets[ 0 ], ( buf_size_t )2u, &mpole, slot_size );

        if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
        {
            status = NS(RFMultipole_attributes_sizes)(
                &sizes[ 0 ], ( buf_size_t )2u, &mpole, slot_size );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(RFMultipole_attributes_counts)(
                &counts[ 0 ], ( buf_size_t )2u, &mpole, slot_size );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* )(
                uintptr_t )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
                    buffer, &mpole, sizeof( NS(RFMultipole) ),
                    NS(RFMultipole_type_id)(), ndataptrs, &offsets[ 0 ],
                        &sizes[ 0 ], &counts[ 0 ] ) );
        }
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* NS(RFMultipole_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT orig )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* added_elem = SIXTRL_NULLPTR;
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );
    buf_size_t const ndataptrs = NS(RFMultipole_num_dataptrs)( orig );

    if( ( orig != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( buf_size_t )2u ) && ( buffer != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ARGPTR_DEC buf_size_t offsets[ 2 ];
        SIXTRL_ARGPTR_DEC buf_size_t sizes[ 2 ];
        SIXTRL_ARGPTR_DEC buf_size_t counts[ 2 ];

        NS(arch_status_t) status = NS(RFMultipole_attributes_offsets)(
            &offsets[ 0 ], ( buf_size_t )2u, orig, slot_size );

        if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
        {
            status = NS(RFMultipole_attributes_sizes)(
                &sizes[ 0 ], ( buf_size_t )2u, orig, slot_size );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(RFMultipole_attributes_counts)(
                &counts[ 0 ], ( buf_size_t )2u, orig, slot_size );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* )(
                uintptr_t )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
                    buffer, orig, sizeof( NS(RFMultipole) ),
                    NS(RFMultipole_type_id)(), ndataptrs, &offsets[ 0 ],
                        &sizes[ 0 ], &counts[ 0 ] ) );
        }
    }

    return added_elem;
}
