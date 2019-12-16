#include "sixtracklib/common/be_rfmultipole/be_rfmultipole.h"
#include "sixtracklib/common/buffer.h"

SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole) const* NS(RFMultiPole_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    return NS(RFMultiPole_const_from_obj_index)( NS(Buffer_get_const_object)(
        buffer, index ) );
}

SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* NS(RFMultiPole_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    return NS(RFMultiPole_from_obj_index)( NS(Buffer_get_object)(
        buffer, index ) );
}

NS(arch_status_t) NS(RFMultiPole_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const slot_size )
{
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( mpole != SIXTRL_NULLPTR ) &&
        ( offsets_begin != SIXTRL_NULLPTR ) &&
        ( slot_size > ( NS(buffer_size_t) )0u ) &&
        ( max_num_offsets >= NS(RFMultiPole_num_dataptrs)( mpole ) ) )
    {
        SIXTRL_ASSERT( NS(RFMultiPole_num_dataptrs)( mpole ) == 2u );

        offsets_begin[ 0 ] = offsetof( NS(RFMultiPole), bal_addr );
        offsets_begin[ 1 ] = offsetof( NS(RFMultiPole), phase_addr );

        status = NS(ARCH_STATUS_SUCCESS);
    }

    return status;
}

NS(arch_status_t) NS(RFMultiPole_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const slot_size )
{
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( mpole != SIXTRL_NULLPTR ) && ( sizes_begin != SIXTRL_NULLPTR ) &&
        ( slot_size > ( NS(buffer_size_t) )0u ) &&
        ( max_num_sizes >= NS(RFMultiPole_num_dataptrs)( mpole ) ) )
    {
        SIXTRL_ASSERT( NS(RFMultiPole_num_dataptrs)( mpole ) == 2u );

        sizes_begin[ 0 ] = sizeof( NS(rf_multipole_real_t) );
        sizes_begin[ 1 ] = sizeof( NS(rf_multipole_real_t) );

        status = NS(ARCH_STATUS_SUCCESS);
    }

    return status;
}

NS(arch_status_t) NS(RFMultiPole_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( counts_begin != SIXTRL_NULLPTR ) &&
        ( max_num_counts >= NS(RFMultiPole_num_dataptrs)( mpole ) ) &&
        ( NS(RFMultiPole_order)( mpole ) >= ( NS(rf_multipole_int_t) )0u ) )
    {
        SIXTRL_ASSERT( NS(RFMultiPole_num_dataptrs)( mpole ) == 2u );

        counts_begin[ 0 ] = ( NS(buffer_size_t)
            )NS(RFMultiPole_num_bal_elements)( mpole );

        counts_begin[ 1 ] = ( NS(buffer_size_t)
            )NS(RFMultiPole_num_phase_elements)( mpole );

        status = NS(ARCH_STATUS_SUCCESS);
    }

    return status;
}

/* ------------------------------------------------------------------------- */

bool NS(RFMultiPole_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    typedef NS(buffer_size_t) buf_size_t;

    bool can_be_added = false;
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    NS(RFMultiPole) mpole;
    buf_size_t const num_dataptrs = NS(RFMultiPole_num_dataptrs)( &mpole );
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    buf_size_t sizes[]  = { ( buf_size_t )0u, ( buf_size_t )0u };
    buf_size_t counts[] = { ( buf_size_t )0u, ( buf_size_t )0u };

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )2u );

    NS(RFMultiPole_preset)( &mpole );
    mpole.order = order;

    status = NS(RFMultiPole_attributes_sizes)(
        &sizes[ 0 ], num_dataptrs, &mpole, slot_size );

    if( status == NS(ARCH_STATUS_SUCCESS) )
    {
        status = NS(RFMultiPole_attributes_counts)(
            &counts[ 0 ], num_dataptrs, &mpole );
    }

    if( status == NS(ARCH_STATUS_SUCCESS) )
    {
        can_be_added = NS(Buffer_can_add_object)( buffer,
            sizeof( NS(RFMultiPole) ), num_dataptrs, sizes, counts,
                ptr_requ_objects, ptr_requ_slots, ptr_requ_dataptrs );
    }

    return can_be_added;
}

SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* NS(RFMultiPole_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* ptr_multipole = SIXTRL_NULLPTR;
    NS(arch_size_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    NS(RFMultiPole) multipole;
    buf_size_t const num_dataptrs = NS(RFMultiPole_num_dataptrs)( &multipole );

    buf_size_t offsets[] = { ( buf_size_t )0u, ( buf_size_t )0u };
    buf_size_t sizes[]   = { ( buf_size_t )0u, ( buf_size_t )0u };
    buf_size_t counts[]  = { ( buf_size_t )0u, ( buf_size_t )0u };
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )2u );

    NS(RFMultiPole_preset)( &multipole );
    multipole.order = order;

    status = NS(RFMultiPole_attributes_offsets)(
        &offsets[ 0 ], num_dataptrs, &multipole, slot_size );

    if( status == NS(ARCH_STATUS_SUCCESS) )
    {
        status = NS(RFMultiPole_attributes_counts)(
            &counts[ 0 ], num_dataptrs, &multipole );

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(RFMultiPole_attributes_sizes)(
                &sizes[ 0 ], num_dataptrs, &multipole, slot_size );
        }
    }

    if( status == NS(ARCH_STATUS_SUCCESS) )
    {
        ptr_multipole = ( SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* )(
            uintptr_t )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
                buffer, &multipole, sizeof( multipole ),
                    NS(RFMultiPole_type_id)( &multipole ), num_dataptrs,
                        &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
    }

    return ptr_multipole;
}

SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* NS(RFMultiPole_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order,
    NS(rf_multipole_real_t) const voltage,
    NS(rf_multipole_real_t) const frequency,
    NS(rf_multipole_real_t) const lag,
    SIXTRL_ARGPTR_DEC NS(rf_multipole_real_t) const* SIXTRL_RESTRICT bal_values,
    SIXTRL_ARGPTR_DEC NS(rf_multipole_real_t) const*
        SIXTRL_RESTRICT phase_values )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* ptr_multipole = SIXTRL_NULLPTR;
    NS(arch_size_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    NS(RFMultiPole) multipole;
    buf_size_t const num_dataptrs = NS(RFMultiPole_num_dataptrs)( &multipole );

    buf_size_t offsets[] = { ( buf_size_t )0u, ( buf_size_t )0u };
    buf_size_t sizes[]   = { ( buf_size_t )0u, ( buf_size_t )0u };
    buf_size_t counts[]  = { ( buf_size_t )0u, ( buf_size_t )0u };
    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )2u );

    NS(RFMultiPole_preset)( &multipole );
    multipole.order = order;
    multipole.voltage = voltage;
    multipole.frequency = frequency;
    multipole.lag = lag;
    multipole.bal_addr = ( NS(buffer_addr_t) )( uintptr_t )bal_values;
    multipole.phase_addr = ( NS(buffer_addr_t) )( uintptr_t )phase_values;

    status = NS(RFMultiPole_attributes_offsets)(
        &offsets[ 0 ], num_dataptrs, &multipole, slot_size );

    if( status == NS(ARCH_STATUS_SUCCESS) )
    {
        status = NS(RFMultiPole_attributes_counts)(
            &counts[ 0 ], num_dataptrs, &multipole );

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(RFMultiPole_attributes_sizes)(
                &sizes[ 0 ], num_dataptrs, &multipole, slot_size );
        }
    }

    if( status == NS(ARCH_STATUS_SUCCESS) )
    {
        ptr_multipole = ( SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* )(
            uintptr_t )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
                buffer, &multipole, sizeof( multipole ),
                    NS(RFMultiPole_type_id)( &multipole ), num_dataptrs,
                        &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
    }

    return ptr_multipole;
}

SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* NS(RFMultiPole_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return NS(RFMultiPole_add)( buffer,
        NS(RFMultiPole_order)( mpole ), NS(RFMultiPole_voltage)( mpole ),
        NS(RFMultiPole_frequency)( mpole ), NS(RFMultiPole_lag)( mpole ),
        NS(RFMultiPole_const_bal)( mpole ),
        NS(RFMultiPole_const_phase)( mpole ) );
}
