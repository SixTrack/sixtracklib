#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_drift/be_drift.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #if !defined( _GPUCODE )
    #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

NS(object_type_id_t) NS(Drift_type_id_ext)( void ) SIXTRL_NOEXCEPT
{
    return NS(Drift_type_id)();
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(arch_status_t) NS(Drift_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( offsets != SIXTRL_NULLPTR ) &&
        ( max_num_offsets > ( buf_size_t )0u ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, offsets, max_num_offsets, ZERO );
    }

    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

NS(arch_status_t) NS(Drift_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( sizes != SIXTRL_NULLPTR ) &&
        ( max_num_sizes > ( buf_size_t )0u ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, sizes, max_num_sizes, ZERO );
    }

    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

NS(arch_status_t) NS(Drift_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( counts != SIXTRL_NULLPTR ) &&
        ( max_num_counts > ( buf_size_t )0u ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, counts, max_num_counts, ZERO );
    }

    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Drift_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT requ_dataptrs ) SIXTRL_NOEXCEPT
{
    NS(Drift) temp;
    NS(arch_status_t) const status = NS(Drift_clear)( &temp );
    NS(buffer_size_t) const ndataptrs = NS(Drift_num_dataptrs)( &temp );

    return ( ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) &&
             ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
             ( NS(Buffer_can_add_object)( buffer, sizeof( NS(Drift) ),
                ndataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR, requ_objects,
                    requ_slots, requ_dataptrs ) ) );
}

SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* added_elem = SIXTRL_NULLPTR;

    NS(Drift) temp;
    NS(arch_status_t) const status = NS(Drift_clear)( &temp );
    NS(buffer_size_t) const ndataptrs = NS(Drift_num_dataptrs)( &temp );

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
        ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(Drift)* )( uintptr_t
            )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
                sizeof( NS(Drift) ), NS(Drift_type_id)(), ndataptrs,
                    SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_add)(
    SIXTRL_BE_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(drift_real_t) const length )
{
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* added_elem = SIXTRL_NULLPTR;

    NS(Drift) temp;
    NS(arch_status_t) status = NS(Drift_set_length)( &temp, length );
    NS(buffer_size_t) const ndataptrs = NS(Drift_num_dataptrs)( &temp );

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
        ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(Drift)* )( uintptr_t
            )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
                sizeof( NS(Drift) ), NS(Drift_type_id)(),
                    ndataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR,
                        SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT orig )
{
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* added_elem = SIXTRL_NULLPTR;
    NS(buffer_size_t) const ndataptrs = NS(Drift_num_dataptrs)( orig );

    if( ( buffer != SIXTRL_NULLPTR ) && ( orig != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0u ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(Drift)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, orig,
            sizeof( NS(Drift) ), NS(Drift_type_id)(), ndataptrs,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

/* ------------------------------------------------------------------------- */

NS(object_type_id_t) NS(DriftExact_type_id_ext)( void ) SIXTRL_NOEXCEPT
{
    return NS(DriftExact_type_id)();
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(arch_status_t) NS(DriftExact_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( offsets != SIXTRL_NULLPTR ) && ( drift != SIXTRL_NULLPTR ) &&
        ( max_num_offsets > ZERO ) && ( slot_size > ZERO ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, offsets, max_num_offsets, ZERO );
        status = NS(ARCH_STATUS_SUCCESS);
    }

    return status;
}

NS(arch_status_t) NS(DriftExact_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( sizes != SIXTRL_NULLPTR ) && ( drift != SIXTRL_NULLPTR ) &&
        ( max_num_sizes > ZERO ) && ( slot_size > ZERO ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, sizes, max_num_sizes, ZERO );
        status = NS(ARCH_STATUS_SUCCESS);
    }

    return status;
}

NS(arch_status_t) NS(DriftExact_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( counts != SIXTRL_NULLPTR ) && ( drift != SIXTRL_NULLPTR ) &&
        ( max_num_counts > ZERO ) && ( slot_size > ZERO ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, counts, max_num_counts, ZERO );
        status = NS(ARCH_STATUS_SUCCESS);
    }

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(DriftExact_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT requ_dataptrs ) SIXTRL_NOEXCEPT
{
    NS(DriftExact) temp;
    NS(arch_status_t) const status = NS(DriftExact_clear)( &temp );
    NS(buffer_size_t) const ndataptrs = NS(DriftExact_num_dataptrs)( &temp );
    SIXTRL_ASSERT( ndataptrs == ( NS(buffer_size_t) )0u );

    return ( ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) &&
             ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
             ( NS(Buffer_can_add_object)( buffer, sizeof( NS(DriftExact) ),
                ndataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR, requ_objects,
                    requ_slots, requ_dataptrs ) ) );
}

SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* NS(DriftExact_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* added_elem = SIXTRL_NULLPTR;

    NS(DriftExact) temp;
    NS(arch_status_t) const status = NS(DriftExact_clear)( &temp );
    NS(buffer_size_t) const ndataptrs = NS(DriftExact_num_dataptrs)( &temp );

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
        ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* )( uintptr_t
            )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
                sizeof( NS(DriftExact) ), NS(DriftExact_type_id)(), ndataptrs,
                    SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* NS(DriftExact_add)(
    SIXTRL_BE_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(drift_real_t) const length )
{
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* added_elem = SIXTRL_NULLPTR;

    NS(DriftExact) temp;
    NS(arch_status_t) status = NS(DriftExact_set_length)( &temp, length );
    NS(buffer_size_t) const ndataptrs = NS(DriftExact_num_dataptrs)( &temp );

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
        ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* )( uintptr_t
            )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
                sizeof( NS(DriftExact) ), NS(DriftExact_type_id)(),
                    ndataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR,
                        SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* NS(DriftExact_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT orig )
{
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* added_elem = SIXTRL_NULLPTR;
    NS(buffer_size_t) const ndataptrs = NS(DriftExact_num_dataptrs)( orig );

    if( ( buffer != SIXTRL_NULLPTR ) && ( orig != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0u ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, orig,
            sizeof( NS(DriftExact) ), NS(DriftExact_type_id)(), ndataptrs,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return added_elem;
}
