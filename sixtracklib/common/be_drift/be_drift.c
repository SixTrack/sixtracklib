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

bool NS(Drift_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT requ_dataptrs ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) const ndataptrs = NS(Drift_num_dataptrs)( SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ndataptrs == ( NS(buffer_size_t) )0u );

    return NS(Buffer_can_add_object)( buffer, sizeof( NS(Drift) ), ndataptrs,
        SIXTRL_NULLPTR, SIXTRL_NULLPTR, requ_objects, requ_slots,
            requ_dataptrs );
}

SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    NS(buffer_size_t) const ndataptrs = NS(Drift_num_dataptrs)( SIXTRL_NULLPTR );

    NS(Drift) temp;
    NS(Drift_preset)( &temp );
    NS(Drift_set_length)( &temp, ( NS(drift_real_t) )0 );
    SIXTRL_ASSERT( ndataptrs == ( NS(buffer_size_t) )0u );

    return ( SIXTRL_BE_ARGPTR_DEC NS(Drift)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
            sizeof( NS(Drift) ), NS(Drift_type_id)(), ndataptrs,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
}

SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_add)(
    SIXTRL_BE_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(drift_real_t) const length )
{
    NS(buffer_size_t) const ndataptrs = NS(Drift_num_dataptrs)( SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ndataptrs == ( NS(buffer_size_t) )0 );

    NS(Drift) temp;
    NS(Drift_set_length)( &temp, length );

    SIXTRL_ASSERT( ndataptrs == ( NS(buffer_size_t) )0u );
    return ( SIXTRL_BE_ARGPTR_DEC NS(Drift)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
            sizeof( NS(Drift) ), NS(Drift_type_id)(), ndataptrs,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
}

SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT orig )
{
    NS(buffer_size_t) const ndataptrs = NS(Drift_num_dataptrs)( orig );
    SIXTRL_ASSERT( ndataptrs == ( NS(buffer_size_t) )0 );

    return ( SIXTRL_BE_ARGPTR_DEC NS(Drift)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, orig,
            sizeof( NS(Drift) ), NS(Drift_type_id)(), ndataptrs,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
}

/* ------------------------------------------------------------------------- */

NS(object_type_id_t) NS(DriftExact_type_id_ext)( void ) SIXTRL_NOEXCEPT
{
    return NS(DriftExact_type_id)();
}

bool NS(DriftExact_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT requ_dataptrs ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    buf_size_t const ndataptrs = NS(DriftExact_num_dataptrs)( SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ndataptrs == ( buf_size_t )0u );

    return NS(Buffer_can_add_object)( buffer, sizeof( NS(DriftExact) ),
        ndataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR,
            requ_objects, requ_slots, requ_dataptrs );
}

SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* NS(DriftExact_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(buffer_size_t) buf_size_t;
    buf_size_t const ndataptrs = NS(DriftExact_num_dataptrs)( SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ndataptrs == ( buf_size_t )0 );

    NS(DriftExact) temp;
    NS(DriftExact_preset)( &temp );

    return ( SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
            sizeof( NS(DriftExact) ), NS(DriftExact_type_id)(), ndataptrs,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
}

SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* NS(DriftExact_add)(
    SIXTRL_BE_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(drift_real_t) const length )
{
    typedef NS(buffer_size_t) buf_size_t;
    buf_size_t const ndataptrs = NS(DriftExact_num_dataptrs)( SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ndataptrs == ( buf_size_t )0 );

    NS(DriftExact) temp;
    NS(DriftExact_set_length)( &temp, length );

    return ( SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
            sizeof( NS(DriftExact) ), NS(DriftExact_type_id)(), ndataptrs,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
}

SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* NS(DriftExact_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT orig )
{
    typedef NS(buffer_size_t) buf_size_t;
    buf_size_t const ndataptrs = NS(DriftExact_num_dataptrs)( orig );
    SIXTRL_ASSERT( ndataptrs == ( buf_size_t )0 );

    return ( SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, orig,
            sizeof( NS(DriftExact) ), NS(DriftExact_type_id)(), ndataptrs,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
}
