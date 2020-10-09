#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_cavity/be_cavity.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


NS(object_type_id_t) NS(Cavity_type_id_ext)( void ) SIXTRL_NOEXCEPT
{
    return NS(Cavity_type_id)();
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(arch_status_t) NS(Cavity_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( cavity ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( offsets != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) &&
        ( max_num_offsets > ( buf_size_t )0u ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, offsets, max_num_offsets, ZERO );
    }

    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

NS(arch_status_t) NS(Cavity_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( cavity ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( sizes != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) &&
        ( max_num_sizes > ( buf_size_t )0u ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, sizes, max_num_sizes, ZERO );
    }

    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

NS(arch_status_t) NS(Cavity_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( cavity ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( counts != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) &&
        ( max_num_counts > ( buf_size_t )0u ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, counts, max_num_counts, ZERO );
    }

    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Cavity_can_be_added)(
    SIXTRL_BE_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BE_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BE_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT requ_dataptrs ) SIXTRL_NOEXCEPT
{
    NS(Cavity) temp;
    NS(arch_status_t) const status = NS(Cavity_clear)( &temp );
    NS(buffer_size_t) const ndataptrs = NS(Cavity_num_dataptrs)( &temp );

    return ( ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) &&
             ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
             ( NS(Buffer_can_add_object)( buffer, sizeof( NS(Cavity) ),
                ndataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR, requ_objects,
                    requ_slots, requ_dataptrs ) ) );
}

SIXTRL_BE_ARGPTR_DEC NS(Cavity)* NS(Cavity_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* added_elem = SIXTRL_NULLPTR;

    NS(Cavity) temp;
    NS(arch_status_t) const status = NS(Cavity_clear)( &temp );
    NS(buffer_size_t) const ndataptrs = NS(Cavity_num_dataptrs)( &temp );

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
        ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(Cavity)* )( uintptr_t
            )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
                sizeof( NS(Cavity) ), NS(Cavity_type_id)(), ndataptrs,
                    SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(Cavity)* NS(Cavity_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_REAL_T  const voltage,
    SIXTRL_REAL_T  const frequency,
    SIXTRL_REAL_T  const lag )
{
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* added_elem = SIXTRL_NULLPTR;
    NS(buffer_size_t) ndataptrs = ( NS(buffer_size_t) )0;

    NS(Cavity) temp;
    NS(arch_status_t) status = NS(Cavity_set_voltage)( &temp, voltage );
    status |= NS(Cavity_set_frequency)( &temp, frequency );
    status |= NS(Cavity_set_lag)( &temp, lag );
    ndataptrs = NS(Cavity_num_dataptrs)( &temp );

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
        ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(Cavity)* )( uintptr_t
            )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
                sizeof( NS(Cavity) ), NS(Cavity_type_id)(),
                    ndataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR,
                        SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(Cavity)* NS(Cavity_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT orig )
{
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* added_elem = SIXTRL_NULLPTR;
    NS(buffer_size_t) const ndataptrs = NS(Cavity_num_dataptrs)( orig );

    if( ( buffer != SIXTRL_NULLPTR ) && ( orig != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0u ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(Cavity)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, orig,
            sizeof( NS(Cavity) ), NS(Cavity_type_id)(), ndataptrs,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return added_elem;
}
