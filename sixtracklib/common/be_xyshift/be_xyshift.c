#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_xyshift/be_xyshift.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #if !defined( _GPUCODE )
    #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

NS(object_type_id_t) NS(XYShift_type_id_ext)( void ) SIXTRL_NOEXCEPT
{
    return NS(XYShift_type_id)();
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(arch_status_t) NS(XYShift_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xyshift,
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

NS(arch_status_t) NS(XYShift_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xyshift,
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

NS(arch_status_t) NS(XYShift_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xyshift,
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

bool NS(XYShift_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT requ_dataptrs ) SIXTRL_NOEXCEPT
{
    NS(XYShift) temp;
    NS(arch_status_t) const status = NS(XYShift_clear)( &temp );
    NS(buffer_size_t) const ndataptrs = NS(XYShift_num_dataptrs)( &temp );

    return ( ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) &&
             ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
             ( NS(Buffer_can_add_object)( buffer, sizeof( NS(XYShift) ),
                ndataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR, requ_objects,
                    requ_slots, requ_dataptrs ) ) );
}

SIXTRL_BE_ARGPTR_DEC NS(XYShift)* NS(XYShift_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* added_elem = SIXTRL_NULLPTR;

    NS(XYShift) temp;
    NS(arch_status_t) const status = NS(XYShift_clear)( &temp );
    NS(buffer_size_t) const ndataptrs = NS(XYShift_num_dataptrs)( &temp );

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
        ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(XYShift)* )( uintptr_t
            )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
                sizeof( NS(XYShift) ), NS(XYShift_type_id)(), ndataptrs,
                    SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(XYShift)* NS(XYShift_add)(
    SIXTRL_BE_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_REAL_T const dx, SIXTRL_REAL_T const dy )
{
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* added_elem = SIXTRL_NULLPTR;
    NS(buffer_size_t) ndataptrs = ( NS(buffer_size_t) )0u;

    NS(XYShift) temp;
    NS(arch_status_t) status = NS(XYShift_set_dx)( &temp, dx );
    status |= NS(XYShift_set_dy)( &temp, dy );
    ndataptrs = NS(XYShift_num_dataptrs)( &temp );

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
        ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(XYShift)* )( uintptr_t
            )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
                sizeof( NS(XYShift) ), NS(XYShift_type_id)(),
                    ndataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR,
                        SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(XYShift)* NS(XYShift_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT orig )
{
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* added_elem = SIXTRL_NULLPTR;
    NS(buffer_size_t) const ndataptrs = NS(XYShift_num_dataptrs)( orig );

    if( ( buffer != SIXTRL_NULLPTR ) && ( orig != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0u ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(XYShift)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, orig,
            sizeof( NS(XYShift) ), NS(XYShift_type_id)(), ndataptrs,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return added_elem;
}
