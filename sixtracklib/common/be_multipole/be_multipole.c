#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_multipole/be_multipole.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

NS(arch_status_t) NS(Multipole_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( offsets != SIXTRL_NULLPTR ) && ( multipole != SIXTRL_NULLPTR ) &&
        ( slot_size > ( NS(buffer_size_t) )0 ) &&
        ( max_num_offsets > ( NS(buffer_size_t) )0 ) )
    {
        offsets[ 0 ] = ( NS(buffer_size_t) )offsetof( NS(Multipole), bal_addr );
        SIXTRL_ASSERT( offsets[ 0 ] % slot_size == ( NS(buffer_size_t) )0 );

        if( max_num_offsets > ( NS(buffer_size_t) )1 )
        {
            NS(buffer_size_t) ii = ( NS(buffer_size_t) )1u;
            for( ; ii < max_num_offsets ; ++ii ) offsets[ ii ] = 0u;
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

NS(arch_status_t) NS(Multipole_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( sizes != SIXTRL_NULLPTR ) && ( multipole != SIXTRL_NULLPTR ) &&
        ( slot_size > ( NS(buffer_size_t) )0 ) &&
        ( max_num_sizes > ( NS(buffer_size_t) )0 ) )
    {
        sizes[ 0 ] = ( NS(buffer_size_t) )sizeof( NS(multipole_real_t) );
        if( max_num_sizes > ( NS(buffer_size_t) )1 )
        {
            NS(buffer_size_t) ii = ( NS(buffer_size_t) )1u;
            for( ; ii < max_num_sizes ; ++ii ) sizes[ ii ] = 0u;
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

NS(arch_status_t) NS(Multipole_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( counts != SIXTRL_NULLPTR ) && ( multipole != SIXTRL_NULLPTR ) &&
        ( slot_size > ( NS(buffer_size_t) )0 ) &&
        ( max_num_counts > ( NS(buffer_size_t) )0 ) )
    {
        counts[ 0 ] = NS(Multipole_bal_length)( multipole );
        if( max_num_counts > ( NS(buffer_size_t) )1 )
        {
            NS(buffer_size_t) ii = ( NS(buffer_size_t) )1u;
            for( ; ii < max_num_counts ; ++ii ) counts[ ii ] = 0u;
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(object_type_id_t) NS(Multipole_type_id_ext)( void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(Multipole_type_id)();
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Multipole_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(multipole_order_t) const order,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objs,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    bool can_be_added = false;

    buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );
    buf_size_t ndataptrs = ( buf_size_t )0u;

    NS(Multipole) data;
    NS(arch_status_t) status = NS(Multipole_clear)( &data );
    status |= NS(Multipole_set_order)( &data, order );
    ndataptrs = NS(Multipole_num_dataptrs)( &data );

    if( ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) &&
        ( ndataptrs == ( buf_size_t )1u ) && ( slot_size > ( buf_size_t )0 ) &&
        ( buffer != SIXTRL_NULLPTR ) && ( order >= ( NS(multipole_order_t) )0 ) )
    {
        NS(buffer_size_t) sizes[ 1 ];
        NS(buffer_size_t) counts[ 1 ];

        status = NS(Multipole_attributes_sizes)(
            &sizes[ 0 ], ( buf_size_t )1u, &data, slot_size );

        if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
        {
            status = NS(Multipole_attributes_counts)(
                &counts[ 0 ], ( buf_size_t )1u, &data, slot_size );
        }

        if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
        {
            can_be_added = NS(Buffer_can_add_object)( buffer, sizeof(
                NS(Multipole) ), ndataptrs, &sizes[ 0 ], &counts[ 0 ],
                    ptr_requ_objs, ptr_requ_slots, ptr_requ_dataptrs );
        }
    }

    return can_be_added;
}

SIXTRL_BE_ARGPTR_DEC NS(Multipole)* NS(Multipole_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(multipole_order_t) const order )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_BUFFER_DATAPTR_DEC NS(Multipole)*
        added_elem = SIXTRL_NULLPTR;

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( order >= ( NS(multipole_order_t) )0 ) )
    {
        buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

        SIXTRL_ARGPTR_DEC buf_size_t sizes[ 1u ];
        SIXTRL_ARGPTR_DEC buf_size_t counts[ 1u ];
        SIXTRL_ARGPTR_DEC buf_size_t offsets[ 1u ];

        buf_size_t ndataptrs = ( buf_size_t )0u;

        NS(Multipole) data;
        NS(arch_status_t) status = NS(Multipole_clear)( &data );
        status |= NS(Multipole_set_order)( &data, order );
        ndataptrs = NS(Multipole_num_dataptrs)( &data );

        if( ( ndataptrs == ( buf_size_t )1u ) &&
            ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) &&
            ( slot_size > ( buf_size_t )0 ) && ( buffer != SIXTRL_NULLPTR ) )
        {
            status = NS(Multipole_attributes_offsets)(
                &offsets[ 0 ], ( buf_size_t )1u, &data, slot_size );

            if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
            {
                status = NS(Multipole_attributes_sizes)(
                    &sizes[ 0 ], ( buf_size_t )1u, &data, slot_size );
            }

            if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
            {
                status = NS(Multipole_attributes_counts)(
                    &counts[ 0 ], ( buf_size_t )1u, &data, slot_size );
            }
        }

        if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
        {
            added_elem = ( SIXTRL_BUFFER_DATAPTR_DEC NS(Multipole)*
                )( uintptr_t )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
                    buffer, &data, sizeof( NS(Multipole) ),
                        NS(Multipole_type_id)(), ndataptrs,
                            &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
        }
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(Multipole)* NS(Multipole_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(multipole_order_t) const order,
    NS(multipole_real_t) const length,
    NS(multipole_real_t) const hxl,
    NS(multipole_real_t) const hyl,
    NS(buffer_addr_t) const bal_addr )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_BUFFER_DATAPTR_DEC NS(Multipole)*
        added_elem = SIXTRL_NULLPTR;

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( order >= ( NS(multipole_order_t) )0 ) )
    {
        buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );

        SIXTRL_ARGPTR_DEC buf_size_t sizes[ 1u ];
        SIXTRL_ARGPTR_DEC buf_size_t counts[ 1u ];
        SIXTRL_ARGPTR_DEC buf_size_t offsets[ 1u ];

        buf_size_t ndataptrs = ( buf_size_t )0u;

        NS(Multipole) data;
        NS(arch_status_t) status = NS(Multipole_clear)( &data );
        status |= NS(Multipole_set_order)( &data, order );
        status |= NS(Multipole_set_length)( &data, length );
        status |= NS(Multipole_set_hxl)( &data, hxl );
        status |= NS(Multipole_set_hyl)( &data, hyl );
        status |= NS(Multipole_set_bal_addr)( &data, bal_addr );

        ndataptrs = NS(Multipole_num_dataptrs)( &data );

        if( ( ndataptrs == ( buf_size_t )1u ) &&
            ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) &&
            ( slot_size > ( buf_size_t )0 ) && ( buffer != SIXTRL_NULLPTR ) )
        {
            status = NS(Multipole_attributes_offsets)(
                &offsets[ 0 ], ( buf_size_t )1u, &data, slot_size );

            if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
            {
                status = NS(Multipole_attributes_sizes)(
                    &sizes[ 0 ], ( buf_size_t )1u, &data, slot_size );
            }

            if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
            {
                status = NS(Multipole_attributes_counts)(
                    &counts[ 0 ], ( buf_size_t )1u, &data, slot_size );
            }
        }

        if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
        {
            added_elem = ( SIXTRL_BUFFER_DATAPTR_DEC NS(Multipole)*
                )( uintptr_t )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
                    buffer, &data, sizeof( NS(Multipole) ),
                        NS(Multipole_type_id)(), ndataptrs,
                            &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
        }
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(Multipole)* NS(Multipole_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT orig )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_BUFFER_DATAPTR_DEC NS(Multipole)*
        added_elem = SIXTRL_NULLPTR;

    if( ( buffer != SIXTRL_NULLPTR ) && ( orig != SIXTRL_NULLPTR ) )
    {
        buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );
        NS(arch_status_t) status = ( NS(arch_status_t)
            )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

        SIXTRL_ARGPTR_DEC buf_size_t sizes[ 1u ];
        SIXTRL_ARGPTR_DEC buf_size_t counts[ 1u ];
        SIXTRL_ARGPTR_DEC buf_size_t offsets[ 1u ];

        buf_size_t ndataptrs = NS(Multipole_num_dataptrs)( orig );

        if( ( ndataptrs == ( buf_size_t )1u ) &&
            ( slot_size > ( buf_size_t )0 ) && ( buffer != SIXTRL_NULLPTR ) )
        {
            status = NS(Multipole_attributes_offsets)(
                &offsets[ 0 ], ( buf_size_t )1u, orig, slot_size );

            if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
            {
                status = NS(Multipole_attributes_sizes)(
                    &sizes[ 0 ], ( buf_size_t )1u, orig, slot_size );
            }

            if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
            {
                status = NS(Multipole_attributes_counts)(
                    &counts[ 0 ], ( buf_size_t )1u, orig, slot_size );
            }
        }

        if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
        {
            added_elem = ( SIXTRL_BUFFER_DATAPTR_DEC NS(Multipole)*
                )( uintptr_t )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
                    buffer, orig, sizeof( NS(Multipole) ),
                        NS(Multipole_type_id)(), ndataptrs,
                            &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
        }
    }

    return added_elem;
}
