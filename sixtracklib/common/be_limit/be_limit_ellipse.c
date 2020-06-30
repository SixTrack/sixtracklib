#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_limit/be_limit_ellipse.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_limit/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE )

NS(object_type_id_t) NS(LimitEllipse_type_id_ext)( void ) SIXTRL_NOEXCEPT
{
    return NS(LimitEllipse_type_id)();
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(arch_status_t) NS(LimitEllipse_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( limit ),
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

NS(arch_status_t) NS(LimitEllipse_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( limit ),
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

NS(arch_status_t) NS(LimitEllipse_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( limit ),
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

bool NS(LimitEllipse_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT requ_dataptrs ) SIXTRL_NOEXCEPT
{
    NS(LimitEllipse) temp;
    NS(arch_status_t) const status = NS(LimitEllipse_clear)( &temp );
    NS(buffer_size_t) const ndataptrs = NS(LimitEllipse_num_dataptrs)( &temp );
    SIXTRL_ASSERT( ndataptrs == ( NS(buffer_size_t) )0u );

    return ( ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) &&
             ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
             ( NS(Buffer_can_add_object)( buffer, sizeof( NS(LimitEllipse) ),
                ndataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR, requ_objects,
                    requ_slots, requ_dataptrs ) ) );
}

SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* NS(LimitEllipse_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* added_elem = SIXTRL_NULLPTR;

    NS(LimitEllipse) temp;
    NS(arch_status_t) const status = NS(LimitEllipse_clear)( &temp );
    NS(buffer_size_t) const ndataptrs = NS(LimitEllipse_num_dataptrs)( &temp );

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
        ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* )( uintptr_t
            )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
                sizeof( NS(LimitEllipse) ), NS(LimitEllipse_type_id)(), ndataptrs,
                    SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* NS(LimitEllipse_add)(
    SIXTRL_BE_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(particle_real_t) const x_semi_axis,
    NS(particle_real_t) const y_semi_axis )
{
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* added_elem = SIXTRL_NULLPTR;

    NS(LimitEllipse) temp;
    NS(arch_status_t) status = NS(LimitEllipse_set_half_axes)(
        &temp, x_semi_axis, y_semi_axis );
    NS(buffer_size_t) const ndataptrs = NS(LimitEllipse_num_dataptrs)( &temp );

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
        ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* )( uintptr_t
            )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
                sizeof( NS(LimitEllipse) ), NS(LimitEllipse_type_id)(),
                    ndataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR,
                        SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* NS(LimitEllipse_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT orig )
{
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* added_elem = SIXTRL_NULLPTR;
    NS(buffer_size_t) const ndataptrs = NS(LimitEllipse_num_dataptrs)( orig );

    if( ( buffer != SIXTRL_NULLPTR ) && ( orig != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0u ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, orig,
            sizeof( NS(LimitEllipse) ), NS(LimitEllipse_type_id)(), ndataptrs,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

#endif /* !defined( _GPUCODE )*/

/* end: sixtracklib/common/be_limit/be_limit_ellipse.c */
