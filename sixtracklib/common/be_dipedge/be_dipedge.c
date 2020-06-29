#include "sixtracklib/common/be_dipedge/be_dipedge.h"

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/constants.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

NS(arch_status_t) NS(DipoleEdge_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( dipedge ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( offsets != SIXTRL_NULLPTR ) &&
        ( max_num_offsets > ZERO ) && ( slot_size > ZERO ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, offsets, max_num_offsets, ZERO );
    }

    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

NS(arch_status_t) NS(DipoleEdge_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( dipedge ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( sizes != SIXTRL_NULLPTR ) &&
        ( max_num_sizes > ZERO ) && ( slot_size > ZERO ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, sizes, max_num_sizes, ZERO );
    }

    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

NS(arch_status_t) NS(DipoleEdge_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( dipedge ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( counts != SIXTRL_NULLPTR ) &&
        ( max_num_counts > ZERO ) && ( slot_size > ZERO ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, counts, max_num_counts, ZERO );
    }

    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(object_type_id_t) NS(DipoleEdge_type_id_ext)( void ) SIXTRL_NOEXCEPT
{
    return NS(DipoleEdge_type_id)();
}

bool NS(DipoleEdge_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        requ_dataptrs ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t)  buf_size_t;
    buf_size_t const ndataptrs = NS(DipoleEdge_num_dataptrs)( SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ndataptrs == ( buf_size_t )0u );

    return NS(Buffer_can_add_object)( buffer, sizeof( NS(DipoleEdge) ),
        ndataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR, requ_objects, requ_slots,
            requ_dataptrs );
}

SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* NS(DipoleEdge_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* added_elem = SIXTRL_NULLPTR;

    NS(DipoleEdge) temp;
    NS(arch_status_t) status = NS(DipoleEdge_clear)( &temp );
    NS(buffer_size_t) const ndataptrs = NS(DipoleEdge_num_dataptrs)( &temp );

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0 ) )
    {
        added_elem = (SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
            sizeof( NS(DipoleEdge) ), NS(DipoleEdge_type_id)(), ndataptrs,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* NS(DipoleEdge_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(dipedge_real_t) const r21, NS(dipedge_real_t) const r43 )
{
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* added_elem = SIXTRL_NULLPTR;
    NS(buffer_size_t) ndataptrs = ( NS(buffer_size_t) )0;

    NS(DipoleEdge) temp;
    NS(arch_status_t) status = NS(DipoleEdge_set_r21)( &temp, r21 );
    status |= NS(DipoleEdge_set_r43)( &temp, r43 );
    ndataptrs = NS(DipoleEdge_num_dataptrs)( &temp );

    if( ( buffer != SIXTRL_NULLPTR ) && ( status == NS(ARCH_STATUS_SUCCESS) ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0 ) )
    {
        added_elem = (SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
            sizeof( NS(DipoleEdge) ), NS(DipoleEdge_type_id)(), ndataptrs,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* NS(DipoleEdge_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT orig )
{
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* added_elem = SIXTRL_NULLPTR;
    NS(buffer_size_t) const ndataptrs = NS(DipoleEdge_num_dataptrs)( orig );

    if( ( buffer != SIXTRL_NULLPTR ) && ( orig != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0 ) )
    {
        added_elem = (SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, orig,
            sizeof( NS(DipoleEdge) ), NS(DipoleEdge_type_id)(), ndataptrs,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return added_elem;
}
