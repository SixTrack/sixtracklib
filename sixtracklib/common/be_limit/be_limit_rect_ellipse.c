#if !defined( SIXTRL_NO_INCLUDES )
#include "sixtracklib/common/be_limit/be_limit_rect_ellipse.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
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
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse) const*
NS(LimitRectEllipse_const_from_buffer)( SIXTRL_BUFFER_ARGPTR_DEC const
    NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(LimitRectEllipse_const_from_obj_index)(
        NS(Buffer_get_const_object)( buffer, index ) );
}

SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
NS(LimitRectEllipse_from_buffer)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
    SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(LimitRectEllipse_from_obj_index)(
        NS(Buffer_get_object)( buffer, index ) );
}

/* ------------------------------------------------------------------------- */

NS(arch_status_t) NS(LimitRectEllipse_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( offsets_begin != SIXTRL_NULLPTR ) && ( limit != SIXTRL_NULLPTR ) &&
        ( slot_size > ZERO ) )
    {
        status = NS(ARCH_STATUS_SUCCESS);

        if( max_num_offsets > ZERO )
        {
            SIXTRACKLIB_SET_VALUES(
                buf_size_t, offsets_begin, max_num_offsets, ZERO );
        }
    }

    return status;
}

NS(arch_status_t) NS(LimitRectEllipse_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( sizes_begin != SIXTRL_NULLPTR ) && ( limit != SIXTRL_NULLPTR ) &&
        ( slot_size > ZERO ) )
    {
        status = NS(ARCH_STATUS_SUCCESS);

        if( max_num_sizes > ZERO )
        {
            SIXTRACKLIB_SET_VALUES(
                buf_size_t, sizes_begin, max_num_sizes, ZERO );
        }
    }

    return status;
}

NS(arch_status_t) NS(LimitRectEllipse_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( counts_begin != SIXTRL_NULLPTR ) && ( limit != SIXTRL_NULLPTR ) )
    {
        status = NS(ARCH_STATUS_SUCCESS);

        if( max_num_counts > ZERO )
        {
            SIXTRACKLIB_SET_VALUES(
                buf_size_t, counts_begin, max_num_counts, ZERO );
        }
    }

    return status;
}

/* ------------------------------------------------------------------------- */

bool NS(LimitRectEllipse_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_num_objs,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_num_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_num_dataptrs
) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    bool can_be_added = false;

    buf_size_t num_dataptrs = ( buf_size_t )0u;
    NS(LimitRectEllipse) limit;
    NS(LimitRectEllipse_preset)( &limit );

    num_dataptrs = NS(LimitRectEllipse_num_dataptrs)( &limit );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    can_be_added = NS(Buffer_can_add_object)( buffer,
        sizeof( NS(LimitRectEllipse) ), num_dataptrs,
            SIXTRL_NULLPTR, SIXTRL_NULLPTR, ptr_requ_num_objs,
                ptr_requ_num_slots, ptr_requ_num_dataptrs );

    return can_be_added;
}

SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
NS(LimitRectEllipse_new)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
    SIXTRL_RESTRICT buffer )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRectEllipse)* ptr_limit = SIXTRL_NULLPTR;

    buf_size_t num_dataptrs = ( buf_size_t )0u;

    NS(LimitRectEllipse) limit;
    NS(LimitRectEllipse_preset)( &limit );

    num_dataptrs = NS(LimitRectEllipse_num_dataptrs)( &limit );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

    ptr_limit = ( SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRectEllipse)* )(
        uintptr_t )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
            buffer, &limit, sizeof( limit ), NS(LimitRectEllipse_type_id)(
                &limit ), num_dataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR,
                    SIXTRL_NULLPTR ) );

    return ptr_limit;
}

SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* NS(LimitRectEllipse_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(particle_real_t) const max_x, NS(particle_real_t) const max_y,
    NS(particle_real_t) const a_squ, NS(particle_real_t) const b_squ )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRectEllipse)* ptr_limit = SIXTRL_NULLPTR;

    buf_size_t num_dataptrs = ( buf_size_t )0u;
    NS(LimitRectEllipse) limit;
    NS(LimitRectEllipse_preset)( &limit );
    NS(LimitRectEllipse_set_max_x)( &limit, max_x );
    NS(LimitRectEllipse_set_max_y)( &limit, max_y );
    NS(LimitRectEllipse_set_a_squ)( &limit, a_squ );
    NS(LimitRectEllipse_set_b_squ)( &limit, b_squ );
    NS(LimitRectEllipse_set_a_squ_b_squ)( &limit, a_squ * b_squ );

    num_dataptrs = NS(LimitRectEllipse_num_dataptrs)( &limit );
    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

    ptr_limit = ( SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRectEllipse)* )(
        uintptr_t )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
            buffer, &limit, sizeof( limit ),
                NS(LimitRectEllipse_type_id)( &limit ), num_dataptrs,
                    SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );

    return ptr_limit;
}

SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
NS(LimitRectEllipse_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT orig )
{
    return NS(LimitRectEllipse_add)( buffer,
        NS(LimitRectEllipse_max_x)( orig ), NS(LimitRectEllipse_max_y)( orig ),
        NS(LimitRectEllipse_a_squ)( orig ),
        NS(LimitRectEllipse_b_squ)( orig ) );
}
