#ifndef SIXTRL_COMMON_BE_LIMIT_C99_H__
#define SIXTRL_COMMON_BE_LIMIT_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

typedef struct NS(Limit)
{
    NS(particle_real_t) x_limit     SIXTRL_ALIGN( 8 );
    NS(particle_real_t) y_limit     SIXTRL_ALIGN( 8 );
}
NS(Limit);

#if !defined( SIXTRL_DEFAULT_X_LIMIT )
    #define SIXTRL_DEFAULT_X_LIMIT 1.0
#endif /* !defined( SIXTRL_DEFAULT_X_LIMIT ) */

#if !defined( SIXTRL_DEFAULT_Y_LIMIT )
    #define SIXTRL_DEFAULT_Y_LIMIT 1.0
#endif /* !defined( SIXTRL_DEFAULT_Y_LIMIT ) */

#if !defined( _GPUCODE )

    SIXTRL_STATIC_VAR NS(particle_real_t) const NS(DEFAULT_X_LIMIT) = (
            NS(particle_real_t) )SIXTRL_DEFAULT_X_LIMIT;

    SIXTRL_STATIC_VAR NS(particle_real_t) const NS(DEFAULT_Y_LIMIT) = (
            NS(particle_real_t) )SIXTRL_DEFAULT_Y_LIMIT;

#endif /* !defined( _GPUCODE ) */


SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(Limit_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(Limit_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Limit)* NS(Limit_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(Limit)* SIXTRL_RESTRICT  );

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(Limit_get_x_limit)(
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT  );

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(Limit_get_y_limit)(
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT  );

SIXTRL_STATIC SIXTRL_FN void NS(Limit_set_x_limit)(
    SIXTRL_BE_ARGPTR_DEC NS(Limit)* SIXTRL_RESTRICT ,
    NS(particle_real_t) const x_limit );

SIXTRL_STATIC SIXTRL_FN void NS(Limit_set_y_limit)(
    SIXTRL_BE_ARGPTR_DEC NS(Limit)* SIXTRL_RESTRICT ,
    NS(particle_real_t) const y_limit );

SIXTRL_STATIC SIXTRL_FN void NS(Limit_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(Limit)* SIXTRL_RESTRICT  );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Limit_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(Limit)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT source );

SIXTRL_STATIC SIXTRL_FN int NS(Limit_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT rhs );

SIXTRL_STATIC SIXTRL_FN int NS(Limit_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(Limit_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT  );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(Limit_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC  const NS(Limit) *const SIXTRL_RESTRICT  );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Limit_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(Limit)* NS(Limit_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(Limit)* NS(Limit_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(particle_real_t) const x_limit, NS(particle_real_t) const y_limit );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(Limit)*
NS(Limit_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT limit );

#endif /* !defined( _GPUCODE )*/

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/*        Implementation of inline functions for NS(Limit)                   */
/* ========================================================================= */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(buffer_size_t)
NS(Limit_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0u );

    ( void )buffer;
    ( void )limit;
    ( void )slot_size;

    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(Limit_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0u );

    ( void )buffer;

    return ( limit != SIXTRL_NULLPTR )
        ? NS(ManagedBuffer_get_slot_based_length)( sizeof( *limit ), slot_size )
        : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Limit)* NS(Limit_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(Limit)* SIXTRL_RESTRICT limit )
{
    if( limit != SIXTRL_NULLPTR )
    {
        NS(Limit_set_x_limit)( limit, SIXTRL_DEFAULT_X_LIMIT );
        NS(Limit_set_y_limit)( limit, SIXTRL_DEFAULT_Y_LIMIT );
    }

    return limit;
}

SIXTRL_INLINE NS(particle_real_t) NS(Limit_get_x_limit)(
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT limit )
{
    return ( limit != SIXTRL_NULLPTR )
        ? limit->x_limit : SIXTRL_DEFAULT_X_LIMIT;
}

SIXTRL_INLINE NS(particle_real_t) NS(Limit_get_y_limit)(
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT limit )
{
    return ( limit != SIXTRL_NULLPTR )
        ? limit->y_limit : SIXTRL_DEFAULT_Y_LIMIT;
}

SIXTRL_INLINE void NS(Limit_set_x_limit)(
    SIXTRL_BE_ARGPTR_DEC NS(Limit)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const x_limit )
{
    if( ( limit != SIXTRL_NULLPTR ) && ( x_limit >= ( NS(particle_real_t) )0 ) )
    {
        limit->x_limit = x_limit;
    }

    return;
}

SIXTRL_INLINE void NS(Limit_set_y_limit)(
    SIXTRL_BE_ARGPTR_DEC NS(Limit)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const y_limit )
{
    if( ( limit != SIXTRL_NULLPTR ) && ( y_limit >= ( NS(particle_real_t) )0 ) )
    {
        limit->y_limit = y_limit;
    }

    return;
}

SIXTRL_INLINE void NS(Limit_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(Limit)* SIXTRL_RESTRICT limit )
{
    NS(Limit_preset)( limit );
}

SIXTRL_INLINE NS(arch_status_t) NS(Limit_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(Limit)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT source )
{
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( destination != SIXTRL_NULLPTR ) &&
        ( source != SIXTRL_NULLPTR ) && ( destination != source ) )
    {
        destination->x_limit = source->x_limit;
        destination->y_limit = source->y_limit;

        status = NS(ARCH_STATUS_SUCCESS);
    }
    else if( ( destination != SIXTRL_NULLPTR ) && ( destination == source ) )
    {
        status = NS(ARCH_STATUS_SUCCESS);
    }

    return status;
}

SIXTRL_INLINE int NS(Limit_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT rhs )
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) &&
        ( rhs != SIXTRL_NULLPTR ) )
    {
        NS(particle_real_t) const delta_x =
            NS(Limit_get_x_limit)( lhs ) - NS(Limit_get_x_limit)( rhs );

        cmp_result = 0;

        if( delta_x > ( NS(particle_real_t) )0.0 )
        {
            cmp_result = +1;
        }
        else if( delta_x < ( NS(particle_real_t) )0.0 )
        {
            cmp_result = -1;
        }

        if( cmp_result == 0 )
        {
            NS(particle_real_t) const delta_y =
                NS(Limit_get_y_limit)( lhs ) - NS(Limit_get_y_limit)( rhs );

            if( delta_y > ( NS(particle_real_t) )0.0 )
            {
                cmp_result = +1;
            }
            else if( delta_y < ( NS(particle_real_t) )0.0 )
            {
                cmp_result = -1;
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE int NS(Limit_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) &&
        ( rhs != SIXTRL_NULLPTR ) )
    {
        NS(particle_real_t) const delta_x =
            NS(Limit_get_x_limit)( lhs ) - NS(Limit_get_x_limit)( rhs );

        cmp_result = 0;

        if( delta_x > ( NS(particle_real_t) )0.0 )
        {
            cmp_result = +1;
        }
        else if( delta_x < ( NS(particle_real_t) )0.0 )
        {
            cmp_result = -1;
        }

        if( cmp_result == 0 )
        {
            NS(particle_real_t) const delta_y =
                NS(Limit_get_y_limit)( lhs ) - NS(Limit_get_y_limit)( rhs );

            if( delta_y > ( NS(particle_real_t) )0.0 )
            {
                cmp_result = +1;
            }
            else if( delta_y < ( NS(particle_real_t) )0.0 )
            {
                cmp_result = -1;
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRL_COMMON_BE_LIMIT_C99_H__ */
/*end: sixtracklib/common/be_/be_.h */
