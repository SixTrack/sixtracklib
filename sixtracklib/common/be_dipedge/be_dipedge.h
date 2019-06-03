#ifndef SIXTRL_COMMON_BE_DIPEDGE_BE_DIPEDGE_C99_H__
#define SIXTRL_COMMON_BE_DIPEDGE_BE_DIPEDGE_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/constants.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

typedef SIXTRL_REAL_T NS(dipedge_real_t);

typedef struct NS(DipoleEdge)
{
    NS(dipedge_real_t) inv_rho        SIXTRL_ALIGN( 8 );
    NS(dipedge_real_t) cos_rot_angle  SIXTRL_ALIGN( 8 );
    NS(dipedge_real_t) tan_rot_angle  SIXTRL_ALIGN( 8 );
    NS(dipedge_real_t) b              SIXTRL_ALIGN( 8 );
    NS(dipedge_real_t) cos_tilt_angle SIXTRL_ALIGN( 8 );
    NS(dipedge_real_t) sin_tilt_angle SIXTRL_ALIGN( 8 );
}
NS(DipoleEdge);

#if !defined( _GPUCODE )

SIXTRL_STATIC_VAR NS(dipedge_real_t) const NS(DIPOLE_EDGE_DEFAULT_INV_RHO) =
    ( NS(dipedge_real_t) )0;

SIXTRL_STATIC_VAR NS(dipedge_real_t) const NS(DIPOLE_EDGE_DEFAULT_B) =
    ( NS(dipedge_real_t) )0;

SIXTRL_STATIC_VAR NS(dipedge_real_t) const NS(DIPOLE_EDGE_ROT_ANGLE_DEG) =
    ( NS(dipedge_real_t) )0;

SIXTRL_STATIC_VAR NS(dipedge_real_t) const NS(DIPOLE_EDGE_TILT_ANGLE_DEG) =
    ( NS(dipedge_real_t) )0;

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(DipoleEdge_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(DipoleEdge_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)*
NS(DipoleEdge_preset)( SIXTRL_BE_ARGPTR_DEC
    NS(DipoleEdge)* SIXTRL_RESTRICT dipedge );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC void NS(DipoleEdge_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(dipedge_real_t) NS(DipoleEdge_get_rho)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

SIXTRL_STATIC SIXTRL_FN NS(dipedge_real_t) NS(DipoleEdge_get_inv_rho)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

SIXTRL_STATIC SIXTRL_FN NS(dipedge_real_t) NS(DipoleEdge_get_rot_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

SIXTRL_STATIC SIXTRL_FN NS(dipedge_real_t) NS(DipoleEdge_get_rot_angle_rad)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

SIXTRL_STATIC SIXTRL_FN NS(dipedge_real_t) NS(DipoleEdge_get_cos_rot_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

SIXTRL_STATIC SIXTRL_FN NS(dipedge_real_t) NS(DipoleEdge_get_sin_rot_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

SIXTRL_STATIC SIXTRL_FN NS(dipedge_real_t) NS(DipoleEdge_get_tan_rot_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

SIXTRL_STATIC SIXTRL_FN NS(dipedge_real_t) NS(DipoleEdge_get_tilt_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

SIXTRL_STATIC SIXTRL_FN NS(dipedge_real_t) NS(DipoleEdge_get_tilt_angle_rad)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

SIXTRL_STATIC SIXTRL_FN NS(dipedge_real_t) NS(DipoleEdge_get_cos_tilt_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

SIXTRL_STATIC SIXTRL_FN NS(dipedge_real_t) NS(DipoleEdge_get_sin_tilt_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

SIXTRL_STATIC SIXTRL_FN NS(dipedge_real_t) NS(DipoleEdge_get_b)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN void NS(DipoleEdge_set_inv_rho)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const inv_rho  );

SIXTRL_STATIC SIXTRL_FN void NS(DipoleEdge_set_rot_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const rot_angle_deg );

SIXTRL_STATIC SIXTRL_FN void NS(DipoleEdge_set_rot_angle_rad)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const rot_angle_rad  );

SIXTRL_STATIC SIXTRL_FN void NS(DipoleEdge_set_cos_rot_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const cos_rot_angle );

SIXTRL_STATIC SIXTRL_FN void NS(DipoleEdge_set_tan_rot_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const sin_rot_angle  );

SIXTRL_STATIC SIXTRL_FN void NS(DipoleEdge_set_tilt_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const tilt_angle_deg );

SIXTRL_STATIC SIXTRL_FN void NS(DipoleEdge_set_tilt_angle_rad)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const tilt_angle_rad  );

SIXTRL_STATIC SIXTRL_FN void NS(DipoleEdge_set_cos_tilt_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const cos_tilt_angle );

SIXTRL_STATIC SIXTRL_FN void NS(DipoleEdge_set_sin_tilt_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const sin_tilt_angle );

SIXTRL_STATIC SIXTRL_FN void NS(DipoleEdge_set_b)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const b  );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(DipoleEdge_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT source );

SIXTRL_STATIC SIXTRL_FN int NS(DipoleEdge_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT rhs );

SIXTRL_STATIC SIXTRL_FN int NS(DipoleEdge_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(DipoleEdge_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(DipoleEdge_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC  const NS(DipoleEdge) *const SIXTRL_RESTRICT  );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(DipoleEdge_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(DipoleEdge)* NS(DipoleEdge_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(DipoleEdge)* NS(DipoleEdge_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(dipedge_real_t) const inv_rho,
    NS(dipedge_real_t) const rot_angle_deg,
    NS(dipedge_real_t) const b,
    NS(dipedge_real_t) const tilt_angle_deg );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(DipoleEdge)*
NS(DipoleEdge_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

#endif /* !defined( _GPUCODE )*/

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/*        Implementation of inline functions for NS(DipoleEdge)                   */
/* ========================================================================= */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(buffer_size_t)
NS(DipoleEdge_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0u );

    ( void )buffer;
    ( void )dipedge;
    ( void )slot_size;

    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(DipoleEdge_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0u );

    ( void )buffer;

    return ( dipedge != SIXTRL_NULLPTR )
        ? NS(ManagedBuffer_get_slot_based_length)(
            sizeof( *dipedge ), slot_size )
        : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* NS(DipoleEdge_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge )
{
    if( dipedge != SIXTRL_NULLPTR )
    {
        NS(DipoleEdge_clear)( dipedge );
        NS(DipoleEdge_set_b)( dipedge, ( NS(dipedge_real_t) )0.0 );
    }

    return dipedge;
}

SIXTRL_INLINE void NS(DipoleEdge_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge )
{
    SIXTRL_STATIC_VAR NS(dipedge_real_t) const ZERO = ( NS(dipedge_real_t) )0;
    SIXTRL_STATIC_VAR NS(dipedge_real_t) const ONE  = ( NS(dipedge_real_t) )1;

    NS(DipoleEdge_set_inv_rho)( dipedge, ZERO );
    NS(DipoleEdge_set_cos_rot_angle)( dipedge, ONE );
    NS(DipoleEdge_set_tan_rot_angle)( dipedge, ZERO );
    NS(DipoleEdge_set_cos_tilt_angle)( dipedge, ONE );
    NS(DipoleEdge_set_sin_tilt_angle)( dipedge, ZERO );

    return;
}

/* ------------------------------------------------------------------------- */


SIXTRL_INLINE NS(dipedge_real_t) NS(DipoleEdge_get_rho)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    SIXTRL_ASSERT( dipedge != SIXTRL_NULLPTR );
    return ( dipedge->inv_rho > ( NS(dipedge_real_t) )0 )
        ? ( NS(dipedge_real_t) )1.0 / dipedge->inv_rho
        : ( NS(dipedge_real_t) )0.0;
}

SIXTRL_INLINE NS(dipedge_real_t) NS(DipoleEdge_get_inv_rho)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    SIXTRL_ASSERT( dipedge != SIXTRL_NULLPTR );
    return dipedge->inv_rho;
}

SIXTRL_INLINE NS(dipedge_real_t) NS(DipoleEdge_get_rot_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    return SIXTRL_RAD2DEG * NS(DipoleEdge_get_rot_angle_rad)( dipedge );
}

SIXTRL_INLINE NS(dipedge_real_t) NS(DipoleEdge_get_rot_angle_rad)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    return atan2( NS(DipoleEdge_get_cos_rot_angle)( dipedge ),
        NS(DipoleEdge_get_sin_rot_angle)( dipedge ) );
}

SIXTRL_INLINE NS(dipedge_real_t) NS(DipoleEdge_get_cos_rot_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    SIXTRL_ASSERT( dipedge != SIXTRL_NULLPTR );
    return dipedge->cos_rot_angle;
}

SIXTRL_INLINE NS(dipedge_real_t) NS(DipoleEdge_get_sin_rot_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    return NS(DipoleEdge_get_cos_rot_angle)( dipedge ) *
           NS(DipoleEdge_get_tan_rot_angle)( dipedge );
}

SIXTRL_INLINE NS(dipedge_real_t) NS(DipoleEdge_get_tan_rot_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    SIXTRL_ASSERT( dipedge != SIXTRL_NULLPTR );
    return dipedge->tan_rot_angle;
}

SIXTRL_INLINE NS(dipedge_real_t) NS(DipoleEdge_get_tilt_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    return SIXTRL_RAD2DEG * NS(DipoleEdge_get_tilt_angle_rad)( dipedge );
}

SIXTRL_INLINE NS(dipedge_real_t) NS(DipoleEdge_get_tilt_angle_rad)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    return atan2( NS(DipoleEdge_get_cos_tilt_angle)( dipedge ),
                  NS(DipoleEdge_get_sin_tilt_angle)( dipedge ) );
}

SIXTRL_INLINE NS(dipedge_real_t) NS(DipoleEdge_get_cos_tilt_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    SIXTRL_ASSERT( dipedge != SIXTRL_NULLPTR );
    return dipedge->cos_tilt_angle;
}

SIXTRL_INLINE NS(dipedge_real_t) NS(DipoleEdge_get_sin_tilt_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    SIXTRL_ASSERT( dipedge != SIXTRL_NULLPTR );
    return dipedge->sin_tilt_angle;
}

SIXTRL_INLINE NS(dipedge_real_t) NS(DipoleEdge_get_b)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    SIXTRL_ASSERT( dipedge != SIXTRL_NULLPTR );
    return dipedge->b;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(DipoleEdge_set_inv_rho)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const inv_rho  )
{
    if( ( dipedge != SIXTRL_NULLPTR ) &&
        ( inv_rho >= ( NS(dipedge_real_t) )0 ) )
    {
        dipedge->inv_rho = inv_rho;
    }
}

SIXTRL_INLINE void NS(DipoleEdge_set_rot_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const rot_angle_deg )
{
    NS(DipoleEdge_set_rot_angle_rad)(
        dipedge, SIXTRL_DEG2RAD * rot_angle_deg );
}

SIXTRL_INLINE void NS(DipoleEdge_set_rot_angle_rad)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const rot_angle_rad  )
{
    if( dipedge != SIXTRL_NULLPTR )
    {
        dipedge->cos_rot_angle = cos( rot_angle_rad );
        dipedge->tan_rot_angle = tan( rot_angle_rad );
    }
}

SIXTRL_INLINE void NS(DipoleEdge_set_cos_rot_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const cos_rot_angle )
{
    if( ( dipedge != SIXTRL_NULLPTR ) &&
        ( cos_rot_angle >= ( NS(dipedge_real_t) )0 ) &&
        ( cos_rot_angle <= ( NS(dipedge_real_t) )1 ) )
    {
        dipedge->cos_rot_angle = cos_rot_angle;
    }
}

SIXTRL_INLINE void NS(DipoleEdge_set_tan_rot_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const tan_rot_angle  )
{
    if( ( dipedge != SIXTRL_NULLPTR ) &&
        ( tan_rot_angle >= ( NS(dipedge_real_t) )0 ) &&
        ( tan_rot_angle <= ( NS(dipedge_real_t) )1 ) )
    {
        dipedge->tan_rot_angle = tan_rot_angle;
    }
}

SIXTRL_INLINE void NS(DipoleEdge_set_tilt_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const tilt_angle_deg )
{
    NS(DipoleEdge_set_tilt_angle_rad)(
        dipedge, SIXTRL_DEG2RAD * tilt_angle_deg );
}

SIXTRL_INLINE void NS(DipoleEdge_set_tilt_angle_rad)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const tilt_angle_rad  )
{
    if( dipedge != SIXTRL_NULLPTR )
    {
        dipedge->cos_tilt_angle = cos( tilt_angle_rad );
        dipedge->sin_tilt_angle = sin( tilt_angle_rad );
    }
}

SIXTRL_INLINE void NS(DipoleEdge_set_cos_tilt_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const cos_tilt_angle )
{
    if( ( dipedge != SIXTRL_NULLPTR ) &&
        ( cos_tilt_angle >= ( NS(dipedge_real_t) )0 ) &&
        ( cos_tilt_angle <= ( NS(dipedge_real_t) )1 ) )
    {
        dipedge->cos_tilt_angle = cos_tilt_angle;
    }
}

SIXTRL_INLINE void NS(DipoleEdge_set_sin_tilt_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const sin_tilt_angle )
{
    if( ( dipedge != SIXTRL_NULLPTR ) &&
        ( sin_tilt_angle >= ( NS(dipedge_real_t) )0 ) &&
        ( sin_tilt_angle <= ( NS(dipedge_real_t) )1 ) )
    {
        dipedge->sin_tilt_angle = sin_tilt_angle;
    }
}

SIXTRL_INLINE void NS(DipoleEdge_set_b)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const b  )
{
    if( dipedge != SIXTRL_NULLPTR )
    {
        dipedge->b = b;
    }
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(arch_status_t) NS(DipoleEdge_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT source )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( destination != SIXTRL_NULLPTR ) &&
        ( source != SIXTRL_NULLPTR ) )
    {
        NS(DipoleEdge_set_inv_rho)( destination,
            NS(DipoleEdge_get_inv_rho)( source ) );

        NS(DipoleEdge_set_cos_rot_angle)( destination,
            NS(DipoleEdge_get_cos_rot_angle)( source ) );

        NS(DipoleEdge_set_tan_rot_angle)( destination,
            NS(DipoleEdge_get_tan_rot_angle)( source ) );

        NS(DipoleEdge_set_b)( destination, NS(DipoleEdge_get_b)( source ) );

        NS(DipoleEdge_set_cos_tilt_angle)( destination,
            NS(DipoleEdge_get_cos_tilt_angle)( source ) );

        NS(DipoleEdge_set_sin_tilt_angle)( destination,
            NS(DipoleEdge_get_sin_tilt_angle)( source ) );

        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE int NS(DipoleEdge_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT rhs )
{
    return NS(DipoleEdge_compare_values_with_treshold)(
        lhs, rhs, ( NS(dipedge_real_t) )0 );
}

SIXTRL_INLINE int NS(DipoleEdge_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        SIXTRL_STATIC_VAR NS(dipedge_real_t) const ZERO =
            ( NS(dipedge_real_t) )0;

        NS(dipedge_real_t) delta = NS(DipoleEdge_get_inv_rho)( lhs ) -
            NS(DipoleEdge_get_inv_rho)( rhs );

        NS(dipedge_real_t) const minus_treshold = -treshold;

        if( ( delta == ZERO ) ||
            ( ( delta > ZERO ) && ( delta < treshold ) ) ||
            ( ( delta < ZERO ) && ( delta > minus_treshold ) ) )
        {
            cmp_result = 0;
        }
        else
        {
            cmp_result = ( delta > 0 ) ? +1 : -1;
        }

        if( cmp_result == 0 )
        {
            delta = NS(DipoleEdge_get_cos_rot_angle)( lhs ) -
                NS(DipoleEdge_get_cos_rot_angle)( rhs );

            if( delta > treshold ) cmp_result = +1;
            else if( delta < minus_treshold ) cmp_result = -1;
        }

        if( cmp_result == 0 )
        {
            delta = NS(DipoleEdge_get_tan_rot_angle)( lhs ) -
                NS(DipoleEdge_get_tan_rot_angle)( rhs );

            if( delta > treshold ) cmp_result = +1;
            else if( delta < minus_treshold ) cmp_result = -1;
        }

        if( cmp_result == 0 )
        {
            delta = NS(DipoleEdge_get_b)( lhs ) - NS(DipoleEdge_get_b)( rhs );
            if( delta > treshold ) cmp_result = +1;
            else if( delta < minus_treshold ) cmp_result = -1;
        }

        if( cmp_result == 0 )
        {
            delta = NS(DipoleEdge_get_cos_tilt_angle)( lhs ) -
                NS(DipoleEdge_get_cos_tilt_angle)( rhs );

            if( delta > treshold ) cmp_result = +1;
            else if( delta < minus_treshold ) cmp_result = -1;
        }

        if( cmp_result == 0 )
        {
            delta = NS(DipoleEdge_get_sin_tilt_angle)( lhs ) -
                NS(DipoleEdge_get_sin_tilt_angle)( rhs );

            if( delta > treshold ) cmp_result = +1;
            else if( delta < minus_treshold ) cmp_result = -1;
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

#endif /* SIXTRL_COMMON_BE_DIPEDGE_BE_DIPEDGE_C99_H__ */
/*end: sixtracklib/common/be_/be_.h */
