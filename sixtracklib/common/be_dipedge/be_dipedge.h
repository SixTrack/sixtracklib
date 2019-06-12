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
#endif /* !defined( SIXTGRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

typedef SIXTRL_REAL_T NS(dipedge_real_t);

typedef struct NS(DipoleEdge)
{
    NS(dipedge_real_t) r21 SIXTRL_ALIGN( 8 );
    NS(dipedge_real_t) r43 SIXTRL_ALIGN( 8 );
}
NS(DipoleEdge);

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

SIXTRL_STATIC SIXTRL_FN void NS(DipoleEdge_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(dipedge_real_t) NS(DipoleEdge_get_r21)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

SIXTRL_STATIC SIXTRL_FN NS(dipedge_real_t) NS(DipoleEdge_get_r43)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN void NS(DipoleEdge_set_r21)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const r21  );

SIXTRL_STATIC SIXTRL_FN void NS(DipoleEdge_set_r43)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const r43 );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(DipoleEdge_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT source );

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
    NS(dipedge_real_t) const r21, NS(dipedge_real_t) const r43 );

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
    }

    return dipedge;
}

SIXTRL_INLINE void NS(DipoleEdge_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge )
{
    SIXTRL_STATIC_VAR NS(dipedge_real_t) const ZERO = ( NS(dipedge_real_t) )0;
    SIXTRL_STATIC_VAR NS(dipedge_real_t) const ONE  = ( NS(dipedge_real_t) )1;

    NS(DipoleEdge_set_r21)( dipedge, ZERO );
    NS(DipoleEdge_set_r43)( dipedge, ONE );

    return;
}

/* ------------------------------------------------------------------------- */


SIXTRL_INLINE NS(dipedge_real_t) NS(DipoleEdge_get_r21)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    SIXTRL_ASSERT( dipedge != SIXTRL_NULLPTR );
    return dipedge->r21;
}

SIXTRL_INLINE NS(dipedge_real_t) NS(DipoleEdge_get_r43)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    SIXTRL_ASSERT( dipedge != SIXTRL_NULLPTR );
    return dipedge->r43;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(DipoleEdge_set_r21)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const r21  )
{
    if( dipedge != SIXTRL_NULLPTR )
    {
        dipedge->r21 = r21;
    }
}

SIXTRL_INLINE void NS(DipoleEdge_set_r43)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const r43 )
{
    if( dipedge != SIXTRL_NULLPTR )
    {
        dipedge->r43 = r43;
    }
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(arch_status_t) NS(DipoleEdge_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dest,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT source )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( dest != SIXTRL_NULLPTR ) && ( source != SIXTRL_NULLPTR ) )
    {
        if( dest != source )
        {
            NS(DipoleEdge_set_r21)( dest, NS(DipoleEdge_get_r21)( source ) );
            NS(DipoleEdge_set_r43)( dest, NS(DipoleEdge_get_r43)( source ) );
        }

        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRL_COMMON_BE_DIPEDGE_BE_DIPEDGE_C99_H__ */
/*end: sixtracklib/common/be_/be_.h */
