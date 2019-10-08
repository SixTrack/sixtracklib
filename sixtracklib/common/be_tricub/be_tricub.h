#ifndef SIXTRACKLIB_COMMON_BE_TRICUB_C99_H__
#define SIXTRACKLIB_COMMON_BE_TRICUB_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */

#if !defined( SIXTRL_BE_TRICUBMAP_MATRIX_DIM )
    #define   SIXTRL_BE_TRICUBMAP_MATRIX_DIM 8
#endif /* !defined( SIXTRL_BE_TRICUBMAP_MATRIX_DIM ) */

typedef SIXTRL_INT64_T NS(be_tricub_int_t);
typedef SIXTRL_REAL_T  NS(be_tricub_real_t);

typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T*
        NS(be_tricub_ptr_real_t);

typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const*
        NS(be_tricub_ptr_const_real_t);

typedef struct NS(TriCub)
{
    NS(be_tricub_int_t)      nx  SIXTRL_ALIGN( 8 );
    NS(be_tricub_int_t)      ny  SIXTRL_ALIGN( 8 );
    NS(be_tricub_int_t)      nz  SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)     x0  SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)     y0  SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)     z0  SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)     dx  SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)     dy  SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)     dz  SIXTRL_ALIGN( 8 );
    NS(be_tricub_ptr_real_t) phi SIXTRL_ALIGN( 8 );
}
NS(TriCub);

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(TriCub_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(TriCub_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN  SIXTRL_BE_ARGPTR_DEC NS(TriCub)*
NS(TriCub_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN void NS(TriCub_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(TriCub)* NS(TriCub_init)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_int_t) const nx, NS(be_tricub_int_t) const ny,
    NS(be_tricub_int_t) const nz );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TriCub_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const tricub );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TriCub_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const tricub );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TriCub_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(be_tricub_int_t) const nx, NS(be_tricub_int_t) const ny,
    NS(be_tricub_int_t) const nz,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)*
NS(TriCub_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_tricub_int_t) const nx, NS(be_tricub_int_t) const ny,
    NS(be_tricub_int_t) const nz );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)*
NS(TriCub_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_tricub_int_t) const nx, NS(be_tricub_int_t) const ny,
    NS(be_tricub_int_t) const nz,
    NS(be_tricub_real_t) const x0,
    NS(be_tricub_real_t) const y0, NS(be_tricub_real_t) const z0,
    NS(be_tricub_real_t) const dx, NS(be_tricub_real_t) const dy,
    NS(be_tricub_real_t) const dz,
    NS(be_tricub_ptr_real_t) ptr_to_phi_data );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)*
NS(TriCub_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT orig );

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCub_get_nx)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCub_get_ny)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCub_get_nz)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCub_get_phi_index)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub,
    NS(be_tricub_int_t) const ix, NS(be_tricub_int_t) const iy,
    NS(be_tricub_int_t) const iz, NS(be_tricub_int_t) const ii );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_get_x0)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_get_y0)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_get_z0)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_get_dx)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_get_dy)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_get_dz)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_ptr_const_real_t)
NS(TriCub_get_ptr_const_phi)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_ptr_real_t) NS(TriCub_get_ptr_phi)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_get_phi)(
    const SIXTRL_BE_ARGPTR_DEC NS(TriCub) *const SIXTRL_RESTRICT tricub,
    NS(be_tricub_int_t) const ix, NS(be_tricub_int_t) const iy,
    NS(be_tricub_int_t) const iz, NS(be_tricub_int_t) const ii );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCub_get_phi_size)(
    const SIXTRL_BE_ARGPTR_DEC NS(TriCub) *const SIXTRL_RESTRICT tricub );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN void NS(TriCub_set_x0)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const x0 );

SIXTRL_STATIC SIXTRL_FN void NS(TriCub_set_y0)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const y0 );

SIXTRL_STATIC SIXTRL_FN void NS(TriCub_set_z0)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const z0 );

SIXTRL_STATIC SIXTRL_FN void NS(TriCub_set_dx)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const dx );

SIXTRL_STATIC SIXTRL_FN void NS(TriCub_set_dy)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const dy );

SIXTRL_STATIC SIXTRL_FN void NS(TriCub_set_dz)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const dz );

SIXTRL_STATIC SIXTRL_FN void NS(TriCub_set_phi)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_int_t) const ix, NS(be_tricub_int_t) const iy,
    NS(be_tricub_int_t) const iz, NS(be_tricub_int_t) const ii,
    NS(be_tricub_real_t) const phi_value );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN void NS(TriCub_assign_ptr_phi)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_ptr_real_t) SIXTRL_RESTRICT ptr_to_phi_data_to_assign );

SIXTRL_STATIC SIXTRL_FN int NS(TriCub_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT source );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */


#if !defined( _GPUCODE )
    #if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.h"
    #endif /* !defined( SIXTRL_NO_INCLUDES ) */
#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_INLINE NS(buffer_size_t)
NS(TriCub_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub,
    NS(buffer_size_t) const slot_size )
{
    ( void )buffer;
    ( void )tricub;
    ( void )slot_size;
    return ( NS(buffer_size_t) )1u;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(TriCub_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t num_slots = ( buf_size_t )0u;
    NS(be_tricub_int_t) const phi_size = NS(TriCub_get_phi_size)( tricub );

    if( ( buffer != SIXTRL_NULLPTR ) && ( slot_size > ( buf_size_t )0u ) &&
        ( tricub != SIXTRL_NULLPTR ) && ( phi_size > 0 ) )
    {
        num_slots = NS(ManagedBuffer_get_slot_based_length)(
            ( ( buf_size_t )phi_size ) * sizeof( NS(be_tricub_real_t) ),
                slot_size );

        num_slots /= slot_size;
    }

    return num_slots;
}

SIXTRL_INLINE  SIXTRL_BE_ARGPTR_DEC NS(TriCub)* NS(TriCub_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub )
{
    if( tricub != SIXTRL_NULLPTR )
    {
        tricub->nx  = ( NS(be_tricub_int_t) )0u;
        tricub->ny  = ( NS(be_tricub_int_t) )0u;
        tricub->nz  = ( NS(be_tricub_int_t) )0u;

        tricub->phi = SIXTRL_NULLPTR;

        NS(TriCub_clear)( tricub );
    }

    return tricub;
}

SIXTRL_INLINE void NS(TriCub_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub )
{
    if( tricub != SIXTRL_NULLPTR )
    {
        NS(be_tricub_int_t) const phi_size =
            NS(TriCub_get_phi_size)( tricub );

        tricub->x0  = ( NS(be_tricub_real_t) )0.0;
        tricub->y0  = ( NS(be_tricub_real_t) )0.0;
        tricub->z0  = ( NS(be_tricub_real_t) )0.0;

        tricub->dx  = ( NS(be_tricub_real_t) )1.0;
        tricub->dy  = ( NS(be_tricub_real_t) )1.0;
        tricub->dz  = ( NS(be_tricub_real_t) )1.0;

        if( ( tricub->phi != SIXTRL_NULLPTR ) && ( phi_size > 0 ) )
        {
            SIXTRL_STATIC_VAR NS(be_tricub_real_t) const ZERO =
                ( NS(be_tricub_real_t) )0.0;

            SIXTRACKLIB_SET_VALUES(
                SIXTRL_REAL_T, tricub->phi, phi_size, ZERO );
        }
    }
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(TriCub)* NS(TriCub_init)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_int_t) const nx, NS(be_tricub_int_t) const ny,
    NS(be_tricub_int_t) const nz )
{
    if( ( tricub != SIXTRL_NULLPTR ) &&
        ( nx > 0 ) && ( ny > 0 ) && ( nx > 0 ) )
    {
        tricub->nx = nx;
        tricub->ny = ny;
        tricub->nz = nz;
    }
    else
    {
        tricub->nx = ( NS(be_tricub_int_t) )0u;
        tricub->ny = ( NS(be_tricub_int_t) )0u;
        tricub->nz = ( NS(be_tricub_int_t) )0u;
    }

    return tricub;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCub_get_nx)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->nx;
}

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCub_get_ny)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->ny;
}

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCub_get_nz)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->nz;
}

/* TODO: Check row-major?`*/

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCub_get_phi_index)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub,
    NS(be_tricub_int_t) const ix, NS(be_tricub_int_t) const iy,
    NS(be_tricub_int_t) const iz, NS(be_tricub_int_t) const ii )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );

    NS(be_tricub_int_t) const iy_stride =
        SIXTRL_BE_TRICUBMAP_MATRIX_DIM * tricub->nz;

    return ix * tricub->ny * iy_stride + iy * iy_stride
         + iz * SIXTRL_BE_TRICUBMAP_MATRIX_DIM + ii;
}

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCub_get_phi_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->nx * tricub->ny * tricub->nz *
           SIXTRL_BE_TRICUBMAP_MATRIX_DIM;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_get_x0)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->x0;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_get_y0)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->y0;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_get_z0)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->z0;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_get_dx)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->dx;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_get_dy)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->dy;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_get_dz)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->dz;
}

SIXTRL_INLINE NS(be_tricub_ptr_const_real_t)
NS(TriCub_get_ptr_const_phi)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->phi;
}

SIXTRL_INLINE NS(be_tricub_ptr_real_t) NS(TriCub_get_ptr_phi)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->phi;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_get_phi)(
    const SIXTRL_BE_ARGPTR_DEC NS(TriCub) *const SIXTRL_RESTRICT tricub,
    NS(be_tricub_int_t) const ix, NS(be_tricub_int_t) const iy,
    NS(be_tricub_int_t) const iz, NS(be_tricub_int_t) const ii )
{
    NS(be_tricub_int_t) const index = NS(TriCub_get_phi_index)(
        tricub, ix, iy, iz, ii );

    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( tricub->phi != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( index < NS(TriCub_get_phi_size)( tricub ) );

    return tricub->phi[ index ];
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(TriCub_set_x0)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const x0 )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    tricub->x0 = x0;
}

SIXTRL_INLINE void NS(TriCub_set_y0)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const y0 )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    tricub->y0 = y0;
}

SIXTRL_INLINE void NS(TriCub_set_z0)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const z0 )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    tricub->z0 = z0;
}

SIXTRL_INLINE void NS(TriCub_set_dx)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const dx )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    tricub->dx = dx;
}

SIXTRL_INLINE void NS(TriCub_set_dy)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const dy )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    tricub->dy = dy;
}

SIXTRL_INLINE void NS(TriCub_set_dz)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const dz )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    tricub->dz = dz;
}

SIXTRL_INLINE void NS(TriCub_set_phi)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_int_t) const ix, NS(be_tricub_int_t) const iy,
    NS(be_tricub_int_t) const iz, NS(be_tricub_int_t) const ii,
    NS(be_tricub_real_t) const phi_value )
{
    NS(be_tricub_int_t) const index = NS(TriCub_get_phi_index)(
        tricub, ix, iy, iz, ii );

    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( tricub->phi != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( index < NS(TriCub_get_phi_size)( tricub ) );

    tricub->phi[ index ] = phi_value;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(TriCub_assign_ptr_phi)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_ptr_real_t) SIXTRL_RESTRICT ptr_to_phi_data_to_assign )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( tricub->phi == SIXTRL_NULLPTR ) ||
                   ( ptr_to_phi_data_to_assign == SIXTRL_NULLPTR ) );

    tricub->phi = ptr_to_phi_data_to_assign;
}

SIXTRL_INLINE int NS(TriCub_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT source )
{
    int success = -1;

    if( ( destination != SIXTRL_NULLPTR ) && ( source != SIXTRL_NULLPTR ) &&
        ( destination != source ) &&
        ( NS(TriCub_get_ptr_const_phi)( destination ) != SIXTRL_NULLPTR ) &&
        ( NS(TriCub_get_ptr_const_phi)( source      ) != SIXTRL_NULLPTR ) &&
        ( NS(TriCub_get_ptr_const_phi)( destination ) !=
          NS(TriCub_get_ptr_const_phi)( source      ) ) &&
        ( NS(TriCub_get_phi_size)( destination ) ==
          NS(TriCub_get_phi_size)( source ) ) )
    {
        NS(TriCub_init)( destination, NS(TriCub_get_nx)( source ),
            NS(TriCub_get_ny)( source ), NS(TriCub_get_nz)( source ) );

        NS(TriCub_set_x0)( destination, NS(TriCub_get_x0)( source ) );
        NS(TriCub_set_y0)( destination, NS(TriCub_get_y0)( source ) );
        NS(TriCub_set_z0)( destination, NS(TriCub_get_z0)( source ) );

        NS(TriCub_set_dx)( destination, NS(TriCub_get_dx)( source ) );
        NS(TriCub_set_dy)( destination, NS(TriCub_get_dy)( source ) );
        NS(TriCub_set_dz)( destination, NS(TriCub_get_dz)( source ) );

        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
            NS(TriCub_get_ptr_phi)( destination ),
            NS(TriCub_get_ptr_const_phi)( source ),
            NS(TriCub_get_phi_size)( source ) );

        success = 0;
    }

    return success;
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */


#endif /* SIXTRACKLIB_COMMON_BE_TRICUB_C99_H__ */
/* end: sixtracklib/common/be_tricub/be_tricub.h */

