#ifndef SIXTRL_COMMON_BE_BEAMBEAM_BE_BEAMBEAM4D_H__
#define SIXTRL_COMMON_BE_BEAMBEAM_BE_BEAMBEAM4D_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_defines.h"
    #include "sixtracklib/common/impl/beam_elements_defines.h"
    #include "sixtracklib/common/impl/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T*        NS(beambeam4d_real_ptr_t);
typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const*  NS(beambeam4d_real_const_ptr_t);

typedef struct NS(BeamBeam4D)
{
    SIXTRL_UINT64_T                           size      SIXTRL_ALIGN( 8 );
    NS(beambeam4d_real_ptr_t) SIXTRL_RESTRICT data      SIXTRL_ALIGN( 8 );
}
NS(BeamBeam4D);

SIXTRL_FN SIXTRL_STATIC  SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_preset)( NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam );

#if !defined( _GPUCODE )

SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const data_size );

SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const  data_size,
    NS(beambeam4d_real_ptr_t) SIXTRL_RESTRICT input_data );

SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT orig );

#endif /* !defined( _GPUCODE ) */

SIXTRL_FN SIXTRL_STATIC NS(beambeam4d_real_const_ptr_t)
NS(BeamBeam4D_get_const_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC NS(beambeam4d_real_ptr_t) NS(BeamBeam4D_get_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC SIXTRL_UINT64_T NS(BeamBeam4D_get_data_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam );

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====             Implementation of inline functions                ===== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #if !defined( _GPUCODE ) || defined( __CUDACC__ )
    #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) || defined( __CUDACC__ ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_preset)( NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam )
{
    if( beam_beam != SIXTRL_NULLPTR )
    {
        beam_beam->size = ( SIXTRL_UINT64_T )0u;
        beam_beam->data   = SIXTRL_NULLPTR;
    }

    return beam_beam;
}

SIXTRL_INLINE NS(beambeam4d_real_const_ptr_t)
NS(BeamBeam4D_get_const_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    return beam_beam->data;
}

SIXTRL_INLINE NS(beambeam4d_real_ptr_t)
NS(BeamBeam4D_get_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam )
{
    return ( NS(beambeam4d_real_ptr_t) )NS(BeamBeam4D_get_const_data)( beam_beam );
}

SIXTRL_INLINE SIXTRL_UINT64_T NS(BeamBeam4D_get_data_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    return beam_beam->size;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRL_COMMON_BE_BEAMBEAM_BE_BEAMBEAM4D_H__ */

/* sixtracklib/common/be_beambeam/be_beambeam4d.h */
