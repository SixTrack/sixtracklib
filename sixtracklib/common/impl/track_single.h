#ifndef SIXTRACKLIB_COMMON_IMPL_TRACK_SINGLE_H__
#define SIXTRACKLIB_COMMON_IMPL_TRACK_SINGLE_H__

#if !defined( _GPUCODE )
#include "sixtracklib/_impl/definitions.h"

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */
    
struct NS( SingleParticle );

int NS( Drift_track_single )( struct NS( SingleParticle ) *
                                  SIXTRL_RESTRICT particle,
                              SIXTRL_REAL_T const length );

int NS( DriftExact_track_single )( struct NS( SingleParticle ) *
                                       SIXTRL_RESTRICT particle,
                                   SIXTRL_REAL_T const length );

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_TRACK_SINGLE_H__ */
