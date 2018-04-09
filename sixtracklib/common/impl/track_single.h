#ifndef SIXTRACKLIB_COMMON_IMPL_TRACK_SINGLE_H__
#define SIXTRACKLIB_COMMON_IMPL_TRACK_SINGLE_H__

#include "sixtracklib/_impl/namespace_begin.h"

#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/restrict.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct NS( SingleParticle );

int NS( Drift_track_single )( struct NS( SingleParticle ) *
                                  SIXTRL_RESTRICT particle,
                              double length );

int NS( DriftExact_track_single )( struct NS( SingleParticle ) *
                                       SIXTRL_RESTRICT particle,
                                   double length );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* SIXTRACKLIB_COMMON_IMPL_TRACK_SINGLE_H__ */
