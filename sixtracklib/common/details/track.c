#if !defined( _GPUCODE )

#include "sixtracklib/_impl/namespace_begin.h"
#include "sixtracklib/common/track.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/impl/particles_type.h"
#include "sixtracklib/common/restrict.h"

#include "sixtracklib/common/impl/track_impl.h"

/* -------------------------------------------------------------------------- */

extern int NS(Drift_track)( 
    NS(Particles)* SIXTRL_RESTRICT particles, uint64_t ip, double length );

extern int NS(DriftExact_track)(
    NS(Particles)* SIXTRL_RESTRICT particles, uint64_t ip, double length );

#endif /* _GPUCODE */

/* -------------------------------------------------------------------------- */

int NS(Drift_track)(
    NS(Particles)* SIXTRL_RESTRICT particles, uint64_t ip, double length )
{
    SIXTRACKLIB_DRIFT_TRACK_IMPL( double, particles, ip, length )
    
    return 1;
}

int NS(DriftExact_track)(
    NS(Particles)* SIXTRL_RESTRICT particles, uint64_t ip, double length )
{
    SIXTRACKLIB_DRIFT_EXACT_TRACK_IMPL( double, particles, ip, length )
    
    return 1;
}

/* -------------------------------------------------------------------------- */

/* end: sixtracklib/common/details/track.c */
