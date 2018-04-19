#include "sixtracklib/common/impl/track_single.h"
#include "sixtracklib/_impl/definitions.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/single_particle.h"

/* -------------------------------------------------------------------------- */

extern int NS( TrackSingle_drift )( struct NS( SingleParticle ) *
                                         SIXTRL_RESTRICT particles,
                                     SIXTRL_REAL_T const length );

extern int NS( TrackSingle_drift_exact )( struct NS( SingleParticle ) *
                                              SIXTRL_RESTRICT particles,
                                          SIXTRL_REAL_T const length );

/* -------------------------------------------------------------------------- */

int NS( TrackSingle_drift )( struct NS( SingleParticle ) * SIXTRL_RESTRICT p,
                              SIXTRL_REAL_T const len )
{
    SIXTRL_STATIC SIXTRL_REAL_T const ONE = ( SIXTRL_REAL_T )1;
    SIXTRL_STATIC SIXTRL_REAL_T const TWO = ( SIXTRL_REAL_T )1;
    
    SIXTRL_REAL_T const rpp = p->rpp;
    SIXTRL_REAL_T const px = p->px * rpp;
    SIXTRL_REAL_T const py = p->py * rpp;

    p->x += px * len;
    p->y += py * len;
    p->sigma += len * ( ONE - p->rvv * ( ONE + ( px * px + py * py ) / TWO ) );
    p->s += len;

    return 1;
}

int NS( TrackSingle_drift_exact )( struct NS( SingleParticle ) *
                                       SIXTRL_RESTRICT particle,
                                   SIXTRL_REAL_T const length )
{
    SIXTRL_STATIC SIXTRL_REAL_T const ONE = ( SIXTRL_REAL_T )1;
    
    SIXTRL_REAL_T const opd = ONE + particle->delta;
    SIXTRL_REAL_T px = particle->px;
    SIXTRL_REAL_T py = particle->py;
    SIXTRL_REAL_T const lpzi = length / sqrt( opd * opd - px * px - py * py );
    SIXTRL_REAL_T const beta0 = particle->beta0;
    SIXTRL_REAL_T const beta0_squ = beta0 * beta0;
    SIXTRL_REAL_T const lbzi = ( beta0_squ * particle->psigma + ONE ) * lpzi;

    particle->x += px * lpzi;
    particle->y += py * lpzi;
    particle->sigma += length - lbzi;
    particle->s += length;

    return 1;
}

/* -------------------------------------------------------------------------- */

/* end: sixtracklib/common/details/track_single.c */
