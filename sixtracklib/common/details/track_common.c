#include "sixtracklib/common/details/track_common.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/single_particle.h"
#include "sixtracklib/common/restrict.h"

/* -------------------------------------------------------------------------- */

extern int Drift_track_single_particle_common(
    struct SingleParticle* SIXTRL_RESTRICT particle, double length );

extern int DriftExact_track_single_particle_common(
    struct SingleParticle* SIXTRL_RESTRICT particle, double length );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */

extern int Drift_track_particles_common( 
    struct Particles* SIXTRL_RESTRICT particles, uint64_t ip, double length );

extern int DriftExact_track_particles_common(
    struct Particles* SIXTRL_RESTRICT particles, uint64_t ip, double length );

/* -------------------------------------------------------------------------- */

int Drift_track_single_particle_common(
    struct SingleParticle* SIXTRL_RESTRICT particle, double length )
{
    double const rpp = particle->rpp;
    double const px  = particle->px * rpp;
    double const py  = particle->py * rpp;
    
    particle->x     += px * length;
    particle->y     += py * length;
    particle->sigma += length * ( 
        1.0 - particle->rvv * ( 1.0 + ( px * px + py * py ) / 2.0 ) );
    
    particle->s     += length;
    
    return 1;
}

int DriftExact_track_single_particle_common(
    struct SingleParticle* SIXTRL_RESTRICT particle, double length )
{
    double const  opd  = 1.0 + particle->delta;
    double        px   = particle->px;
    double        py   = particle->py;
    double const lpzi  = length / sqrt( opd * opd - px * px - py * py );
    double const beta0 = particle->beta0;
    double const lbzi  = ( beta0 * beta0 * particle->psigma + 1.0 ) * lpzi;
    
    particle->x     += px * lpzi;
    particle->y     += py * lpzi;
    particle->sigma += length - lbzi;
    particle->s     += length;
    
    return 1;
}

/* -------------------------------------------------------------------------- */

int Drift_track_particles_common(
    struct Particles* SIXTRL_RESTRICT particles, uint64_t ip, double length )
{
    double const rpp = particles->rpp[ ip ];
    double const px  = particles->px[ ip ] * rpp;
    double const py  = particles->py[ ip ] * rpp;
    
    particles->x[ ip ]     += px * length;
    particles->y[ ip ]     += py * length;
    particles->sigma[ ip ] += length * ( 
        1.0 - particles->rvv[ ip ] * ( 1.0 + ( px * px + py * py ) / 2.0 ) );
    
    particles->s[ ip ]     += length;
    
    return 1;
}

int DriftExact_track_particles_common(
    struct Particles* SIXTRL_RESTRICT particles, uint64_t ip, double length )
{
    double const opd   = 1.0 + particles->delta[ ip ];
    double const px    = particles->px[ ip ];
    double const py    = particles->py[ ip ];
    double const lpzi  = length / sqrt( opd * opd - px * px - py * py );
    double const beta0 = particles->beta0[ ip ];
    double const lbzi  = ( beta0 * beta0 * particles->psigma[ ip ] + 1.0 ) * lpzi;
    
    particles->x[ ip ]     += px * lpzi;
    particles->y[ ip ]     += py * lpzi;
    particles->sigma[ ip ] += length - lbzi;
    particles->s[ ip ]     += length;
    
    return 1;
}

/* -------------------------------------------------------------------------- */

/* end: sixtracklib/common/details/track_common.c */
