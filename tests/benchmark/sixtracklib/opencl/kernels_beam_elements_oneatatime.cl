#if defined(cl_khr_fp64)
#  pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(cl_amd_fp64)
#  pragma OPENCL EXTENSION cl_amd_fp64: enable
#else
# error double precision is not supported
#endif
#include "sixtracklib/_impl/namespace_begin.h"
#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/impl/particles_type.h"
#include "sixtracklib/common/impl/particles_api.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/impl/beam_elements_type.h"
#include "sixtracklib/common/impl/beam_elements_api.h"
#include "sixtracklib/common/beam_elements.h"


kernel void unserialize(
    global uchar *copy_buffer, // uint8_t is uchar
    global uchar *copy_buffer_particles, // uint8_t is uchar
    ulong NUM_PARTICLES
    )
{
  size_t gid = get_global_id(0);
  NS(Blocks) copied_beam_elements;
  NS(Blocks_preset)( &copied_beam_elements ); // very important for initialization
  int ret = NS(Blocks_unserialize)(&copied_beam_elements, copy_buffer);
  NS(Blocks) copied_particles_buffer;
  NS(Blocks_preset) (&copied_particles_buffer);

  ret = NS(Blocks_unserialize)(&copied_particles_buffer, copy_buffer_particles);
}

kernel void track_drift_particle(
    global uchar *copy_buffer_drift, // uint8_t is uchar
    global uchar *copy_buffer_particles, // uint8_t is uchar
    ulong NUM_PARTICLES,
    ulong NUM_TURNS, // number of times a particle is mapped over each of the beam_elements
    ulong offset
    )
{
  NS(block_num_elements_t) ii = get_global_id(0);
  if(ii >= NUM_PARTICLES) return;

  /* For the particles */
  NS(Blocks) copied_particles_buffer;
  NS(Blocks_preset) (&copied_particles_buffer);

  int ret = NS(Blocks_unserialize)(&copied_particles_buffer, copy_buffer_particles);
  SIXTRL_GLOBAL_DEC st_BlockInfo const* it  =  // is 'it' pointing to the outer particles? check.
    st_Blocks_get_const_block_infos_begin( &copied_particles_buffer );
  SIXTRL_GLOBAL_DEC NS(Particles) const* particles = 
    ( SIXTRL_GLOBAL_DEC st_Particles const* )it->begin; 

  // *particles now points to the first 'outer' particle
  // @ Assuming only a single outer particle
  // each 'ii' refers to an inner particle

  /* for the beam element */
  NS(Blocks) copied_beam_elements;
  NS(Blocks_preset)( &copied_beam_elements ); // very important for initialization
  ret = NS(Blocks_unserialize)(&copied_beam_elements, copy_buffer_drift);

  SIXTRL_STATIC SIXTRL_REAL_T const ONE      = ( SIXTRL_REAL_T )1;
  SIXTRL_STATIC SIXTRL_REAL_T const ONE_HALF = ( SIXTRL_REAL_T )0.5L;

  // for each particle we apply the beam_elements, as applicable (decided by the switch case)

  for (size_t nt=0; nt < NUM_TURNS; ++nt) {
    SIXTRL_GLOBAL_DEC st_BlockInfo const* belem_it  = 
      st_Blocks_get_const_block_infos_begin( &copied_beam_elements );
    SIXTRL_GLOBAL_DEC st_BlockInfo const* belem_end =
      st_Blocks_get_const_block_infos_end( &copied_beam_elements );

    belem_it +=offset;
    belem_end = belem_it + 250;
    for( ; belem_it != belem_end ; ++belem_it )
    {
      st_BlockInfo const info = *belem_it;
      NS(BlockType) const type_id =  st_BlockInfo_get_type_id(&info );
      switch( type_id )
      {
        case st_BLOCK_TYPE_DRIFT:
          {
            __global st_Drift const* drift = 
              st_Blocks_get_const_drift( &info );
            st_Drift const drift_private = *drift;
            SIXTRL_REAL_T const length = st_Drift_get_length( &drift_private );  
            SIXTRL_REAL_T const rpp = particles->rpp[ii]; 
            SIXTRL_REAL_T const px = particles->px[ii] * rpp; 
            SIXTRL_REAL_T const py = particles->py[ii] * rpp; 
            SIXTRL_REAL_T const dsigma = 
              ONE - particles->rvv[ii]  * ( ONE + ONE_HALF * ( px * px + py * py ) );
            SIXTRL_REAL_T sigma = particles->sigma[ii];
            SIXTRL_REAL_T s = particles->s[ii];
            SIXTRL_REAL_T x = particles->x[ii];
            SIXTRL_REAL_T y = particles->y[ii];
            sigma += length * dsigma;
            s     += length;
            x     += length * px;
            y     += length * py;
            particles->s[ ii ] = s;
            particles->x[ ii ] = x;
            particles->y[ ii ] = y;
            particles->sigma[ ii ] = sigma;
            break;
          }
      };
    }
  }

};


kernel void track_drift_exact_particle(
    global uchar *copy_buffer_drift_exact, // uint8_t is uchar
    global uchar *copy_buffer_particles, // uint8_t is uchar
    ulong NUM_PARTICLES,
    ulong NUM_TURNS, // number of times a particle is mapped over each of the beam_elements
    ulong offset
    )
{
  NS(block_num_elements_t) ii = get_global_id(0);
  if(ii >= NUM_PARTICLES) return;

  /* For the particles */
  NS(Blocks) copied_particles_buffer;
  NS(Blocks_preset) (&copied_particles_buffer);

  int ret = NS(Blocks_unserialize)(&copied_particles_buffer, copy_buffer_particles);
  SIXTRL_GLOBAL_DEC st_BlockInfo const* it  =  // is 'it' pointing to the outer particles? check.
    st_Blocks_get_const_block_infos_begin( &copied_particles_buffer );
  SIXTRL_GLOBAL_DEC NS(Particles) const* particles = 
    ( SIXTRL_GLOBAL_DEC st_Particles const* )it->begin; 

  // *particles now points to the first 'outer' particle
  // @ Assuming only a single outer particle
  // each 'ii' refers to an inner particle

  /* for the beam element */
  NS(Blocks) copied_beam_elements;
  NS(Blocks_preset)( &copied_beam_elements ); // very important for initialization
  ret = NS(Blocks_unserialize)(&copied_beam_elements, copy_buffer_drift_exact);

  SIXTRL_STATIC SIXTRL_REAL_T const ONE      = ( SIXTRL_REAL_T )1;
  SIXTRL_STATIC SIXTRL_REAL_T const ONE_HALF = ( SIXTRL_REAL_T )0.5L;

  // for each particle we apply the beam_elements, as applicable (decided by the switch case)

  for (size_t nt=0; nt < NUM_TURNS; ++nt) {
    SIXTRL_GLOBAL_DEC st_BlockInfo const* belem_it  = 
      st_Blocks_get_const_block_infos_begin( &copied_beam_elements );
    SIXTRL_GLOBAL_DEC st_BlockInfo const* belem_end =
      st_Blocks_get_const_block_infos_end( &copied_beam_elements ) ;

    belem_it +=offset;
    belem_end = belem_it + 250;
    for( ; belem_it != belem_end ; ++belem_it )
    {
      st_BlockInfo const info = *belem_it;
      NS(BlockType) const type_id =  st_BlockInfo_get_type_id(&info );
      switch( type_id )
      {
        case st_BLOCK_TYPE_DRIFT_EXACT:
          {
            __global st_DriftExact const* drift_exact = 
              st_Blocks_get_const_drift_exact( &info );
            st_DriftExact const drift_exact_private = *drift_exact;

            SIXTRL_STATIC SIXTRL_REAL_T const ONE = ( SIXTRL_REAL_T )1u;

            SIXTRL_REAL_T const length = NS(DriftExact_get_length)( &drift_exact_private );
            SIXTRL_REAL_T const delta  = particles->delta[ii];
            SIXTRL_REAL_T const beta0  = particles->beta0[ii];
            SIXTRL_REAL_T const px  = particles->px[ii];
            SIXTRL_REAL_T const py  = particles->py[ii];
            SIXTRL_REAL_T sigma        = particles->sigma[ii];

            SIXTRL_REAL_T const opd   = delta + ONE;
            SIXTRL_REAL_T const lpzi  = ( length ) / 
              sqrt( opd * opd - px * px - py * py );

            SIXTRL_REAL_T const lbzi  = ( beta0 * beta0 * sigma + ONE ) * lpzi;

            SIXTRL_REAL_T s = particles->s[ii];
            SIXTRL_REAL_T x = particles->x[ii];
            SIXTRL_REAL_T y = particles->y[ii];

            x     += px * lpzi;
            y     += py * lpzi;
            s     += length;
            sigma += length - lbzi;

            particles->s[ ii ] = s;
            particles->x[ ii ] = x;
            particles->y[ ii ] = y;
            particles->sigma[ ii ] = sigma;
            break;
          }

      };
    }
  }

};

kernel void track_cavity_particle(
    global uchar *copy_buffer_cavity, // uint8_t is uchar
    global uchar *copy_buffer_particles, // uint8_t is uchar
    ulong NUM_PARTICLES,
    ulong NUM_TURNS, // number of times a particle is mapped over each of the beam_elements
    ulong offset
    )
{
  NS(block_num_elements_t) ii = get_global_id(0);
  if(ii >= NUM_PARTICLES) return;

  /* For the particles */
  NS(Blocks) copied_particles_buffer;
  NS(Blocks_preset) (&copied_particles_buffer);

  int ret = NS(Blocks_unserialize)(&copied_particles_buffer, copy_buffer_particles);
  SIXTRL_GLOBAL_DEC st_BlockInfo const* it  =  // is 'it' pointing to the outer particles? check.
    st_Blocks_get_const_block_infos_begin( &copied_particles_buffer );
  SIXTRL_GLOBAL_DEC NS(Particles) const* particles = 
    ( SIXTRL_GLOBAL_DEC st_Particles const* )it->begin; 

  // *particles now points to the first 'outer' particle
  // @ Assuming only a single outer particle
  // each 'ii' refers to an inner particle

  /* for the beam element */
  NS(Blocks) copied_beam_elements;
  NS(Blocks_preset)( &copied_beam_elements ); // very important for initialization
  ret = NS(Blocks_unserialize)(&copied_beam_elements, copy_buffer_cavity);

  SIXTRL_STATIC SIXTRL_REAL_T const ONE      = ( SIXTRL_REAL_T )1;
  SIXTRL_STATIC SIXTRL_REAL_T const ONE_HALF = ( SIXTRL_REAL_T )0.5L;

  // for each particle we apply the beam_elements, as applicable (decided by the switch case)

  for (size_t nt=0; nt < NUM_TURNS; ++nt) {
    SIXTRL_GLOBAL_DEC st_BlockInfo const* belem_it  = 
      st_Blocks_get_const_block_infos_begin( &copied_beam_elements );
    SIXTRL_GLOBAL_DEC st_BlockInfo const* belem_end =
      st_Blocks_get_const_block_infos_end( &copied_beam_elements ) ;
    
    belem_it +=offset;
    belem_end = belem_it + 250;

    for( ; belem_it != belem_end ; ++belem_it )
    {
      st_BlockInfo const info = *belem_it;
      NS(BlockType) const type_id =  st_BlockInfo_get_type_id(&info );
      switch( type_id )
      {
        case st_BLOCK_TYPE_CAVITY:
          {
            __global st_Cavity const* cavity = 
              st_Blocks_get_const_cavity( &info );
            st_Cavity const cavity_private = *cavity;

            SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE  = ( SIXTRL_REAL_T )1.0L;
            SIXTRL_STATIC_VAR SIXTRL_REAL_T const TWO  = ( SIXTRL_REAL_T )2.0L;
            SIXTRL_STATIC_VAR SIXTRL_REAL_T const PI   =
              ( SIXTRL_REAL_T )3.1415926535897932384626433832795028841971693993751L;
            SIXTRL_STATIC_VAR SIXTRL_REAL_T const CLIGHT = ( SIXTRL_REAL_T )299792458u;
            SIXTRL_REAL_T const beta0  = particles->beta0[ii];
            SIXTRL_REAL_T sigma        = particles->sigma[ii];
            SIXTRL_REAL_T psigma_init        = particles->psigma[ii];
            SIXTRL_REAL_T chi        = particles->chi[ii];
            SIXTRL_REAL_T p0c        = particles->p0c[ii];

            SIXTRL_REAL_T const phase = NS(Cavity_get_lag)( &cavity_private ) -
              ( TWO * PI * NS(Cavity_get_frequency)( &cavity_private ) *
                ( sigma / beta0 )
              ) / CLIGHT;


            SIXTRL_REAL_T const psigma =
              psigma_init +
              ( chi *
                NS(Cavity_get_voltage)( &cavity_private ) * sin( phase ) ) /
              ( p0c * beta0 );


            SIXTRL_REAL_T const pt    = psigma * beta0;
            SIXTRL_REAL_T const opd   = sqrt( pt * pt + TWO * psigma + ONE );
            SIXTRL_REAL_T const beta  = opd / ( ONE / beta0 + pt );

            particles->psigma[ii] = psigma;
            particles->delta[ii] = opd - ONE;
            particles->rpp[ii] = ONE  / opd ;
            particles->rvv[ii] = beta0 / beta;

            break;
          }
      };
    }
  }

};

kernel void track_align_particle(
    global uchar *copy_buffer_align, // uint8_t is uchar
    global uchar *copy_buffer_particles, // uint8_t is uchar
    ulong NUM_PARTICLES,
    ulong NUM_TURNS, // number of times a particle is mapped over each of the beam_elements
    ulong offset
    )
{
  NS(block_num_elements_t) ii = get_global_id(0);
  if(ii >= NUM_PARTICLES) return;

  /* For the particles */
  NS(Blocks) copied_particles_buffer;
  NS(Blocks_preset) (&copied_particles_buffer);

  int ret = NS(Blocks_unserialize)(&copied_particles_buffer, copy_buffer_particles);
  SIXTRL_GLOBAL_DEC st_BlockInfo const* it  =  // is 'it' pointing to the outer particles? check.
    st_Blocks_get_const_block_infos_begin( &copied_particles_buffer );
  SIXTRL_GLOBAL_DEC NS(Particles) const* particles = 
    ( SIXTRL_GLOBAL_DEC st_Particles const* )it->begin; 

  // *particles now points to the first 'outer' particle
  // @ Assuming only a single outer particle
  // each 'ii' refers to an inner particle

  /* for the beam element */
  NS(Blocks) copied_beam_elements;
  NS(Blocks_preset)( &copied_beam_elements ); // very important for initialization
  ret = NS(Blocks_unserialize)(&copied_beam_elements, copy_buffer_align);

  SIXTRL_STATIC SIXTRL_REAL_T const ONE      = ( SIXTRL_REAL_T )1;
  SIXTRL_STATIC SIXTRL_REAL_T const ONE_HALF = ( SIXTRL_REAL_T )0.5L;

  // for each particle we apply the beam_elements, as applicable (decided by the switch case)

  for (size_t nt=0; nt < NUM_TURNS; ++nt) {
    SIXTRL_GLOBAL_DEC st_BlockInfo const* belem_it  = 
      st_Blocks_get_const_block_infos_begin( &copied_beam_elements );
    SIXTRL_GLOBAL_DEC st_BlockInfo const* belem_end =
      st_Blocks_get_const_block_infos_end( &copied_beam_elements );

    belem_it +=offset;

    for( ; belem_it != belem_end ; ++belem_it )
    {
      st_BlockInfo const info = *belem_it;
      NS(BlockType) const type_id =  st_BlockInfo_get_type_id(&info );
      switch( type_id )
      {
        case st_BLOCK_TYPE_ALIGN:
          {
            __global st_Align const* align = 
              st_Blocks_get_const_align( &info );
            st_Align const align_private = *align;
            SIXTRL_REAL_T const sz = NS(Align_get_sz)( &align_private );
            SIXTRL_REAL_T const cz = NS(Align_get_cz)( &align_private );
            SIXTRL_REAL_T x = particles->x[ii];
            SIXTRL_REAL_T y = particles->y[ii];
            SIXTRL_REAL_T px = particles->px[ii];
            SIXTRL_REAL_T py = particles->py[ii];

            SIXTRL_REAL_T temp     = cz * x - sz * y - NS(Align_get_dx)( &align_private );
            y    =  sz * x + cz * y - NS(Align_get_dy)( &align_private );
            x    =  temp;

            temp =  cz * px + sz * py;
            py   = -sz * px + cz * py;
            px   =  temp;

            particles->x[ii] = x;
            particles->y[ii] = y;
            particles->px[ii] = px;
            particles->py[ii] = py;

            break;
          }

      };
    }
  }

};
