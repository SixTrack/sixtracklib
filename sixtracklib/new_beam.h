#ifndef _BEAM
#define _BEAM

#include "value.h"
#include "particle.h"

typedef struct Beam {
  uint64_t npart;
  Particle* particles;
} Beam;

#ifndef _GPUCODE
  #include <stdlib.h>
  Beam* Beam_new(uint64_t npart);
#endif

#endif


