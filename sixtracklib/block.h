#ifndef _BLOCK_
#define _BLOCK_

#include "beam.h"
#include "value.h"

typedef enum type_t {IntegerID, DoubleID,
             DriftID, DriftExactID,
             MultipoleID, CavityID, AlignID,
             BlockID,LinMapID,BB4DID} type_t;


#ifdef _GPUCODE

CLKERNEL void Block_track(CLGLOBAL value_t *data,
                         CLGLOBAL Particle *particles,
                         uint64_t blockid, uint64_t nturn, uint64_t npart,
                         uint64_t elembyelemid,  uint64_t turnbyturnid);
#else

int Block_track(value_t *data, Beam *beam,
                uint64_t blockid, uint64_t nturn,
                uint64_t elembyelemid,  uint64_t turnbyturnid);

#endif


#endif
