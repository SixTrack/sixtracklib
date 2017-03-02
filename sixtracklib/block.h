//SixTrackLib
//
//Authors: R. De Maria, G. Iadarola, D. Pellegrini
//
//Copyright 2017 CERN. This software is distributed under the terms of the GNU
//Lesser General Public License version 2.1, copied verbatim in the file
//`COPYING''.
//
//In applying this licence, CERN does not waive the privileges and immunities
//granted to it by virtue of its status as an Intergovernmental Organization or
//submit itself to any jurisdiction.


#ifndef _BLOCK_
#define _BLOCK_

#include "beam.h"
#include "value.h"

typedef enum type_t {IntegerID, DoubleID,
             DriftID, DriftExactID,
             MultipoleID, CavityID, AlignID,
             BlockID} type_t;

typedef struct {
    unsigned int size;
    uint64_t last;
    CLGLOBAL value_t *data;
} block_t;

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
