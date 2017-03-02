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

// declarations for block handling functions
CLGLOBAL block_t* block_initialize(unsigned int size);
CLGLOBAL block_t* block_reshape(CLGLOBAL block_t *block, unsigned int n);
void block_clean(CLGLOBAL block_t *block);
CLGLOBAL block_t* block_add_drift(CLGLOBAL block_t *block, uint64_t *offsets, uint64_t nel, double length);
CLGLOBAL block_t* block_add_cavity(CLGLOBAL block_t *block, uint64_t *offsets, uint64_t nel, double volt, double freq, double lag);
CLGLOBAL block_t* block_add_align(CLGLOBAL block_t *block, uint64_t *offsets, uint64_t nel, double tilt, double dx, double dy);
CLGLOBAL block_t* block_add_multipole(CLGLOBAL block_t *block, uint64_t *offsets, uint64_t nel, double *knl, unsigned int knl_len, double *ksl, unsigned int ksl_len, double length, double hxl, double hyl);
CLGLOBAL block_t* block_add_block(CLGLOBAL block_t *block, uint64_t *offsets, uint64_t nel);

#endif
