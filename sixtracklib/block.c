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


#include <stdlib.h>
#include <math.h>
#include "block.h"
#include "track.c"


// Data management

type_t get_type(CLGLOBAL value_t *data, uint64_t elemid ) {
  return (type_t) data[elemid].i64;
}

//Block

uint64_t Block_get_nelen(CLGLOBAL value_t *data, size_t elemid ) {
  return data[elemid + 1].i64;
}

CLGLOBAL uint64_t *Block_get_elemids(CLGLOBAL value_t *data, size_t elemid ) {
  return &data[elemid + 2].u64 ;
}

// Tracking single

//#ifndef _GPUCODE
//#include <stdio.h>
//#endif

int track_single(CLGLOBAL value_t *data,
                 CLGLOBAL Particle *particles,
                 CLGLOBAL uint64_t *elemids,
                 uint64_t i_part, uint64_t i_elem,
                 uint64_t elembyelemoff, uint64_t turnbyturnoff){
   CLGLOBAL Particle* p = &particles[i_part];
   CLGLOBAL value_t *elem;
   uint64_t elemid;
   if (p->state >= 0 ) {
       elemid=elemids[i_elem];
       if ( (turnbyturnoff>0) && (i_elem==0) ){
         uint64_t dataoff=turnbyturnoff+sizeof(Particle)/8 * i_part;
         for (int i_attr=0;i_attr<sizeof(Particle)/8;i_attr++) {
            data[dataoff + i_attr] =
                 ((CLGLOBAL value_t *) p)[i_attr];
         }
       };
       enum type_t typeid = get_type(data, elemid);
       elem=data+elemid+1; //Data starts after typeid
//       _DP("Block_track: elemid=%zu typedid=%u\n",elemid,typeid);
       switch (typeid) {
           case DriftID:
                Drift_track(p, (CLGLOBAL Drift*) elem);
           break;
           case MultipoleID:
                Multipole_track(p, (CLGLOBAL Multipole*) elem);
           break;
           case CavityID:
                Cavity_track(p, (CLGLOBAL Cavity*) elem);
           break;
           case AlignID:
                Align_track(p, (CLGLOBAL Align*) elem);
           break;
           case IntegerID: break;
           case DoubleID: break;
           case BlockID: break;
           case DriftExactID:
                DriftExact_track(p, (CLGLOBAL DriftExact*) elem);
           break;
       }
       if (elembyelemoff>0){
         uint64_t dataoff=elembyelemoff+sizeof(Particle)/8 * i_part;
         for (int i_attr=0;i_attr<sizeof(Particle)/8;i_attr++) {
            data[dataoff + i_attr] =
                 ((CLGLOBAL value_t *) p)[i_attr];
         }
       };
   }
   return 1;
}

// Tracking loop

#ifdef _GPUCODE

CLKERNEL void Block_track(
                 CLGLOBAL value_t *data, CLGLOBAL Particle *particles,
                 uint64_t blockid, uint64_t nturn, uint64_t npart,
                 uint64_t elembyelemid, uint64_t turnbyturnid){
   uint64_t nelem    = Block_get_nelen(data, blockid);
   CLGLOBAL uint64_t *elemids = Block_get_elemids(data, blockid);
   uint64_t i_part = get_global_id(0);
   uint64_t elembyelemoff=0;
   uint64_t turnbyturnoff=0;
   for (int i_turn=0; i_turn< nturn; i_turn++){
     for (int i_elem=0; i_elem< nelem; i_elem++) {
       if (elembyelemid>0){
         elembyelemoff=elembyelemid +
                      sizeof(Particle)/8 * npart * i_turn +
                      sizeof(Particle)/8 * npart * nturn  * i_elem ;
//            printf("%lu \n",elembyelemoff);
       }
       if (turnbyturnid>0){
         turnbyturnoff=turnbyturnid +
                        sizeof(Particle)/8 * npart * i_turn;
       }
      track_single(data, particles, elemids,
                   i_part, i_elem, elembyelemoff, turnbyturnoff);
    }
    if (particles[i_part].state>=0) {
      particles[i_part].turn++;
    }
  }
}

#else

#include <stdio.h>


int Block_track(value_t *data, Beam *beam,
                uint64_t blockid, uint64_t nturn,
                uint64_t elembyelemid, uint64_t turnbyturnid){
   uint64_t nelem    = Block_get_nelen(data, blockid);
   uint64_t *elemids = Block_get_elemids(data, blockid);
   uint64_t npart=beam->npart;
   uint64_t elembyelemoff=0;
   uint64_t turnbyturnoff=0;
   for (int i_turn=0; i_turn< nturn; i_turn++) {
     for (int i_elem=0; i_elem< nelem; i_elem++) {
       for (uint64_t i_part=0; i_part < npart; i_part++){
          if (elembyelemid>0){
            elembyelemoff=elembyelemid +
                         sizeof(Particle)/8 * npart * i_turn +
                         sizeof(Particle)/8 * npart * nturn  * i_elem ;
//            printf("cpu %lu \n",elembyelemoff);
          }
          if (turnbyturnid>0){
            turnbyturnoff=turnbyturnid +
                         sizeof(Particle)/8 * npart * i_turn;
//            printf("%lu \n",turnbyturnoff);
          }
          track_single(data, beam->particles, elemids,
                       i_part, i_elem, elembyelemoff, turnbyturnoff);
       }
     }
     for (uint64_t i_part=0; i_part < npart; i_part++){
       if (beam->particles[i_part].state >= 0)
                 beam->particles[i_part].turn++;
       }
     }
   return 1;
}

#endif





/*================================
 * buffer handling
*/


CLGLOBAL block_t* block_initialize(unsigned int size) {
    CLGLOBAL block_t *block = malloc(sizeof(CLGLOBAL block_t*));
    if(!size) {
        size = 512;
    }
    CLGLOBAL value_t *data = malloc(sizeof(CLGLOBAL value_t) * size);
    if(!data) {
        return NULL;
    }
    block->size = sizeof(CLGLOBAL value_t) * size;
    block->last = 0;
    block->data = data;
    return block;
}

void block_reshape(CLGLOBAL block_t *block, unsigned int n) {
    if(block->last+(sizeof(CLGLOBAL value_t)*n) >= block->size) {
        CLGLOBAL value_t *ndata = realloc(block->data, (block->size+(sizeof(CLGLOBAL value_t)*n))*2);
        if(!ndata) {
            return;
        }
        block->size = (block->size+(sizeof(CLGLOBAL value_t)*n))*2;
        block-> data = ndata;
    }
}

void block_clean(CLGLOBAL block_t *block) {
    free(block->data);
    free(block);
}

CLGLOBAL block_t* block_add_drift(CLGLOBAL block_t *block, uint64_t *offsets, uint64_t nel, double length) {
    block_reshape(block, 2);
    offsets[nel] = block->last;
    block->data[block->last++].u64 = DriftID;
    block->data[block->last++].f64 = length;
    return block;
}

CLGLOBAL block_t* block_add_cavity(CLGLOBAL block_t *block, uint64_t *offsets, uint64_t nel, double volt, double freq, double lag) {
    block_reshape(block, 4);
    offsets[nel] = block->last;
    block->data[block->last++].u64 = CavityID;
    block->data[block->last++].f64 = volt;
    block->data[block->last++].f64 = freq;
    block->data[block->last++].f64 = lag/180.0 * M_PI;
    return block;
}

CLGLOBAL block_t* block_add_align(CLGLOBAL block_t *block, uint64_t *offsets, uint64_t nel, double tilt, double dx, double dy) {
    block_reshape(block, 5);
    offsets[nel] = block->last;
    block->data[block->last++].u64 = AlignID;
    block->data[block->last++].f64 = cos(tilt/180.0 * M_PI);
    block->data[block->last++].f64 = sin(tilt/180.0 * M_PI);
    block->data[block->last++].f64 = dx;
    block->data[block->last++].f64 = dy;
    return block;
}

CLGLOBAL block_t* block_add_multipole(CLGLOBAL block_t *block, uint64_t *offsets, uint64_t nel, double *knl, unsigned int knl_len, double *ksl, unsigned int ksl_len, double length, double hxl, double hyl) {
    double bal[(knl_len >= ksl_len ? 2*knl_len : 2*ksl_len)];
    int i = 0;
    for(; i < knl_len || i < ksl_len; i++) {
        if(i < knl_len) {
            bal[2*i] = knl[i];
        } else {
            bal[2*i] = 0;
        }
        if(i < ksl_len) {
            bal[2*i+1] = ksl[i];
        } else {
            bal[2*i+1] = 0;
        }
    }
    uint64_t order = i-1;
    for(int j = 0, fact = 1; j < i; j++, fact *= fact+1) {
        bal[2*j] /= fact;
        bal[2*j+1] /= fact;
    }
    block_reshape(block, 5+(2*i));
    offsets[nel] = block->last;
    block->data[block->last++].u64 = MultipoleID;
    block->data[block->last++].u64 = order;
    block->data[block->last++].f64 = length;
    block->data[block->last++].f64 = hxl;
    block->data[block->last++].f64 = hyl;
    for(int j = 0; j < i; j++) {
        block->data[block->last++].f64 = bal[j];
    }
    return block;
}

CLGLOBAL block_t* block_add_block(CLGLOBAL block_t *block, uint64_t *offsets, uint64_t nel) {
    block_reshape(block, nel+2);
    block->data[block->last++].u64 = BlockID;
    block->data[block->last++].u64 = nel;
    for(uint64_t i = 0; i < nel; i++) {
        block->data[block->last++].u64 = offsets[i++];
    }
    return block;
}
