#include "block.h"

#define _CUDA_HOST_DEVICE_
#define DATA_PTR_IS_OFFSET
#include "../common/track.c"

//Data managemen

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
   CLGLOBAL value_t * elem;
   uint64_t elemid;
   if (p->state >= 0 ) {
       elemid=elemids[i_elem];
       if ( (turnbyturnoff>0) && (i_elem==0) ){
         uint64_t dataoff=turnbyturnoff+sizeof(Particle)/8 * i_part;
         for (unsigned i_attr=0;i_attr<sizeof(Particle)/8;i_attr++) {
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
           case DriftExactID:
                DriftExact_track(p, (CLGLOBAL DriftExact*) elem);
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
           case LinMapID:
                LinMap_track(p, (CLGLOBAL LinMap_data*) elem);
           break;
           case BB4DID:
                BB4D_track(p, (CLGLOBAL BB4D_data *) elem);
           break;
           case IntegerID: break;
           case DoubleID: break;
           case BlockID: break;
       }
       if (elembyelemoff>0){
         uint64_t dataoff=elembyelemoff+sizeof(Particle)/8 * i_part;
         for (unsigned i_attr=0;i_attr<sizeof(Particle)/8;i_attr++) {
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

//#include <stdio.h>
int Block_track(value_t *data, Beam *beam,
                uint64_t blockid, uint64_t nturn,
                uint64_t elembyelemid, uint64_t turnbyturnid){
   uint64_t nelem    = Block_get_nelen(data, blockid);
   uint64_t *elemids = Block_get_elemids(data, blockid);
   uint64_t npart=beam->npart;
   uint64_t elembyelemoff=0;
   uint64_t turnbyturnoff=0;
   for (uint64_t i_turn=0; i_turn< nturn; i_turn++) {
     for (uint64_t i_elem=0; i_elem< nelem; i_elem++) {
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


