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

//Drift

double Drift_get_length(CLGLOBAL value_t *data, uint64_t elemid){
    return data[elemid + 1].f64;
}

//DriftExact

double driftexact_get_length(CLGLOBAL value_t *data, uint64_t elemid){
    return data[elemid + 1].f64;
}

//Multipole


//long int Multipole_get_order(CLGLOBAL value_t *data, uint64_t elemid){
//    return data[elemid + 1].i64;
//}
//
//double Multipole_get_l(CLGLOBAL value_t *data, uint64_t elemid){
//    return data[elemid + 2].f64;
//}
//
//double Multipole_get_hxl(CLGLOBAL value_t *data, uint64_t elemid){
//    return data[elemid + 3].f64;
//}
//
//double Multipole_get_hyl(CLGLOBAL value_t *data, uint64_t elemid){
//    return data[elemid + 4].f64;
//}
//
//CLGLOBAL double* Multipole_get_bal(CLGLOBAL value_t *data, uint64_t elemid){
//    return &data[elemid + 5].f64;
//}

//Cavity
double Cavity_get_volt(CLGLOBAL value_t *data, uint64_t elemid){
    return data[elemid + 1].f64;
}

double Cavity_get_freq(CLGLOBAL value_t *data, uint64_t elemid){
    return data[elemid + 2].f64;
}

double Cavity_get_lag(CLGLOBAL value_t *data, uint64_t elemid){
    return data[elemid + 3].f64;
}

//Align


double Align_get_cz(CLGLOBAL value_t *data, uint64_t elemid){
    return data[elemid + 1].f64;
}

double Align_get_sz(CLGLOBAL value_t *data, uint64_t elemid){
    return data[elemid + 2].f64;
}

double Align_get_dx(CLGLOBAL value_t *data, uint64_t elemid){
    return data[elemid + 3].f64;
}

double Align_get_dy(CLGLOBAL value_t *data, uint64_t elemid){
    return data[elemid + 3].f64;
}



// Tracking signle

//#ifndef _GPUCODE
//#include <stdio.h>
//#endif
//#ifndef _GPUCODE
//printf("%lu %d\n",elembyelemid,i_attr);
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
       if (elembyelemoff>0){
         uint64_t dataoff=elembyelemoff+sizeof(Particle)/8 * i_part;
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
//                           Drift_get_length(data,elemid)        );
           break;
           case MultipoleID:
                Multipole_track(p, (CLGLOBAL Multipole*) elem);
//                               Multipole_get_order(data,elemid),
//                               Multipole_get_l(data,elemid),
//                               Multipole_get_hxl(data,elemid),
//                               Multipole_get_hyl(data,elemid),
//                               Multipole_get_bal(data,elemid)    );
           break;
           case CavityID:
                Cavity_track(p,
                               Cavity_get_volt(data,elemid),
                               Cavity_get_freq(data,elemid),
                               Cavity_get_lag(data,elemid)       );
           break;
           case AlignID:
                Align_track(p,
                               Align_get_cz(data,elemid),
                               Align_get_sz(data,elemid),
                               Align_get_dx(data,elemid),
                               Align_get_dy(data,elemid)    );
           break;
           case IntegerID: break;
           case DoubleID: break;
           case BlockID: break;
           case DriftExactID: break;
       }
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
                        sizeof(Particle)/8 * npart * nelem * nturn * i_elem +
                        sizeof(Particle)/8 * npart * nelem * i_turn ;
       }
       if (turnbyturnid>0){
         turnbyturnoff=turnbyturnid +
                        sizeof(Particle)/8 * npart * i_turn;
       }
      track_single(data, particles, elemids,
                   i_part, i_elem, elembyelemoff, turnbyturnoff);
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
            printf("%lu \n",elembyelemoff);
          }
          if (turnbyturnid>0){
            turnbyturnoff=turnbyturnid +
                         sizeof(Particle)/8 * npart * i_turn;
            printf("%lu \n",turnbyturnoff);
          }
          track_single(data, beam->particles, elemids,
                       i_part, i_elem, elembyelemoff, turnbyturnoff);
       }
     }
   }
   return 1;
}

#endif






