#include "block.h"

#include "particle.h"
#include "track.c"


// Data management

type_t get_type(CLGLOBAL value_t *elem) { return (type_t) elem[0].u64; }

//Drift

double Drift_get_length(CLGLOBAL value_t *elem){ return elem[1].f64;}

//DriftExact

double DriftExact_get_length(CLGLOBAL value_t *elem){ return elem[1].f64;}

//Multipole

int64_t Multipole_get_order(CLGLOBAL value_t *elem){ return elem[1].i64;}
double  Multipole_get_l    (CLGLOBAL value_t *elem){ return elem[2].f64;}
double  Multipole_get_hxl  (CLGLOBAL value_t *elem){ return elem[3].f64;}
double  Multipole_get_hyl  (CLGLOBAL value_t *elem){ return elem[4].f64;}
CLGLOBAL double* Multipole_get_bal(CLGLOBAL value_t *elem){
    return &elem[6].f64;
}

//Cavity
double Cavity_get_volt(CLGLOBAL value_t *elem){ return elem[1].f64;}
double Cavity_get_freq(CLGLOBAL value_t *elem){ return elem[2].f64;}
double Cavity_get_lag (CLGLOBAL value_t *elem){ return elem[3].f64;}

//Align

double Align_get_cz(CLGLOBAL value_t *elem){return elem[1].f64;}
double Align_get_sz(CLGLOBAL value_t *elem){return elem[2].f64;}
double Align_get_dx(CLGLOBAL value_t *elem){return elem[3].f64;}
double Align_get_dy(CLGLOBAL value_t *elem){return elem[4].f64;}


// Tracking loop

#ifdef _GPUCODE


void track_single(Particles *particles, uint64_t partid, CLGLOBAL value_t * elem){
    if (particles->state[partid] >= 0 ) {
        enum type_t typeid = get_type(elem);
        //           _DP("Block_track: elemid=%zu typedid=%u\n",elemid,typeid);
        switch (typeid) {
            case DriftID:
                Drift_track(particles, partid,
                        Drift_get_length(elem)        );
                break;
            case DriftExactID:
                Drift_track(particles, partid,
                        Drift_get_length(elem)        );
                break;
            case MultipoleID:
                Multipole_track(particles, partid,
                        Multipole_get_order(elem),
                        Multipole_get_l(elem),
                        Multipole_get_hxl(elem),
                        Multipole_get_hyl(elem),
                        Multipole_get_bal(elem)    );
                break;
            case CavityID:
                Cavity_track(particles, partid,
                        Cavity_get_volt(elem),
                        Cavity_get_freq(elem),
                        Cavity_get_lag(elem)       );
                break;
            case AlignID:
                Align_track(particles, partid,
                        Align_get_cz(elem),
                        Align_get_sz(elem),
                        Align_get_dx(elem),
                        Align_get_dy(elem)    );
                break;
        }//end switch
    }//end if state
}

CLKERNEL void Block_track(CLGLOBAL value_t   *elems,
        CLGLOBAL uint64_t  *elemids,
        uint64_t nelems,
        uint64_t nturns,
        CLGLOBAL value_t *particles_p, //Particles
        CLGLOBAL value_t *elembyelem_p,  //ElembyElem
        CLGLOBAL value_t *turnbyturn_p)  //TurnbyTurn
{
    CLGLOBAL value_t * elem;
    uint64_t elemid;
    uint64_t partid = get_global_id(0);

    Particles*  particles = (Particles*)  particles_p;

    printf( "%s test\n","test" );

    bool elembyelem_flag = (elembyelem_p[0].i64 >= 0);
    bool turnbyturn_flag = (turnbyturn_p[0].i64 >= 0);

    Particles* elembyelem = (Particles*) elembyelem_p;
    Particles* turnbyturn = (Particles*) turnbyturn_p;

    Particles_unpack(  particles);
    if (elembyelem_flag) {
        if (partid==0) Particles_unpack( elembyelem );
        Particles_copy(particles, elembyelem, partid, partid);
    };
    if (turnbyturn_flag) {
        if (partid==0) Particles_unpack( turnbyturn );
        Particles_copy(particles, turnbyturn, partid, partid);
    };

    uint64_t nparts=particles->npart;
    uint64_t tbt=nparts;
    uint64_t ebe=nparts;

    for (int jj = 0; jj < nturns; jj++) {
        for (int ii = 0; ii < nelems; ii++) {
            elemid = elemids[ii];
            elem   = elems+elemid;
            track_single(particles,partid,elem);
            if (elembyelem_flag)
                ebe+=nparts;
                Particles_copy(particles, elembyelem, partid, ebe+partid);
        }  //end elem loop
        if (turnbyturn_flag)
            tbt+=nparts;
            Particles_copy(particles, turnbyturn, partid, tbt+partid);
    }  //end turn loop
}

#else

int Block_track(value_t *elems, Beam *beam, uint64_t blockid){
    uint64_t nelem    = Block_get_nelen(elems, blockid);
    uint64_t *elemids = Block_get_elemids(elems, blockid);
    uint64_t elemid;
    for (int ii=0; ii< nelem; ii++) {
        elemid=elemids[ii];
        for (uint64_t partid=0; partid < beam->npart; partid++){
            track_single(elems, beam->particles, elemid, partid,0);
        };
    }
    return 1;
}

#endif

