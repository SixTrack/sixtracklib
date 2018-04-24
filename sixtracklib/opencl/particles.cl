NS(Particles)* NS(Particles_unpack)(NS(Particles)* p, __global unsigned char* mem_begin ) 
{
    p->q0     = vloadn( (__constant double *) p + pp[1].u64  );
    p->mass0  = ( (__constant double *) p + pp[2].u64 );
    p->beta0  = ( (__constant double *) p + pp[3].u64 );
    p->gamma0 = ( (__constant double *) p + pp[4].u64 );
    p->p0c    = ( (__constant double *) p + pp[5].u64 );
    p->partid = ( (__constant int64_t*) p + pp[6].u64 );
    p->elemid = ( (__constant int64_t*) p + pp[7].u64 );
    p->turn   = ( (__constant int64_t*) p + pp[8].u64 );
    p->state  = ( (__constant int64_t*) p + pp[9].u64 );
    p->s      = ( (__constant double *) p + pp[10].u64 );
    p->x      = ( (__constant double *) p + pp[11].u64 );
    p->px     = ( (__constant double *) p + pp[12].u64 );
    p->y      = ( (__constant double *) p + pp[13].u64 );
    p->py     = ( (__constant double *) p + pp[14].u64 );
    p->sigma  = ( (__constant double *) p + pp[15].u64 );
    p->psigma = ( (__constant double *) p + pp[16].u64 );
    p->delta  = ( (__constant double *) p + pp[17].u64 );
    p->rpp    = ( (__constant double *) p + pp[18].u64 );
    p->rvv    = ( (__constant double *) p + pp[19].u64 );
    p->chi    = ( (__constant double *) p + pp[20].u64 );
    return (Particles*) p;
}

__kernel void NS(Block_track( 
        __constant unsigned char* memory,
        __constant uint64_t  *elemids,
        uint64_t nelems,
        uint64_t nturns,
        __global NS(Particles)* particles, //Particles
        __global NS(Particles)* elembyelem_p,  //ElembyElem
        __global NS(Particles)* turnbyturn_p)  //TurnbyTurn
{
    CLGLOBAL value_t * elem;
    uint64_t elemid;
    uint64_t partid = get_global_id(0);

    Particles*  particles = (Particles*)  particles_p;

    //printf( "beta0[%d] %g\n",partid, particles->beta0[partid]);

    bool elembyelem_flag = (elembyelem_p[0].i64 != 0);
    bool turnbyturn_flag = (turnbyturn_p[0].i64 != 0);

    Particles* elembyelem = (Particles*) elembyelem_p;
    Particles* turnbyturn = (Particles*) turnbyturn_p;

    if (turnbyturn_flag) {
        Particles_copy(particles, turnbyturn, partid, partid);
    };

    //printf( "tbt->beta0[%d] %g\n",partid, turnbyturn->beta0[partid]);

    uint64_t nparts=particles->npart;
    uint64_t tbt=nparts;
    uint64_t ebe=0;
    //printf("%g %g %g\n",nparts,tbt,ebe);

    for (int jj = 0; jj < nturns; jj++) {
        for (int ii = 0; ii < nelems; ii++) {
            elemid = elemids[ii];
            //printf("elemid %u\n",elemid);
            elem   = elems+elemid;
            track_single(particles,partid,elem);
            if (elembyelem_flag){
                Particles_copy(particles, elembyelem, partid, ebe+partid);
                ebe+=nparts;
            }
        }  //end elem loop
        if (turnbyturn_flag){
            Particles_copy(particles, turnbyturn, partid, tbt+partid);
            tbt+=nparts;
        }
    }  //end turn loop
}
