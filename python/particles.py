#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cobjects import CBuffer, CObject, CField

class Particles( CObject ):
    _typeid       = 1
    num_particles = CField(  0, 'uint64', alignment=8 )
    q0            = CField(  1, 'real',  length='num_particles', default=0.0, pointer=True, alignment=8 )
    mass0         = CField(  2, 'real',  length='num_particles', default=0.0, pointer=True, alignment=8 )
    beta0         = CField(  3, 'real',  length='num_particles', default=0.0, pointer=True, alignment=8 )
    gamma0        = CField(  4, 'real',  length='num_particles', default=0.0, pointer=True, alignment=8 )
    p0c           = CField(  5, 'real',  length='num_particles', default=0.0, pointer=True, alignment=8 )
    s             = CField(  6, 'real',  length='num_particles', default=0.0, pointer=True, alignment=8 )
    x             = CField(  7, 'real',  length='num_particles', default=0.0, pointer=True, alignment=8 )
    y             = CField(  8, 'real',  length='num_particles', default=0.0, pointer=True, alignment=8 )
    px            = CField(  9, 'real',  length='num_particles', default=0.0, pointer=True, alignment=8 )
    py            = CField( 10, 'real',  length='num_particles', default=0.0, pointer=True, alignment=8 )
    zeta          = CField( 11, 'real',  length='num_particles', default=0.0, pointer=True, alignment=8 )
    psigma        = CField( 12, 'real',  length='num_particles', default=0.0, pointer=True, alignment=8 )
    delta         = CField( 13, 'real',  length='num_particles', default=0.0, pointer=True, alignment=8 )
    rpp           = CField( 14, 'real',  length='num_particles', default=1.0, pointer=True, alignment=8 )
    rvv           = CField( 15, 'real',  length='num_particles', default=1.0, pointer=True, alignment=8 )
    chi           = CField( 16, 'real',  length='num_particles', default=0.0, pointer=True, alignment=8 )
    particle      = CField( 17, 'int64', length='num_particles', default=-1,  pointer=True, alignment=8 )
    at_element    = CField( 18, 'int64', length='num_particles', default=-1,  pointer=True, alignment=8 )
    at_turn       = CField( 19, 'int64', length='num_particles', default=-1,  pointer=True, alignment=8 )
    state         = CField( 20, 'int64', length='num_particles', default=-1,  pointer=True, alignment=8 )

#end: