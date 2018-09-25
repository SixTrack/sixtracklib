#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cobjects import CBuffer, CObject, CField
import numpy as np


class Drift( CObject ):
    _typeid =  2
    length  = CField( 0, 'real', default=0.0, alignment=8 )

class DriftExact( CObject ):
    _typeid = 3
    length  = CField( 0, 'real', default=0.0, alignment=8 )

class MultiPole( CObject ):
    _typeid = 4
    order   = CField( 0, 'int64',   default=0,    alignment=8 )
    length  = CField( 1, 'real',    default=0.0,  alignment=8 )
    hxl     = CField( 2, 'real',    default=0.0,  alignment=8 )
    hyl     = CField( 3, 'real',    default=0.0,  alignment=8 )
    bal     = CField( 4, 'real',    default=0.0,
                      length='2 * order + 2', pointer=True, alignment=8 )

    def _factorial( self, x ):
        if  not isinstance( x, int ):
            return 0
        return ( x > 0 ) and ( x * self._factorial( x - 1 ) ) or 1

    def __init__( self, order=None, knl=None, ksl=None, bal=None, **kwargs ):

        if  bal is None and ( not( knl is None ) or not( ksl is None ) ):
            if knl   is None: knl = []
            if ksl   is None: ksl = []
            if order is None: order = 0

            n = max( ( order + 1 ), max( len( knl ), len( ksl ) ) )
            _knl = np.array( knl )
            nknl = np.zeros( n, dtype=_knl.dtype )
            nknl[:len(knl)] = knl
            knl = nknl
            del( _knl )

            _ksl = np.array( ksl )
            nksl = np.zeros( n, dtype=_ksl.dtype )
            nksl[:len(ksl)] = ksl
            ksl = nksl
            del( _ksl )

            assert( n > 0 )
            order = n - 1

            bal = np.zeros( 2 * order + 2 )
            assert( len( knl ) == len( ksl ) )

            for ii in range( 0, len( knl ) ):
                inv_factorial = 1.0 / float( self._factorial( ii ) )
                jj = 2 * ii
                bal[ jj     ] = knl[ ii ] * inv_factorial
                bal[ jj + 1 ] = ksl[ ii ] * inv_factorial

        elif not( bal is None ) and bal and \
             len( bal ) > 2 and ( ( len( bal ) % 2 ) == 0 ):

            order = ( len( bal ) - 2 ) / 2
            assert( order > 0 )

        elif bal is None and knl is None and ksl is None and \
             not( order is None ) and order > 0:
             bal = np.zeros( 2 * order + 2 )


        if  not( bal is None or order is None ):
            CObject.__init__( self, bal=bal, order=order, **kwargs )
        else:
            COBject.__init__( self, bal=[], order=0, **kwargs )


class Cavity( CObject ):
    _typeid = 5
    voltage   = CField( 0, 'real', default=0.0,  alignment=8 )
    frequency = CField( 1, 'real', default=0.0,  alignment=8 )
    lag       = CField( 2, 'real', default=0.0,  alignment=8 )

class XYShift( CObject ):
    _typeid = 6
    dx      = CField( 0, 'real',   default=0.0,  alignment=8 )
    dy      = CField( 1, 'real',   default=0.0,  alignment=8 )

class SRotation( CObject ):
    _typeid = 7
    cos_z   = CField( 0, 'real',   default=1.0,  alignment=8 )
    sin_z   = CField( 1, 'real',   default=0.0,  alignment=8 )

class BeamBeam4D( CObject ):
    pass

class BeamBeam6D( CObject ):
    pass

