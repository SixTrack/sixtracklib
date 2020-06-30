#include "sixtracklib/testlib/common/beam_elements/beam_elements.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/beam_elements.h"

/* ========================================================================= */
/* =====           BEAM_ELEMENTS_TOOLS IMPLEMENTATION                  ===== */
/* ========================================================================= */

void NS(BeamElement_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const struct NS(Object) *const SIXTRL_RESTRICT
        be_info )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( be_info != SIXTRL_NULLPTR ) )
    {
        NS(object_type_id_t) const type_id = NS(Object_get_type_id)( be_info );
        NS(buffer_addr_t) const addr = NS(Object_get_begin_addr)( be_info );

        switch( type_id )
        {
            case NS(OBJECT_TYPE_DRIFT):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(Drift) const* ptr_t;
                NS(Drift_print)( fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_DRIFT_EXACT):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(DriftExact) const* ptr_t;
                NS(DriftExact_print)( fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_MULTIPOLE):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(Multipole) const* ptr_t;
                NS(Multipole_print)( fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_RF_MULTIPOLE):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(RFMultipole) const* ptr_t;
                NS(RFMultipole_print)( fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_CAVITY):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(Cavity) const* ptr_t;
                NS(Cavity_print)( fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_XYSHIFT):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(XYShift) const* ptr_t;
                NS(XYShift_print)( fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_SROTATION):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(SRotation) const* ptr_t;
                NS(SRotation_print)( fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_BEAM_BEAM_4D):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D) const* ptr_t;
                NS(BeamBeam4D_print)( fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_BEAM_BEAM_6D):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D) const* ptr_t;
                NS(BeamBeam6D_print)( fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_SC_COASTING):
            {
                typedef NS(SpaceChargeCoasting) sc_elem_t;
                typedef SIXTRL_BE_ARGPTR_DEC sc_elem_t const* ptr_t;
                NS(SpaceChargeCoasting_print)( fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_SC_QGAUSSIAN_PROF):
            {
                typedef NS(SpaceChargeQGaussianProfile) sc_elem_t;
                typedef SIXTRL_BE_ARGPTR_DEC sc_elem_t const* ptr_t;
                NS(SpaceChargeQGaussianProfile_print)(
                    fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_SC_INTERPOLATED_PROF):
            {
                typedef NS(SpaceChargeInterpolatedProfile) sc_elem_t;
                typedef SIXTRL_BE_ARGPTR_DEC sc_elem_t const* ptr_t;
                NS(SpaceChargeInterpolatedProfile_print)(
                    fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_BEAM_MONITOR):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor) const* ptr_t;
                NS(BeamMonitor_print)( fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_LIMIT_RECT):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(LimitRect) const* ptr_t;
                NS(LimitRect_print)( fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_LIMIT_ELLIPSE):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse) const* ptr_t;
                NS(LimitEllipse_print)( fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_LIMIT_RECT_ELLIPSE):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse) const* ptr_t;
                NS(LimitRectEllipse_print)( fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_DIPEDGE):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge) const* ptr_t;
                NS(DipoleEdge_print)( fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            case NS(OBJECT_TYPE_TRICUB):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(TriCub) const* ptr_t;
                NS(TriCub_print)( fp, ( ptr_t )( uintptr_t )addr );
                break;
            }

            default:
            {
                printf( "|unknown          | type_id  = %3d\r\n"
                        "                  | size     = %8lu bytes\r\n"
                        "                  | addr     = %16p\r\n",
                        ( int )type_id, NS(Object_get_size)( be_info ),
                        ( void* )( uintptr_t )addr );
            }
        };
    }

    return;
}
