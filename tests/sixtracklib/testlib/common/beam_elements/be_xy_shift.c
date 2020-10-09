#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/testlib/common/beam_elements/be_xy_shift.h"
    #include "sixtracklib/common/be_xyshift/be_xyshift.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

void NS(XYShift_print)( SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xyshift )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( xyshift != SIXTRL_NULLPTR ) )
    {
        fprintf( fp,
            "|xy_shift         | dx        = %+20.14f m\r\n"
            "                  | dy        = %+20.14f m\r\n",
            NS(XYShift_dx)( xyshift ), NS(XYShift_dy)( xyshift ) );
    }
}
