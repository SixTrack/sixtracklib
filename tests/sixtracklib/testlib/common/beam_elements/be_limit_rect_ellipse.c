#include "sixtracklib/testlib/common/beam_elements/be_limit_rect_ellipse.h"
#include "sixtracklib/common/be_limit/be_limit_rect_ellipse.h"

void NS(LimitRectEllipse_print)( SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( limit != SIXTRL_NULLPTR ) )
    {
        fprintf( fp, "|limit_rect_ellips| half-axis x = %+21.18f m\r\n"
                "                  | half-axis y = %+21.18f m\r\n"
                "                  | x limit     = %+21.18f m\r\n"
                "                  | y limit     = %+21.18f m\r\n",
                NS(LimitRectEllipse_a)( limit ),
                NS(LimitRectEllipse_b)( limit ),
                NS(LimitRectEllipse_max_x)( limit ),
                NS(LimitRectEllipse_max_y)( limit ) );
    }
}
