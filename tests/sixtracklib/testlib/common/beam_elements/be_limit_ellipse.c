#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include "sixtracklib/testlib/common/beam_elements/be_limit_ellipse.h"
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */
    
#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_limit/definitions.h"
    #include "sixtracklib/common/be_limit/be_limit_ellipse.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

void NS(LimitRect_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamMonitor) *const 
        SIXTRL_RESTRICT limit_rect )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( limit_rect != SIXTRL_NULLPTR ) )
    {
        fprintf( fp, 
                "|limit_ellipse    | origin x    = %+16.12f m;\r\n"
                "                  | origin y    = %+16.12f m;\r\n"
                "                  | half-axis x = %+16.12f m;\r\n"
                "                  | half-axis y = %+16.12f m;\r\n",
                NS(LimitEllipse_get_origin_x)( limit_ellipse ),
                NS(LimitEllipse_get_origin_y)( limit_ellipse ),
                NS(LimitEllipse_get_x_half_axis)( limit_ellipse ),
                NS(LimitEllipse_get_y_half_axis)( limit_ellipse ) );
    }
    
    return;
}

#endif /* !defined( _GPUCODE ) */

/* end: tests/sixtracklib/testlib/common/beam_elements/be_limit_rect.c */
