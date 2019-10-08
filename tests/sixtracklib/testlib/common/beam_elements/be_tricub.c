
#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include "sixtracklib/testlib/common/beam_elements/be_tricub.h"
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_tricub/be_tricub.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

void NS(TriCub_print)( SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT e )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( e != SIXTRL_NULLPTR ) )
    {
        printf( "|tricub          | nx             = %+20ld\r\n"
                "                 | ny             = %+20ld\r\n"
                "                 | nz             = %+20ld\r\n"
                "                 | x0             = %+20.12f\r\n"
                "                 | y0             = %+20.12f\r\n"
                "                 | z0             = %+20.12f\r\n"
                "                 | dx             = %+20.12f\r\n"
                "                 | dy             = %+20.12f\r\n"
                "                 | dz             = %+20.12f\r\n",
                NS(TriCub_get_nx)( e ), NS(TriCub_get_ny)( e ),
                NS(TriCub_get_nz)( e ),
                NS(TriCub_get_x0)( e ), NS(TriCub_get_y0)( e ),
                NS(TriCub_get_z0)( e ),
                NS(TriCub_get_dx)( e ), NS(TriCub_get_dy)( e ),
                NS(TriCub_get_dz)( e ) );
    }
}

#endif /* !defined( _GPUCODE ) */

/* end: /sixtracklib/testlib/common/beam_elements/be_tricub.c */
