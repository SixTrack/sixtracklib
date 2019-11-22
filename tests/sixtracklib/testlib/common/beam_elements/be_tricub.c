
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
        fprintf( fp,
            "|tricub          | x_shift             = %+20.12f\r\n"
            "                 | y_shift             = %+20.12f\r\n"
            "                 | zeta_shift          = %+20.12f\r\n"
            "                 | dipolar_kick_px     = %+20.12f\r\n"
            "                 | dipolar_kick_py     = %+20.12f\r\n"
            "                 | dipolar_kick_pdelta = %+20.12f\r\n"
            "                 | length              = %+20.12f\r\n"
            "                 | table_addr          = %20lu\r\n",
            NS(TriCub_x_shift)( e ), NS(TriCub_y_shift)( e ),
            NS(TriCub_zeta_shift)( e ), NS(TriCub_dipolar_kick_px)( e ),
            NS(TriCub_dipolar_kick_py)( e ), NS(TriCub_dipolar_kick_delta)( e ),
            NS(TriCub_length)( e ),
            ( long unsigned )NS(TriCub_data_addr)( e ) );
    }
}

#endif /* !defined( _GPUCODE ) */

/* end: /sixtracklib/testlib/common/beam_elements/be_tricub.c */
