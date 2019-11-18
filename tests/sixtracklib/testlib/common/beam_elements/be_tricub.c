
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
            "|tricub          | x              = %+20.12f\r\n"
            "                 | y              = %+20.12f\r\n"
            "                 | z              = %+20.12f\r\n"
            "                 | length         = %+20.12f\r\n"
            "                 | table_addr     = %20lu\r\n",
            NS(TriCub_x)( e ), NS(TriCub_y)( e ), NS(TriCub_z)( e ),
            NS(TriCub_length)( e ),
            ( long unsigned )NS(TriCub_data_addr)( e ) );
    }
}

#endif /* !defined( _GPUCODE ) */

/* end: /sixtracklib/testlib/common/beam_elements/be_tricub.c */
