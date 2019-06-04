#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include "sixtracklib/testlib/common/beam_elements/be_dipedge.h"
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */
    
#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_limit/definitions.h"
    #include "sixtracklib/common/be_dipedge/be_dipedge.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

void NS(DipoleEdge_print)( SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    if( dipedge != SIXTRL_NULLPTR )
    {
        fprintf( fp,
                "|dipole_edge      | r21      = %+16.12f m^-1;\r\n"
                "                  | r43      = %+16.12f m^-1;\r\n",
                NS(DipoleEdge_get_r21)( dipedge ),
                NS(DipoleEdge_get_r43)( dipedge ) );
    }
    
    return;
}

#endif /* !defined( _GPUCODE ) */

/* end: tests/sixtracklib/testlib/common/beam_elements/be_limit_rect.c */
