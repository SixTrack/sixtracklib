#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include "sixtracklib/testlib/common/beam_elements/be_limit_rect.h"
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */
    
#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_limit/definitions.h"
    #include "sixtracklib/common/be_limit/be_limit_rect.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

void NS(LimitRect_print)( SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT limit_rect )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( limit_rect != SIXTRL_NULLPTR ) )
    {
        fprintf( fp,
                 "|limit_rect       | min_x    = %+16.12f m;\r\n"
                 "                  | max_x    = %+16.12f m;\r\n"
                 "                  | min_y    = %+16.12f m;\r\n"
                 "                  | max_y    = %+16.12f m;\r\n",
                 NS(LimitRect_get_min_x)( limit_rect ),
                 NS(LimitRect_get_max_x)( limit_rect ),
                 NS(LimitRect_get_min_y)( limit_rect ),
                 NS(LimitRect_get_max_y)( limit_rect ) );
    }
    
    return;
}

#endif /* !defined( _GPUCODE ) */

/* end: tests/sixtracklib/testlib/common/beam_elements/be_limit_rect.c */
