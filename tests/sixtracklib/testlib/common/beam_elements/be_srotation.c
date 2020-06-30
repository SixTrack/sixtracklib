#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/testlib/common/beam_elements/be_srotation.h"
    #include "sixtracklib/common/be_srotation/be_srotation.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


void NS(SRotation_print)( SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srot )
{
    if( ( srot != SIXTRL_NULLPTR ) && ( fp != SIXTRL_NULLPTR ) )
    {
        fprintf( fp,
            "|srotation        | angle    = %+20.16f deg  ( %+20.17f rad )\r\n"
            "                  | cos_z    = %+20.18f;\r\n"
            "                  | sin_z    = %+20.18f;\r\n",
            NS(SRotation_angle_deg)( srot ), NS(SRotation_angle)( srot ),
            NS(SRotation_cos_angle)( srot ), NS(SRotation_sin_angle)( srot ) );
    }
}
