#include "sixtracklib/testlib/common/beam_elements/be_multipole.h"
#include "sixtracklib/common/be_multipole/be_multipole.h"

void NS(Multipole_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT mp )
{
    if( ( mp != SIXTRL_NULLPTR ) && ( fp != SIXTRL_NULLPTR ) )
    {
        NS(multipole_order_t) const order = NS(Multipole_order)( mp );

        printf( "|multipole        | order    = %3ld;\r\n"
                "                  | length   = %+20.12f m;\r\n"
                "                  | hxl      = %+20.12f m;\r\n"
                "                  | hyl      = %+20.12f m;\r\n",
                ( long int )order, NS(Multipole_length)( mp ),
                NS(Multipole_hxl)( mp ), NS(Multipole_hyl)( mp ) );

        printf( "                  |"
                    "    idx"
                    "                  knl" "                  ksl\r\n" );

        if( order >= ( NS(multipole_order_t) )0 )
        {
            NS(multipole_order_t) ii = ( NS(multipole_order_t) )0;
            for( ; ii <= order ; ++ii )
            {
                printf( "                  | %6ld %+20.12f %+20.12f\r\n",
                        ( long int )ii,
                        NS(Multipole_knl)( mp, ii ),
                        NS(Multipole_ksl)( mp, ii ) );
            }
        }
        else
        {
            printf( "                  |"
                    "    ---"
                    "                  n/a" "                  n/a\r\n" );
        }
    }
}
