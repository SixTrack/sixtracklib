#include "sixtracklib/testlib/common/beam_elements/be_rfmultipole.h"
#include "sixtracklib/common/be_rfmultipole/be_rfmultipole.h"

void NS(RFMultipole_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mp )
{
    if( ( mp != SIXTRL_NULLPTR ) && ( fp != SIXTRL_NULLPTR ) )
    {
        NS(rf_multipole_int_t) const order = NS(RFMultipole_order)( mp );

        fprintf( fp,
                "|rf-multipole     | order      = %3ld;\r\n"
                "                  | voltage    = %+20.12f m\r\n"
                "                  | frequency  = %+20.12f m\r\n"
                "                  | lag        = %+20.12f m\r\n"
                "                  | bal_addr   = %21lx\r\n"
                "                  | phase_addr = %21lx\r\n",
                ( long int )order,
                NS(RFMultipole_voltage)( mp ),
                NS(RFMultipole_frequency)( mp ),
                NS(RFMultipole_lag)( mp ),
                ( uintptr_t )NS(RFMultipole_bal_addr)( mp ),
                ( uintptr_t )NS(RFMultipole_phase_addr)( mp ) );

        fprintf( fp, "                  |"
                    "    idx"
                    "                  knl"
                    "                  ksl"
                    "               phase_n"
                    "               phase_s\r\n" );

        if( order >= ( NS(rf_multipole_int_t) )0 )
        {
            NS(rf_multipole_int_t) ii = ( NS(rf_multipole_int_t) )0;
            for( ; ii <= order ; ++ii )
            {
                fprintf( fp,
                        "                  | %6ld %+20.12f %+20.12f "
                        "%+20.12f %+20.12f\r\n",
                        ( long int )ii,
                        NS(RFMultipole_knl)( mp, ii ),
                        NS(RFMultipole_ksl)( mp, ii ),
                        NS(RFMultipole_phase_n)( mp, ii ),
                        NS(RFMultipole_phase_s)( mp, ii ) );
            }
        }
        else
        {
            fprintf( fp, "                  |"
                    "    ---"
                    "                  n/a"
                    "                  n/a"
                    "                  n/a"
                    "                  n/a\r\n" );
        }
    }
}
