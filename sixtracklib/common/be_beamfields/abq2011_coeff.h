#ifndef SIXTRACKLIB_COMMON_BE_BEAMFIELDS_ABQ2011_COEFF_H__
#define SIXTRACKLIB_COMMON_BE_BEAMFIELDS_ABQ2011_COEFF_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_beamfields/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C"  {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_EXTERN SIXTRL_REAL_T const
    NS(CERRF_ABQ2011_FOURIER_COEFF)[ SIXTRL_CERRF_ABQ2011_N_FOURIER ];

SIXTRL_EXTERN SIXTRL_REAL_T const
    NS(CERRF_ABQ2011_ROOT_TAYLOR_COEFF)[ SIXTRL_CERRF_ABQ2011_NUM_TAYLOR_COEFF ];

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BE_BEAMFIELDS_ABQ2011_COEFF_H__ */
