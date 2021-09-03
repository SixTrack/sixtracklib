#ifndef SIXTRACKLIB_COMMON_BE_BEAMFIELDS_DAWSON_COEFF_H__
#define SIXTRACKLIB_COMMON_BE_BEAMFIELDS_DAWSON_COEFF_H__

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
    NS(CERRF_DAWSON_XI)[ SIXTRL_CERRF_DAWSON_N_XN ];

SIXTRL_EXTERN SIXTRL_REAL_T const
    NS(CERRF_DAWSON_FZ_XI)[ SIXTRL_CERRF_DAWSON_N_XN ];

SIXTRL_EXTERN SIXTRL_INT32_T const
    NS(CERRF_DAWSON_NT_XI_ABS_D10)[ SIXTRL_CERRF_DAWSON_N_XN ];

SIXTRL_EXTERN SIXTRL_INT32_T const
    NS(CERRF_DAWSON_NT_XI_REL_D14)[ SIXTRL_CERRF_DAWSON_N_XN ];

SIXTRL_EXTERN SIXTRL_REAL_T const
    NS(CERRF_DAWSON_FZ_KK_XI)[ SIXTRL_CERRF_DAWSON_NUM_TAYLOR_COEFF ];

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BE_BEAMFIELDS_DAWSON_COEFF_H__ */
