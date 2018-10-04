#ifndef SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TIME_TOOLS_H__
#define SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TIME_TOOLS_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_HOST_FN double NS(Time_get_seconds_since_epoch)( void );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TIME_TOOLS_H__ */

/* end: tests/sixtracklib/testlib/common/time.h */
