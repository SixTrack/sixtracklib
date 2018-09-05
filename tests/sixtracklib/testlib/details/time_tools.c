#include "sixtracklib/testlib/time_tools.h"

#include <time.h>
#include <sys/time.h>

/* TODO: Implement for Win32 where gettimeofday() is (probably) not available! */

extern SIXTRL_HOST_FN double NS(Time_get_seconds_since_epoch)( void );

double NS(Time_get_seconds_since_epoch)()
{
    struct timeval  tv;

    int ret = gettimeofday( &tv, SIXTRL_NULLPTR );

    #if !defined( NDEBUG )
    SIXTRL_ASSERT( ret == 0 );
    #else
    ( void )ret;
    #endif /* !defiend( NDEBUG ) */

    return ( double )tv.tv_sec + ( ( double )1e-6 * ( double )tv.tv_usec );
}

/* end: tests/sixtracklib/testlib/details/time_tools.c */
