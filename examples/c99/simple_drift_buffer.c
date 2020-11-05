#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.h"

int main( int argc, char* argv[] )
{
    typedef NS(buffer_size_t)  buf_size_t;

    buf_size_t const NUM_BEAM_ELEMENTS = ( buf_size_t)25u;
    buf_size_t const DATA_CAPACITY     = ( buf_size_t )( 1u << 20u );

    NS(Buffer)* eb = NS(Buffer_new)( DATA_CAPACITY );

    buf_size_t ii = ( buf_size_t )0u;
    double length = ( double )10.0L;

    NS(Drift)* copy_of_drift = 0; /* see below */

    /* add a new drift to the eb buffer using NS(Drift_new) function ->
     * the resulting drift will have default values, i.e. length == 0 */

    NS(Drift)* drift = NS(Drift_new)( eb );
    assert( drift != SIXTRL_NULLPTR );

    NS(Drift_set_length)( drift, length );

    length += ( double )1.0L;
    ++ii;

    /* add a new drift to the eb buffer using the NS(Drift_add) function ->
     * the resulting drift will already have the provided length */

    drift = NS(Drift_add)( eb, length );
    assert( drift != SIXTRL_NULLPTR );

    length += ( double )1.0L;
    ++ii;

    /* add a new drift to the eb buffer which is an exact copy of an already
     * existing NS(Drift) -> the two instances will have the exact same length */

    copy_of_drift = NS(Drift_add_copy)( eb, drift );

    assert( copy_of_drift != SIXTRL_NULLPTR );
    assert( copy_of_drift != drift  );
    assert( memcmp( copy_of_drift, drift, sizeof( NS(Drift) ) ) == 0 );

    #if defined( NDEBUG )
    ( void )copy_of_drift;
    #endif /* if defined( NDEBUG ) */

    length += ( double )1.0L;
    ++ii;

    for( ; ii < NUM_BEAM_ELEMENTS ; ++ii, length += ( double )1.0L )
    {
        drift = NS(Drift_add)( eb, length );
        assert( drift != SIXTRL_NULLPTR );
    }

    /* print out all existing beam elements using the convenience
     * NS(BeamElement_print)() function from testlib */

    for( ii = 0u ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        /* get the ii-th beam element object from the eb buffer */
        NS(Object)* be_object = NS(Buffer_get_object)( eb, ii );
        assert( be_object != SIXTRL_NULLPTR );

        /* We are about to modify the length of each drift ->
         * print the current drift before we change the length to have
         * something to compare against: */

        printf( "before changing the length of beam belement %d\r\n",
                ( int )ii );

        /* Print the be with the generic print helper function */
        NS(BeamElement_print_out)( be_object );

        /* We can get access to actual stored object if we know which type
         * it represents. In our case, that's easy - all stored objects are
         * drifts. But let's check that: */

        if( NS(Object_get_type_id)( be_object ) == NS(OBJECT_TYPE_DRIFT) )
        {
            NS(Drift)* stored_drift =
                ( NS(Drift)* )NS(Object_get_begin_ptr)( be_object );

            /* We could now modify the length of the already stored Drift.
             * If you want to prevent accidentially doing that, use the
             * const interface */

            NS(Drift) const* read_only_drift =
                ( NS(Drift) const* )NS(Object_get_const_begin_ptr)( be_object );

            /* Modify the length via the rw pointer: */
            double const new_length = NS(Drift_length)( stored_drift ) + ( double )1;

            NS(Drift_set_length)( stored_drift, new_length );

            /* Since stored_drift and read_only_drift point to the same
             * location, read_only_drift yields the new length: */

            if( fabs( NS(Drift_length)( read_only_drift ) - new_length ) >= 1e-12 )
            {
                printf( "Error!" );
                break;
            }
        }

        /* Print the be with the generic print helper function after the change */
        printf( "after  changing the length of beam belement %d\r\n",
                ( int )ii );

        NS(BeamElement_print_out)( be_object );

        printf( "\r\n" );
    }

    /* Cleaning up */

    NS(Buffer_delete)( eb );

    return 0;
}

/* end:  examples/c99/simple_drift_buffer.c */
