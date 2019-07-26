#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.h"

int main( int argc, char* argv[] )
{
    typedef st_buffer_size_t  buf_size_t;

    buf_size_t const NUM_BEAM_ELEMENTS = ( buf_size_t)25u;
    buf_size_t const DATA_CAPACITY     = ( buf_size_t )( 1u << 20u );

    st_Buffer* eb = st_Buffer_new( DATA_CAPACITY );

    buf_size_t ii = ( buf_size_t )0u;
    double length = ( double )10.0L;

    st_Drift* copy_of_drift = 0; /* see below */

    /* add a new drift to the eb buffer using st_Drift_new function ->
     * the resulting drift will have default values, i.e. length == 0 */

    st_Drift* drift = st_Drift_new( eb );
    assert( drift != SIXTRL_NULLPTR );

    st_Drift_set_length( drift, length );

    length += ( double )1.0L;
    ++ii;

    /* add a new drift to the eb buffer using the st_Drift_add function ->
     * the resulting drift will already have the provided length */

    drift = st_Drift_add( eb, length );
    assert( drift != SIXTRL_NULLPTR );

    length += ( double )1.0L;
    ++ii;

    /* add a new drift to the eb buffer which is an exact copy of an already
     * existing st_Drift -> the two instances will have the exact same length */

    copy_of_drift = st_Drift_add_copy( eb, drift );

    assert( copy_of_drift != SIXTRL_NULLPTR );
    assert( copy_of_drift != drift  );
    assert( memcmp( copy_of_drift, drift, sizeof( st_Drift ) ) == 0 );

    #if defined( NDEBUG )
    ( void )copy_of_drift;
    #endif /* if defined( NDEBUG ) */

    length += ( double )1.0L;
    ++ii;

    for( ; ii < NUM_BEAM_ELEMENTS ; ++ii, length += ( double )1.0L )
    {
        drift = st_Drift_add( eb, length );
        assert( drift != SIXTRL_NULLPTR );
    }

    /* print out all existing beam elements using the convenience
     * st_BeamElement_print() function from testlib */

    for( ii = 0u ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        /* get the ii-th beam element object from the eb buffer */
        st_Object* be_object = st_Buffer_get_object( eb, ii );
        assert( be_object != SIXTRL_NULLPTR );

        /* We are about to modify the length of each drift ->
         * print the current drift before we change the length to have
         * something to compare against: */

        printf( "before changing the length of beam belement %d\r\n",
                ( int )ii );

        /* Print the be with the generic print helper function */
        st_BeamElement_print_out( be_object );

        /* We can get access to actual stored object if we know which type
         * it represents. In our case, that's easy - all stored objects are
         * drifts. But let's check that: */

        if( st_Object_get_type_id( be_object ) == st_OBJECT_TYPE_DRIFT )
        {
            st_Drift* stored_drift =
                ( st_Drift* )st_Object_get_begin_ptr( be_object );

            /* We could now modify the length of the already stored Drift.
             * If you want to prevent accidentially doing that, use the
             * const interface */

            st_Drift const* read_only_drift =
                ( st_Drift const* )st_Object_get_const_begin_ptr( be_object );

            /* Modify the length via the rw pointer: */
            double const new_length =
                st_Drift_get_length( stored_drift ) + ( double )1.0L;

            st_Drift_set_length( stored_drift, new_length );

            /* Since stored_drift and read_only_drift point to the same
             * location, read_only_drift yields the new length: */

            if( fabs( st_Drift_get_length( read_only_drift ) - new_length ) >= 1e-12 )
            {
                printf( "Error!" );
                break;
            }
        }

        /* Print the be with the generic print helper function after the change */
        printf( "after  changing the length of beam belement %d\r\n",
                ( int )ii );

        st_BeamElement_print_out( be_object );

        printf( "\r\n" );
    }

    /* Cleaning up */

    st_Buffer_delete( eb );

    return 0;
}

/* end:  examples/c99/simple_drift_buffer.c */
