#include <cassert>
#include <iostream>
#include <iomanip>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.hpp"

int main( int argc, char* argv[] )
{
    namespace st     = sixtrack;
    using buf_size_t = st::Buffer::size_type;

    buf_size_t const NUM_BEAM_ELEMENTS = ( buf_size_t)25u;
    buf_size_t const DATA_CAPACITY     = ( buf_size_t )( 1u << 20u );

    st::Buffer eb( DATA_CAPACITY );

    buf_size_t ii = buf_size_t{ 0 };
    double length = double{ 10.0L };

    /* add a new drift to the eb buffer using the st::Drift_new function ->
     * the resulting drift will have default values, i.e. length == 0 */

    st::Drift* drift = st::Drift_new( eb );
    assert( drift  != nullptr );

    drift->setLength( length );
    length += double{ 1.0L };
    ++ii;

    /* add a new drift to the eb buffer using the st::Drift_add function ->
     * the resulting drift will already have the provided length */

    drift = st::Drift_add( eb, length );
    assert( drift != 0 );

    length += double{ 1.0L };
    ++ii;

    /* add a new drift to the eb buffer which is an exact copy of an already
     * existing st::Drift -> the two instances will have the exact same length */

    st::Drift* copy_of_drift = st::Drift_add_copy( eb, *drift );

    assert( copy_of_drift != 0 );
    assert( copy_of_drift != drift  );
    assert( memcmp( copy_of_drift, drift, sizeof( st_Drift ) ) == 0 );

    #if defined( NDEBUG )
    ( void )copy_of_drift;
    #endif /* defiend( NDEBUG ) */

    length += ( double )1.0L;
    ++ii;

    /* Using C++, we can add beam elements and similar Objects also via the
     * buffer using a template interface. The calls below are equivalent to the
     * free-standing calls above: */

    drift = eb.createNew< st::Drift >();
    assert( drift != nullptr );
    drift->setLength( length );

    length += ( double )1.0L;
    ++ii;

    for( ; ii < NUM_BEAM_ELEMENTS ; ++ii, length += ( double )1.0L )
    {
        drift = eb.add< st::Drift >( length );
        assert( drift != 0 );
    }

    /* print out all existing beam elements using the convenience
     * st_BeamElement_print() function from testlib */

    for( ii = 0u ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        /* get the ii-th beam element object from the eb buffer */
        st::Buffer::object_t* be_object = eb[ ii ];
        assert( be_object != 0 );

        /* We are about to modify the length of each drift ->
         * print the current drift before we change the length to have
         * something to compare against: */

        std::cout << "before changing the length of beam belement "
                  << ii << "\r\n";

        /* Print the be with the generic print helper function */
        ::st_BeamElement_print( be_object );

        /* Try to access the stored drift: */
        st::Drift* stored_drift = st::Drift::FromBuffer( eb, ii );

        /* if the object with the index ii would not have been a drift,
         * the returned pointer would have been a nullptr: */

        assert( stored_drift != nullptr);

        double const new_length = stored_drift->getLength() + double{ 1L };
        stored_drift->setLength( new_length );

        /* Print the be with the generic print helper function after the change */
        std::cout << "after  changing the length of beam belement "
                  << ii << "\r\n";

        st_BeamElement_print( be_object );

        std::cout << "\r\n";
    }

    std::cout << std::endl;

    return 0;
}

/* end:  examples/c99/simple_drift_buffer.c */
