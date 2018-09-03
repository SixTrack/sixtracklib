#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "sixtracklib/sixtracklib.h"

int main( int argc, char* argv[] )
{
    typedef st_buffer_size_t  buf_size_t;

    buf_size_t const NUM_BEAM_ELEMENTS = ( buf_size_t)1000u;
    buf_size_t const DATA_CAPACITY     = ( buf_size_t )( 1u << 20u );

    st_Buffer* eb = st_Buffer_new( DATA_CAPACITY );

    buf_size_t ii = ( buf_size_t )0u;
    double length = ( double )0.0;

    for( ; ii < NUM_BEAM_ELEMENTS ; ++ii, length += 1.0 )
    {
        st_Drift* drift = st_Drift_add( eb, length );
        assert( drift != SIXTRL_NULLPTR );
    }

    /* --------------------------------------------------------------------- */
    /* Copy to flat buffer and reconstruct the contents of this storage using
     * a second st_Buffer instance */

    if( st_Buffer_get_size( eb ) > ( buf_size_t )0u )
    {
        typedef unsigned char raw_t;
        typedef raw_t*        ptr_raw_t;

        st_Buffer buffer2;

        int success = -1;

        ptr_raw_t copy_buffer = ( ptr_raw_t )malloc(
            st_Buffer_get_size( eb ) * sizeof( raw_t ) );

        buf_size_t const buf_size = st_Buffer_get_size( eb );
        memcpy( copy_buffer, st_Buffer_get_const_data_begin( eb ), buf_size );

        st_Buffer_preset( &buffer2 );
        success = st_Buffer_init_from_data( &buffer2, copy_buffer, buf_size );

        if( success == 0 )
        {
            buf_size_t const num_beam_elements =
                st_Buffer_get_num_of_objects( eb );

            st_Object const* cmp_obj_it  = st_Buffer_get_const_objects_begin( eb );
            st_Object const* cmp_obj_end = st_Buffer_get_const_objects_end( eb );
            st_Object const* obj_it      = st_Buffer_get_objects_begin( &buffer2 );

            ii = ( buf_size_t )0u;

            for( ; cmp_obj_it != cmp_obj_end ; ++cmp_obj_it, ++obj_it, ++ii )
            {
                st_Drift const* cmp_drift = ( st_Drift const*
                    )st_Object_get_const_begin_ptr( cmp_obj_it );

                st_Drift const* drift     = ( st_Drift const*
                    )st_Object_get_const_begin_ptr( obj_it );

                printf( "%6lu / %6lu | cmp_drift : length = %18.12f\r\n"
                        "                | "
                        "drift     : length = %18.12f\r\n\r\n",
                        ii, num_beam_elements,
                        st_Drift_get_length( cmp_drift ),
                        st_Drift_get_length( drift ) );
            }
        }

        st_Buffer_free( &buffer2 );
        free( copy_buffer );
    }

    /* --------------------------------------------------------------------- */
    /* Copy to flat buffer and access the contents of the buffer using
     * the st_ManagedBuffer_ interface -> that's the least intrusive
     * way to read data from such a buffer, for example in the context of
     * Hybrid / GPU computing */

    if( st_Buffer_get_size( eb ) > ( buf_size_t )0u )
    {
        typedef unsigned char raw_t;
        typedef raw_t*        ptr_raw_t;

        buf_size_t const buf_size  = st_Buffer_get_size( eb ) * sizeof( raw_t );
        buf_size_t const slot_size = st_Buffer_get_slot_size( eb );
        ptr_raw_t copy_buffer = ( ptr_raw_t )malloc( buf_size );

        int success = -1;

        memcpy( copy_buffer, st_Buffer_get_const_data_begin( eb ), buf_size );
        success = st_ManagedBuffer_remap( copy_buffer, slot_size );

        if( success == 0 )
        {
            buf_size_t const num_beam_elements =
                st_ManagedBuffer_get_num_objects( copy_buffer, slot_size );

            st_Object const* obj_it  =
                st_ManagedBuffer_get_const_objects_index_begin(
                    copy_buffer, slot_size );

            st_Object const* obj_end =
                st_ManagedBuffer_get_const_objects_index_end(
                    copy_buffer, slot_size );

            ii = ( buf_size_t )0u;

            for( ; obj_it != obj_end ; ++obj_it, ++ii )
            {
                st_Drift const* drift = ( st_Drift const*
                    )st_Object_get_const_begin_ptr( obj_it );

                printf( "%6lu / %6lu | drift : length = %18.12f\r\n",
                    ii, num_beam_elements, st_Drift_get_length( drift ) );
            }
        }

        free( copy_buffer );
    }

    st_Buffer_delete( eb );

    return 0;
}
/* end:  examples/c99/simple_drift_buffer.c */
