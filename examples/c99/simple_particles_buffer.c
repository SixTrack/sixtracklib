#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "sixtracklib/sixtracklib.h"

static void printParticles( const st_Particles *const particles );

int main( int argc, char* argv[] )
{
    typedef st_buffer_size_t  buf_size_t;

    buf_size_t const DATA_CAPACITY = ( buf_size_t )( 1u << 20u );
    buf_size_t const NUM_PARTICLES = ( buf_size_t )4u;

    st_Buffer* pb = st_Buffer_new( DATA_CAPACITY );

    double const x_offset = ( double )5.0;
    buf_size_t ii = ( buf_size_t )0u;

    st_Particles* particles = st_Particles_new( pb, NUM_PARTICLES );

    st_Particles_set_rpp( particles, ( double[] ){ 1.0, 1.0, 1.0, 1.0 } );
    st_Particles_set_rvv( particles, ( double[] ){ 1.0, 1.0, 1.0, 1.0 } );
    st_Particles_set_px(  particles, ( double[] ){ 0.0, 0.0, 0.0, 0.0 } );
    st_Particles_set_px(  particles, ( double[] ){ 1e-5, 1e-5, 1e-5, 1e-5 } );
    st_Particles_set_s(   particles, ( double[] ){ 0.0, 0.0, 0.0, 0.0 } );

    st_Particles_set_x( particles, ( double[] ){ 0.0, 2.0, 3.0, 4.0 } );
    st_Particles_set_x_value( particles, 0,  1.0 );

    st_Particles_set_y( particles, ( double[] ){ -4.0, -3.0, -2.0, 1.0 } );
    st_Particles_set_y_value( particles, 3, -1.0 );

    st_Particles* dest_for_copy = st_Particles_new( pb, NUM_PARTICLES );

    st_Particles_copy( dest_for_copy, particles );

    for( ; ii < NUM_PARTICLES ; ++ii )
    {
        st_Particles_set_x_value( dest_for_copy, ii,
            st_Particles_get_x_value( particles, ii ) + x_offset );
    }

    st_Particles* difference    = st_Particles_new( pb, NUM_PARTICLES );
    st_Particles_calculate_difference( dest_for_copy, particles, difference );

    for( ii = 0u ; ii < NUM_PARTICLES ; ++ii )
    {
        assert( fabs( st_Particles_get_x_value( difference, ii ) - x_offset ) <
                ( double )1e-16 );
    }

    /* --------------------------------------------------------------------- */
    /* Copy to flat buffer and reconstruct the contents of this storage using
     * a second st_Buffer instance */

    if( st_Buffer_get_num_of_objects( pb ) == ( buf_size_t )3u )
    {
        typedef unsigned char* ptr_raw_t;

        st_Buffer buffer2;

        buf_size_t buffer_size = st_Buffer_get_size( pb );
        ptr_raw_t  copy_buffer = ( ptr_raw_t )malloc( buffer_size );
        int success = -1;

        memcpy( copy_buffer, st_Buffer_get_const_data_begin( pb ),
                st_Buffer_get_size( pb ) );

        st_Buffer_preset( &buffer2 );
        success = st_Buffer_init_from_data( &buffer2, copy_buffer, buffer_size );

        if( success == 0 )
        {
            buf_size_t const num_particles_blocks =
                st_Buffer_get_num_of_objects( pb );

            st_Object const* cmp_obj_it  = st_Buffer_get_const_objects_begin( pb );
            st_Object const* cmp_obj_end = st_Buffer_get_const_objects_end( pb );
            st_Object const* obj_it      = st_Buffer_get_objects_begin( &buffer2 );

            ii = ( buf_size_t )0u;

            for( ; cmp_obj_it != cmp_obj_end ; ++cmp_obj_it, ++obj_it, ++ii )
            {
                st_Particles const* particles = ( st_Particles const*
                    )st_Object_get_const_begin_ptr( obj_it );

                st_Particles const* cmp_particles = ( st_Particles const*
                    )st_Object_get_const_begin_ptr( cmp_obj_it );

                printf( "particle blocks: %3lu / %3lu\r\n",
                        ii, num_particles_blocks );

                printf( " -> original particles object : \r\n" );
                printParticles( cmp_particles );

                printf( " -> copy particles object : \r\n" );
                printParticles( particles );
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

    if( st_Buffer_get_num_of_objects( pb ) == ( buf_size_t )3u )
    {
        typedef unsigned char* ptr_raw_t;

        buf_size_t buffer_size     = st_Buffer_get_size( pb );
        buf_size_t const slot_size = st_Buffer_get_slot_size( pb );
        ptr_raw_t  copy_buffer     = ( ptr_raw_t )malloc( buffer_size );

        int success = -1;

        memcpy( copy_buffer, st_Buffer_get_const_data_begin( pb ),
                st_Buffer_get_size( pb ) );

        success = st_ManagedBuffer_remap( copy_buffer, slot_size );
        assert( success == 0 );

        if( success == 0 )
        {
            buf_size_t const num_particles_blocks =
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
                st_Particles const* particles = ( st_Particles const*
                    )st_Object_get_const_begin_ptr( obj_it );

                printf( "managed buffer particle blocks: %3lu / %3lu\r\n",
                        ii, num_particles_blocks );

                printf( " -> copy particles object : \r\n" );
                printParticles( particles );
            }
        }

        free( copy_buffer );
    }

    st_Buffer_delete( pb );

    return 0;
}

inline void printParticles( const st_Particles *const particles )
{
    typedef st_buffer_size_t buf_size_t;

    if( particles != SIXTRL_NULLPTR )
    {
        buf_size_t const num_particles =
            st_Particles_get_num_of_particles( particles );

        double const* it  = SIXTRL_NULLPTR;
        double const* end = SIXTRL_NULLPTR;

        printf( "\r\n" "s        | " );
        it  = st_Particles_get_const_s( particles );
        end = it + num_particles;
        for( ; it != end ; ++it ) printf( " %18.12f", *it );

        printf( "\r\n" "x        | " );
        it  = st_Particles_get_const_x( particles );
        end = it + num_particles;
        for( ; it != end ; ++it ) printf( " %18.12f", *it );

        printf( "\r\n" "y        | " );
        it  = st_Particles_get_const_y( particles );
        end = it + num_particles;
        for( ; it != end ; ++it ) printf( " %18.12f", *it );

        printf( "\r\n" "px       | " );
        it  = st_Particles_get_const_px( particles );
        end = it + num_particles;
        for( ; it != end ; ++it ) printf( " %18.12f", *it );

        printf( "\r\n" "py       | " );
        it  = st_Particles_get_const_py( particles );
        end = it + num_particles;
        for( ; it != end ; ++it ) printf( " %18.12f", *it );

        printf( "\r\n" "zeta     | " );
        it  = st_Particles_get_const_zeta( particles );
        end = it + num_particles;
        for( ; it != end ; ++it ) printf( " %18.12f", *it );

        printf( "\r\n" "psigma   | " );
        it  = st_Particles_get_const_psigma( particles );
        end = it + num_particles;
        for( ; it != end ; ++it ) printf( " %18.12f", *it );

        printf( "\r\n" "rpp      | " );
        it  = st_Particles_get_const_rpp( particles );
        end = it + num_particles;
        for( ; it != end ; ++it ) printf( " %18.12f", *it );

        printf( "\r\n" "rvv      | " );
        it  = st_Particles_get_const_rvv( particles );
        end = it + num_particles;
        for( ; it != end ; ++it ) printf( " %18.12f", *it );

        printf( "\r\n" "chi      | " );
        it  = st_Particles_get_const_rvv( particles );
        end = it + num_particles;
        for( ; it != end ; ++it ) printf( " %18.12f", *it );

        printf( "\r\n" "\r\n" );
    }

    return;
}

/* end:  examples/c99/simple_particles_buffer.c */
