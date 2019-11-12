#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <stdio.h>
    #include <limits.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/sixtracklib.h"
    #include "sixtracklib/testlib.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

int main()
{
    typedef NS(Buffer)                  buffer_t;
    typedef NS(buffer_size_t)           buf_size_t;
    typedef NS(Drift)                   drift_t;
    typedef unsigned long long          prng_seed_t;
    typedef NS(drift_real_t)            real_t;
    typedef NS(particle_num_elements_t) nelem_t;

    int success = 0;

    buf_size_t const NUM_PARTICLES     = ( buf_size_t )1000u;
    buf_size_t const NUM_BEAM_ELEMENTS = ( buf_size_t )1000u;

    /* To have a sizeable number of zero-length drifts, we use
     * a percentage ratio and extent the drift to the negative numbers;
     * all negative drift lengths will be then set to 0.0 */

    real_t const ZERO_LENGTH_RATIO  = ( real_t )0.1;
    real_t const MAX_DRIFT_LENGTH   = ( real_t )100.0;
    real_t const MIN_DRIFT_LENGTH   = -( MAX_DRIFT_LENGTH * ZERO_LENGTH_RATIO );
    real_t const DELTA_DRIFT_LENGTH = MAX_DRIFT_LENGTH - MIN_DRIFT_LENGTH;

    prng_seed_t const seed = ( prng_seed_t )20180910u;

    buffer_t*       buffer = NS(Buffer_new)( ( buf_size_t )( 1u << 24u ) );
    NS(Particles)*  particles = NS(Particles_new)( buffer, NUM_PARTICLES );
    NS(Particles)*  cmp_particles = SIXTRL_NULLPTR;

    buf_size_t ii = ( buf_size_t )0u;

    SIXTRL_ASSERT( buffer        != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( particles     != SIXTRL_NULLPTR );

    NS(Random_init_genrand64)( seed );
    NS(Particles_realistic_init)( particles );

    for( ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        drift_t* drift = SIXTRL_NULLPTR;

        real_t length = MIN_DRIFT_LENGTH +
            DELTA_DRIFT_LENGTH * NS(Random_genrand64_real1)();

        if( length <  ( real_t )0.0 )
        {
            length = ( real_t )0.0;
        }

        drift = NS(Drift_add)( buffer, length );

        if( drift == SIXTRL_NULLPTR )
        {
            success = 1;
            break;
        }
    }

    if( success == 0 )
    {
        cmp_particles = NS(Particles_new)( buffer, NUM_PARTICLES );

        if( cmp_particles != SIXTRL_NULLPTR )
        {
            NS(Particles_copy)( cmp_particles, particles );
        }
        else
        {
            success = 1;
        }
    }

    if( NS(Buffer_get_num_of_objects)( buffer ) != ( NUM_BEAM_ELEMENTS + 2u ) )
    {
        success = 1;
    }

    if( success == 0 )
    {
        NS(Object) const* obj_begin =
            NS(Buffer_get_const_objects_begin)( buffer );

        NS(Object) const* obj_end = NS(Buffer_get_const_objects_end)( buffer );

        NS(Object) const* be_begin = obj_begin + 1u;
        NS(Object) const* be_end   = be_begin + NUM_BEAM_ELEMENTS;

        nelem_t const npart =
            NS(Particles_get_num_of_particles)( cmp_particles );

        nelem_t ii = ( nelem_t)0u;

        ( void )obj_end;

        SIXTRL_ASSERT( obj_begin != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( obj_end   != obj_begin );

        SIXTRL_ASSERT( NS(Object_get_type_id)( obj_begin ) ==
                       NS(OBJECT_TYPE_PARTICLE) );

        SIXTRL_ASSERT( NS(Object_get_begin_addr)( obj_begin ) ==
                       ( uintptr_t )particles );

        SIXTRL_ASSERT( be_begin != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( be_end   != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( be_end   != be_begin );
        SIXTRL_ASSERT( be_end   != obj_end  );

        SIXTRL_ASSERT( NS(Object_get_type_id)( be_begin ) ==
                       NS(OBJECT_TYPE_DRIFT) );

        SIXTRL_ASSERT( NS(Object_get_type_id)( be_end ) ==
                       NS(OBJECT_TYPE_PARTICLE) );

        SIXTRL_ASSERT( NS(Object_get_begin_addr)( be_end ) ==
                       ( uintptr_t )cmp_particles );

        success = NS(TRACK_SUCCESS);

        for( ; ii < npart; ++ii )
        {
            success |= NS(Track_particle_until_turn_objs)( cmp_particles, ii,
                be_begin, be_end, NS(Particles_get_at_turn_value)(
                    cmp_particles, ii ) + 1u );
        }
    }

    if( success == NS(TRACK_SUCCESS) )
    {
        FILE* fp = fopen( NS(PATH_TO_TEST_TRACKING_BE_DRIFT_DATA), "wb" );

        if( fp != SIXTRL_NULLPTR )
        {
            buf_size_t const cnt = fwrite( ( unsigned char const* )(
                uintptr_t )NS(Buffer_get_data_begin_addr)( buffer ),
                NS(Buffer_get_size)( buffer ), ( buf_size_t )1u, fp );

            SIXTRL_ASSERT( cnt == ( buf_size_t )1u );
            ( void )cnt;

            fclose( fp );
            fp = SIXTRL_NULLPTR;
        }
        else
        {
            success = 1;
        }
    }

    NS(Buffer_delete)( buffer );

    SIXTRL_ASSERT( success == 0 );

    return success;
}

/* end: tests/testdata/generators/generate_track_be_drift_data.c */
