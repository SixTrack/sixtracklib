#include "sixtracklib/common/tests/test_track_tools.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/track.h"
#include "sixtracklib/common/impl/track_api.h"

#include "sixtracklib/common/details/random.h"
#include "sixtracklib/common/tests/test_particles_tools.h"

extern bool NS(Tracks_store_testdata_to_binary_file)(
        uint64_t const num_of_turns,
        NS(Blocks) const* SIXTRL_RESTRICT initial_particles_buffer,
        NS(Blocks) const* SIXTRL_RESTRICT result_particles_buffer,
        NS(Blocks) const* SIXTRL_RESTRICT beam_elements_buffer,
        NS(Blocks) const* SIXTRL_RESTRICT elem_by_elem_buffer,
        char const path_to_bin_testdata_file[] );

extern bool NS(Tracks_restore_testdata_from_binary_file)(
    char const path_to_bin_testdata_file[], uint64_t* ptr_num_of_turns,
    NS(Blocks)* SIXTRL_RESTRICT initial_particles_buffer,
    NS(Blocks)* SIXTRL_RESTRICT result_particles_buffer,
    NS(Blocks)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Blocks)* SIXTRL_RESTRICT elem_by_elem_buffer );

extern bool NS(TestData_test_tracking_single_particle)(
    char const *const SIXTRL_RESTRICT path_to_testdata_file );

extern bool NS(TestData_test_tracking_particles)(
    char const *const SIXTRL_RESTRICT path_to_testdata_file );

/* ------------------------------------------------------------------------- */

bool NS(Tracks_store_testdata_to_binary_file)(
        uint64_t const num_of_turns,
        NS(Blocks) const* SIXTRL_RESTRICT initial_particles_buffer,
        NS(Blocks) const* SIXTRL_RESTRICT result_particles_buffer,
        NS(Blocks) const* SIXTRL_RESTRICT beam_elements_buffer,
        NS(Blocks) const* SIXTRL_RESTRICT elem_by_elem_buffer,
        char const path_to_bin_testdata_file[] )
{
    bool success = false;

    if( ( num_of_turns > UINT64_C( 0 ) ) &&
        ( path_to_bin_testdata_file != 0 ) &&
        ( strlen( path_to_bin_testdata_file ) > 0u ) &&
        ( initial_particles_buffer != 0 ) &&
        ( NS(Blocks_are_serialized)( initial_particles_buffer ) ) &&
        ( NS(Blocks_get_num_of_blocks)( initial_particles_buffer ) > 0 ) &&
        ( NS(Blocks_get_total_num_bytes)( initial_particles_buffer ) > 0u ) &&
        ( NS(Blocks_get_const_data_begin)(
            ( NS(Blocks)* )initial_particles_buffer ) != 0 ) &&
        ( result_particles_buffer != 0 ) &&
        ( NS(Blocks_are_serialized)( result_particles_buffer ) ) &&
        ( NS(Blocks_get_num_of_blocks)( result_particles_buffer ) > 0 ) &&
        ( NS(Blocks_get_total_num_bytes)( result_particles_buffer ) > 0u ) &&
        ( NS(Blocks_get_const_data_begin)(
            ( NS(Blocks)* )result_particles_buffer ) != 0 ) &&
        ( beam_elements_buffer != 0 ) &&
        ( NS(Blocks_are_serialized)( beam_elements_buffer ) ) &&
        ( NS(Blocks_get_num_of_blocks)( beam_elements_buffer ) > 0 ) &&
        ( NS(Blocks_get_total_num_bytes)( beam_elements_buffer ) > 0u ) &&
        ( NS(Blocks_get_const_data_begin)(
            ( NS(Blocks)* )beam_elements_buffer ) != 0 ) &&
        ( ( elem_by_elem_buffer == 0 ) ||
            ( ( NS(Blocks_are_serialized)( elem_by_elem_buffer ) ) &&
            ( NS(Blocks_get_num_of_blocks)( elem_by_elem_buffer ) > 0 ) &&
            ( NS(Blocks_get_total_num_bytes)( elem_by_elem_buffer ) > 0u ) &&
            ( NS(Blocks_get_const_data_begin)(
                ( NS(Blocks)* )elem_by_elem_buffer ) != 0 ) )
        ) )
    {
        uint64_t const num_of_particle_blocks =
            NS(Blocks_get_num_of_blocks)( initial_particles_buffer );

        uint64_t num_of_beam_elements                = UINT64_C( 0 );
        uint64_t num_of_required_elem_by_elem_blocks = UINT64_C( 0 );
        uint64_t num_of_elem_by_elem_blocks          = UINT64_C( 0 );

        uint64_t* num_particles_vec = ( uint64_t* )malloc(
            sizeof( uint64_t ) * num_of_particle_blocks );

        if( num_particles_vec == 0 ) return success;

        if( ( num_of_particle_blocks == ( uint64_t
                )NS(Blocks_get_num_of_blocks)( result_particles_buffer ) ) )
        {
            uint64_t ii = UINT64_C( 0 );

            NS(BlockInfo) const* blocks_it  =
                NS(Blocks_get_const_block_infos_begin)(
                    initial_particles_buffer );

            NS(BlockInfo) const* blocks_end =
                NS(Blocks_get_const_block_infos_end)(
                    initial_particles_buffer );

            NS(BlockInfo) const* result_blocks_it =
                NS(Blocks_get_const_block_infos_begin)(
                    result_particles_buffer );

            success = true;

            for( ; ii < num_of_particle_blocks ; ++ii )
            {
                num_particles_vec[ ii ] = UINT64_C( 0 );
            }

            for( ii = UINT64_C( 0 ) ; blocks_it != blocks_end ;
                    ++blocks_it, ++result_blocks_it, ++ii )
            {
                NS(Particles) const* result_particles =
                    NS(Blocks_get_const_particles)( result_blocks_it );

                NS(Particles) const* initial_particles =
                    NS(Blocks_get_const_particles)( blocks_it );

                NS(block_num_elements_t) const result_num_of_particles =
                    NS(Particles_get_num_particles)( result_particles );

                NS(block_num_elements_t) const initial_num_of_particles =
                    NS(Particles_get_num_particles)( initial_particles );

                if( ( result_particles  == 0 ) ||
                    ( initial_particles == 0 ) ||
                    ( result_num_of_particles != initial_num_of_particles ) )
                {
                    success = false;
                    break;
                }

                num_particles_vec[ ii ] = ( uint64_t )initial_num_of_particles;
            }
        }

        num_of_beam_elements =
            NS(Blocks_get_num_of_blocks)( beam_elements_buffer );

        num_of_required_elem_by_elem_blocks =
            num_of_turns * num_of_beam_elements * num_of_particle_blocks;

        if( ( success ) && ( elem_by_elem_buffer != 0 ) )
        {
            num_of_elem_by_elem_blocks =
                NS(Blocks_get_num_of_blocks)( elem_by_elem_buffer );

            NS(BlockInfo) const* blocks_it =
                NS(Blocks_get_const_block_infos_begin)( elem_by_elem_buffer );

            if( ( blocks_it != 0 ) && ( num_of_elem_by_elem_blocks >=
                  num_of_required_elem_by_elem_blocks ) )
            {
                uint64_t ii = UINT64_C( 0 );

                for( ; ii < num_of_turns ; ++ii )
                {
                    uint64_t jj = UINT64_C( 0 );

                    for( ; jj < num_of_beam_elements ; ++jj )
                    {
                        uint64_t kk = UINT64_C( 0 );

                        for( ; kk < num_of_particle_blocks ; ++kk, ++blocks_it )
                        {
                            NS(Particles) const* particles =
                                NS(Blocks_get_const_particles)( blocks_it );

                            if( ( particles == 0 ) ||
                                ( NS(Particles_get_num_particles)( particles )
                                    != num_particles_vec[ kk ] ) )
                            {
                                success = false;
                                break;
                            }
                        }
                    }

                    if( !success ) break;
                }
            }
            else
            {
                success = false;
            }
        }

        if( success )
        {
            static size_t const U64_SIZE = sizeof( uint64_t );
            static size_t const ONE  = ( size_t )1u;

            FILE* fp = fopen( path_to_bin_testdata_file, "wb" );

            if( fp != 0 )
            {
                uint64_t ii = UINT64_C( 0 );

                success  = ( ONE == fwrite( &num_of_turns, U64_SIZE,
                                            ONE, fp ) );

                success &= ( ONE == fwrite( &num_of_beam_elements, U64_SIZE,
                                            ONE, fp ) );

                success &= ( ONE == fwrite( &num_of_particle_blocks, U64_SIZE,
                                            ONE, fp ) );


                for( ; ii < num_of_particle_blocks ; ++ii )
                {
                    success &= ( ONE == fwrite( &num_particles_vec[ ii ],
                                                U64_SIZE, ONE, fp ) );
                }

                if( elem_by_elem_buffer != 0 )
                {
                    success = ( ONE == fwrite(
                        &num_of_elem_by_elem_blocks, U64_SIZE, ONE, fp ) );
                }
                else
                {
                    uint64_t dummy = UINT64_C( 0 );
                    success = ( ONE == fwrite( &dummy, U64_SIZE, ONE, fp ) );
                }
            }

            if( ( fp != 0 ) && ( success ) )
            {
                uint64_t const beam_elements_bytes =
                    NS(Blocks_get_total_num_bytes)( beam_elements_buffer );

                success  = ( ONE == fwrite( &beam_elements_bytes, U64_SIZE,
                                            ONE, fp ) );

                success &= ( ONE == fwrite( NS(Blocks_get_const_data_begin)(
                                ( NS(Blocks)* )beam_elements_buffer ),
                                beam_elements_bytes, ONE, fp ) );
            }

            if( ( fp != 0 ) && ( success ) )
            {
                uint64_t const initial_particles_bytes =
                    NS(Blocks_get_total_num_bytes)( initial_particles_buffer );

                success  = ( ONE == fwrite( &initial_particles_bytes,
                                            U64_SIZE, ONE, fp ) );

                success &= ( ONE == fwrite( NS(Blocks_get_const_data_begin)(
                                ( NS(Blocks)* )initial_particles_buffer ),
                                initial_particles_bytes, ONE, fp ) );
            }

            if( ( fp != 0 ) && ( success ) )
            {
                uint64_t const result_particles_bytes =
                    NS(Blocks_get_total_num_bytes)( result_particles_buffer );

                success  = ( ONE == fwrite( &result_particles_bytes,
                                            U64_SIZE, ONE, fp ) );

                success &= ( ONE == fwrite( NS(Blocks_get_const_data_begin)(
                                ( NS(Blocks)* )result_particles_buffer ),
                                result_particles_bytes, ONE, fp ) );
            }

            if( ( fp != 0 ) && ( success ) )
            {
                uint64_t const elem_by_elem_bytes =
                    ( elem_by_elem_buffer != 0 )
                        ? NS(Blocks_get_total_num_bytes)( elem_by_elem_buffer )
                        : UINT64_C( 0 );

                success  = ( ONE == fwrite( &elem_by_elem_bytes,
                                U64_SIZE, ONE, fp ) );

                if( ( elem_by_elem_buffer != 0 ) &&
                    ( elem_by_elem_bytes > UINT64_C( 0 ) ) )
                {
                    success &= ( ONE == fwrite(
                        NS(Blocks_get_const_data_begin)( ( NS(Blocks)*
                            )elem_by_elem_buffer ), elem_by_elem_bytes,
                                ONE, fp ) );
                }
            }

            if( fp != 0 )
            {
                fflush( fp );
                fclose( fp );
                fp = 0;
            }
        }

        free( num_particles_vec );
        num_particles_vec = 0;
    }

    return success;
}

bool NS(Tracks_restore_testdata_from_binary_file)(
    char const path_to_bin_testdata_file[],
    uint64_t* ptr_num_of_turns,
    NS(Blocks)* SIXTRL_RESTRICT initial_particles_buffer,
    NS(Blocks)* SIXTRL_RESTRICT result_particles_buffer,
    NS(Blocks)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Blocks)* SIXTRL_RESTRICT elem_by_elem_buffer )
{
    bool success = false;

    FILE* fp = fopen( path_to_bin_testdata_file, "rb" );

    if( ( fp != 0 ) && ( ptr_num_of_turns != 0 ) )
    {
        static size_t const U64_SIZE = sizeof( uint64_t );
        static size_t const ONE  = ( size_t )1u;

        uint64_t* num_particles_vec = 0;

        uint64_t num_of_turns = UINT64_C( 0 );
        uint64_t num_of_beam_elements    = UINT64_C( 0 );
        uint64_t beam_elements_bytes     = UINT64_C( 0 );
        uint64_t num_of_particle_blocks  = UINT64_C( 0 );
        uint64_t initial_particles_bytes = UINT64_C( 0 );
        uint64_t result_particles_bytes  = UINT64_C( 0 );
        uint64_t num_elem_by_elem_blocks = UINT64_C( 0 );
        uint64_t elem_by_elem_bytes      = UINT64_C( 0 );

        success  = ( ONE == fread( &num_of_turns, U64_SIZE, ONE, fp ) );

        success &= ( ONE == fread( &num_of_beam_elements, U64_SIZE,
                                   ONE, fp ) );

        success &= ( ONE == fread( &num_of_particle_blocks,
                                   U64_SIZE, ONE, fp ) );

        if( ( success ) && ( num_of_particle_blocks != UINT64_C( 0 ) ) )
        {
            *ptr_num_of_turns = num_of_turns;

            num_particles_vec = ( uint64_t* )malloc(
                U64_SIZE * num_of_particle_blocks );

            if( num_particles_vec != 0 )
            {
                uint64_t ii = UINT64_C( 0 );

                for( ; ii < num_of_particle_blocks ; ++ii )
                {
                    success &= ( ONE == fread( &num_particles_vec[ ii ],
                                               U64_SIZE, ONE, fp ) );
                }
            }
            else
            {
                success = false;
            }
        }

        if( success )
        {
            success = ( ONE == fread( &num_elem_by_elem_blocks, U64_SIZE,
                                      ONE, fp ) );
        }

        if( success )
        {
            success = ( ONE == fread( &beam_elements_bytes, U64_SIZE,
                                      ONE, fp ) );

            if( ( success ) && ( beam_elements_bytes > UINT64_C( 0 ) ) )
            {
                unsigned char* temp_buffer =
                    ( unsigned char* )malloc( beam_elements_bytes );

                NS(Blocks_preset)( beam_elements_buffer );

                if( temp_buffer != 0 )
                {
                    success = ( ONE == fread( temp_buffer, beam_elements_bytes,
                                              ONE, fp ) );

                    if( ( success ) && ( beam_elements_buffer != 0 ) )
                    {
                        success = ( NS(Blocks_init_from_serialized_data)(
                            beam_elements_buffer, temp_buffer,
                                beam_elements_bytes ) == 0 );
                    }

                    free( temp_buffer );
                    temp_buffer = 0;
                }
                else
                {
                    success = false;
                }
            }
        }

        if( success )
        {
            success = ( ONE == fread( &initial_particles_bytes, U64_SIZE,
                                      ONE, fp ) );

            if( ( success ) && ( initial_particles_bytes > UINT64_C( 0 ) ) )
            {
                unsigned char* temp_buffer =
                    ( unsigned char* )malloc( initial_particles_bytes );

                NS(Blocks_preset)( initial_particles_buffer );

                if( temp_buffer != 0 )
                {
                    success = ( ONE == fread( temp_buffer,
                                    initial_particles_bytes, ONE, fp ) );

                    if( ( success ) && ( initial_particles_buffer != 0 ) )
                    {
                        success = ( NS(Blocks_init_from_serialized_data)(
                            initial_particles_buffer, temp_buffer,
                                initial_particles_bytes ) == 0 );
                    }

                    free( temp_buffer );
                    temp_buffer = 0;
                }
                else
                {
                    success = false;
                }
            }
        }

        if( success )
        {
            success = ( ONE == fread( &result_particles_bytes, U64_SIZE,
                                      ONE, fp ) );

            if( ( success ) && ( result_particles_bytes > UINT64_C( 0 ) ) )
            {
                unsigned char* temp_buffer =
                    ( unsigned char* )malloc( result_particles_bytes );

                NS(Blocks_preset)( result_particles_buffer );

                if( temp_buffer != 0 )
                {
                    success = ( ONE == fread( temp_buffer,
                                    result_particles_bytes, ONE, fp ) );

                    if( ( success ) && ( result_particles_buffer != 0 ) )
                    {
                        success = ( NS(Blocks_init_from_serialized_data)(
                            result_particles_buffer, temp_buffer,
                                result_particles_bytes ) == 0 );
                    }

                    free( temp_buffer );
                    temp_buffer = 0;
                }
                else
                {
                    success = false;
                }
            }
        }

        if( success )
        {
            success = ( ONE == fread( &elem_by_elem_bytes,
                                      U64_SIZE, ONE, fp ) );

            if( ( success ) && ( elem_by_elem_bytes > UINT64_C( 0 ) ) )
            {
                unsigned char* temp_buffer =
                    ( unsigned char* )malloc( elem_by_elem_bytes );

                NS(Blocks_preset)( elem_by_elem_buffer );

                if( temp_buffer != 0 )
                {
                    success = ( ONE == fread( temp_buffer,
                                    elem_by_elem_bytes, ONE, fp ) );

                    if( ( success ) && ( elem_by_elem_buffer != 0 ) )
                    {
                        success = ( NS(Blocks_init_from_serialized_data)(
                            elem_by_elem_buffer, temp_buffer,
                                elem_by_elem_bytes ) == 0 );
                    }

                    free( temp_buffer );
                    temp_buffer = 0;
                }
                else
                {
                    success = false;
                }
            }
        }

        free( num_particles_vec );
        num_particles_vec = 0;
    }

    if( fp != 0 )
    {
        fclose( fp );
        fp = 0;
    }

    return success;
}


bool NS(TestData_test_tracking_single_particle)(
    char const *const SIXTRL_RESTRICT path_to_testdata_file )
{
    bool success = false;

    SIXTRL_UINT64_T NUM_OF_TURNS = ( SIXTRL_UINT64_T )0;

    NS(Blocks) initial_particles_buffer;
    NS(Blocks) result_particles_buffer;
    NS(Blocks) beam_elements;
    NS(Blocks) elem_by_elem;

    NS(Blocks_preset)( &initial_particles_buffer );
    NS(Blocks_preset)( &result_particles_buffer );
    NS(Blocks_preset)( &beam_elements );
    NS(Blocks_preset)( &elem_by_elem );

    if( NS(Tracks_restore_testdata_from_binary_file)(
        path_to_testdata_file, &NUM_OF_TURNS, &initial_particles_buffer,
        &result_particles_buffer, &beam_elements, &elem_by_elem ) )
    {
        /* ----------------------------------------------------------------- */

        NS(block_size_t) const NUM_OF_BEAM_ELEMENTS =
            NS(Blocks_get_num_of_blocks)( &beam_elements );

        success  = ( NS(Blocks_are_serialized)( &beam_elements ) );
        success &= ( NUM_OF_BEAM_ELEMENTS > ( NS(block_size_t) ) 0u );

        /* ----------------------------------------------------------------- */

        NS(block_size_t) const NUM_OF_PARTICLE_BLOCKS =
            NS(Blocks_get_num_of_blocks)( &initial_particles_buffer );

        success &= ( NS(Blocks_are_serialized)( &result_particles_buffer  ) );
        success &= ( NS(Blocks_are_serialized)( &initial_particles_buffer ) );
        success &= ( NS(Blocks_get_num_of_blocks)( &result_particles_buffer )
                     == NUM_OF_PARTICLE_BLOCKS );

        NS(block_size_t) const required_num_elem_by_elem_blocks =
            NUM_OF_TURNS * NUM_OF_BEAM_ELEMENTS * NUM_OF_PARTICLE_BLOCKS;

        success &= ( ( ( NS(Blocks_get_num_of_blocks)( &elem_by_elem ) >=
                required_num_elem_by_elem_blocks ) &&
            ( NS(Blocks_get_total_num_bytes)( &elem_by_elem ) > 0 ) &&
            ( NS(Blocks_are_serialized)( &elem_by_elem ) ) ) ||
            ( NS(Blocks_get_num_of_blocks)( &elem_by_elem ) == 0 ) );

        /* ----------------------------------------------------------------- */

        NS(Blocks) particles_buffer;
        NS(Blocks_preset)( &particles_buffer );

        NS(Blocks) calculated_elem_by_elem;
        NS(Blocks_preset)( &calculated_elem_by_elem );

        success &= ( 0 == NS(Blocks_init_from_serialized_data)(
            &particles_buffer,
            NS(Blocks_get_const_data_begin)( &initial_particles_buffer ),
            NS(Blocks_get_total_num_bytes)( &initial_particles_buffer ) ) );

        success &= ( NS(Particles_buffers_have_same_structure)(
            &initial_particles_buffer, &particles_buffer ) );

        success &= ( !NS(Particles_buffers_map_to_same_memory)(
            &initial_particles_buffer, &particles_buffer ) );

        success &= ( NS(Particles_buffers_have_same_structure)(
            &result_particles_buffer, &particles_buffer ) );

        success &= ( !NS(Particles_buffers_map_to_same_memory)(
            &result_particles_buffer, &particles_buffer ) );

        if( NS(Blocks_get_num_of_blocks)( &elem_by_elem ) > 0 )
        {
            int const elem_by_elem_succes =
                NS(Blocks_init_from_serialized_data)(
                    &calculated_elem_by_elem,
                    NS(Blocks_get_const_data_begin)( &elem_by_elem ),
                    NS(Blocks_get_total_num_bytes)( &elem_by_elem ) );

            success &= ( elem_by_elem_succes == 0 );

            NS(Particles_buffer_preset_values)( &calculated_elem_by_elem );
        }

        /* ----------------------------------------------------------------- */

        NS(BlockInfo)* io_block_it = 0;

        NS(block_size_t) const NUM_IO_ELEMENTS_PER_TURN =
            NUM_OF_PARTICLE_BLOCKS * NUM_OF_BEAM_ELEMENTS;

        NS(block_size_t) const AVAILABLE_ELEM_BY_ELEM_BLOCKS =
            NS(Blocks_get_num_of_blocks)( &elem_by_elem );

        if( ( NUM_OF_TURNS * NUM_IO_ELEMENTS_PER_TURN )
                <= AVAILABLE_ELEM_BY_ELEM_BLOCKS )
        {
            io_block_it = NS(Blocks_get_block_infos_begin)(
                &calculated_elem_by_elem );
        }

        bool const use_elem_by_elem_buffer = ( io_block_it != 0 );

        for( NS(block_size_t) ii = 0 ; ii < NUM_OF_TURNS ; ++ii )
        {
            NS(BlockInfo)* part_block_it =
                NS(Blocks_get_block_infos_begin)( &particles_buffer );

            NS(BlockInfo)* part_block_end =
                NS(Blocks_get_block_infos_end)( &particles_buffer );

            success &= ( part_block_it  != 0 );
            success &= ( part_block_end != 0 );

            if( !success )
            {
                break;
            }

            for( ; part_block_it != part_block_end ; ++part_block_it )
            {
                NS(Particles)* io_particles =
                    NS(Blocks_get_particles)( io_block_it );

                NS(Particles)* particles =
                    NS(Blocks_get_particles)( part_block_it );

                NS(block_num_elements_t) const num_of_particles =
                    NS(Particles_get_num_particles)( particles );

                if( ( particles == 0 ) ||
                    ( ( io_block_it  != 0 ) && ( io_particles == 0 ) ) ||
                    ( num_of_particles <= 0 ) )
                {
                    success = false;
                    break;
                }

                NS(block_num_elements_t) ll = 0;

                for( ; ll < num_of_particles ; ++ll )
                {
                    if( 0 != NS(Track_beam_elements_particle)(
                            particles, ll, &beam_elements, io_block_it ) )
                    {
                        success = false;
                        break;
                    }
                }

                if( ( success ) && ( use_elem_by_elem_buffer ) )
                {
                    SIXTRL_ASSERT( io_block_it != 0 );
                    io_block_it = io_block_it + NUM_IO_ELEMENTS_PER_TURN;
                }
            }
        }

        /* ----------------------------------------------------------------- */

        if( success )
        {
            int const cmp_result_particles =
                NS(Particles_buffer_compare_values)(
                    &result_particles_buffer, &particles_buffer );

            success = ( cmp_result_particles == 0 );

            if( ( success ) && ( use_elem_by_elem_buffer ) )
            {
                success  = ( NS(Particles_buffers_have_same_structure)(
                    &elem_by_elem, &calculated_elem_by_elem ) );

                success &= ( !NS(Particles_buffers_map_to_same_memory)(
                    &elem_by_elem, &calculated_elem_by_elem ) );

                int const elem_by_elem_cmp_result =
                    NS(Particles_buffer_compare_values)(
                        &elem_by_elem, &calculated_elem_by_elem );

                success &= ( elem_by_elem_cmp_result == 0 );
            }
        }

        NS(Blocks_free)( &calculated_elem_by_elem );
        NS(Blocks_free)( &particles_buffer );
    }

    /* --------------------------------------------------------------------- */

    NS(Blocks_free)( &elem_by_elem );
    NS(Blocks_free)( &beam_elements );
    NS(Blocks_free)( &initial_particles_buffer );
    NS(Blocks_free)( &result_particles_buffer );

    return success;
}


bool NS(TestData_test_tracking_particles)(
    char const *const SIXTRL_RESTRICT path_to_testdata_file )
{
    bool success = false;

    SIXTRL_UINT64_T NUM_OF_TURNS = ( SIXTRL_UINT64_T )0;

    NS(Blocks) initial_particles_buffer;
    NS(Blocks) result_particles_buffer;
    NS(Blocks) beam_elements;
    NS(Blocks) elem_by_elem;

    NS(Blocks_preset)( &initial_particles_buffer );
    NS(Blocks_preset)( &result_particles_buffer );
    NS(Blocks_preset)( &beam_elements );
    NS(Blocks_preset)( &elem_by_elem );

    if( NS(Tracks_restore_testdata_from_binary_file)(
        path_to_testdata_file, &NUM_OF_TURNS, &initial_particles_buffer,
        &result_particles_buffer, &beam_elements, &elem_by_elem ) )
    {
        /* ----------------------------------------------------------------- */

        NS(block_size_t) const NUM_OF_BEAM_ELEMENTS =
            NS(Blocks_get_num_of_blocks)( &beam_elements );

        success  = ( NS(Blocks_are_serialized)( &beam_elements ) );
        success &= ( NUM_OF_BEAM_ELEMENTS > ( NS(block_size_t) )0 );

        /* ----------------------------------------------------------------- */

        NS(block_size_t) const NUM_OF_PARTICLE_BLOCKS =
            NS(Blocks_get_num_of_blocks)( &initial_particles_buffer );

        success &= ( NS(Blocks_are_serialized)( &result_particles_buffer  ) );
        success &= ( NS(Blocks_are_serialized)( &initial_particles_buffer ) );
        success &= ( NS(Blocks_get_num_of_blocks)( &result_particles_buffer )
                     == NUM_OF_PARTICLE_BLOCKS );

        NS(block_size_t) const required_num_elem_by_elem_blocks =
            NUM_OF_TURNS * NUM_OF_BEAM_ELEMENTS * NUM_OF_PARTICLE_BLOCKS;

        success &= ( ( ( NS(Blocks_get_num_of_blocks)( &elem_by_elem ) >=
                required_num_elem_by_elem_blocks ) &&
            ( NS(Blocks_get_total_num_bytes)( &elem_by_elem ) > 0 ) &&
            ( NS(Blocks_are_serialized)( &elem_by_elem ) ) ) ||
            ( NS(Blocks_get_num_of_blocks)( &elem_by_elem ) == 0 ) );

        /* ----------------------------------------------------------------- */

        NS(Blocks) particles_buffer;
        NS(Blocks_preset)( &particles_buffer );

        NS(Blocks) calculated_elem_by_elem;
        NS(Blocks_preset)( &calculated_elem_by_elem );

        success &= ( 0 == NS(Blocks_init_from_serialized_data)(
            &particles_buffer,
            NS(Blocks_get_const_data_begin)( &initial_particles_buffer ),
            NS(Blocks_get_total_num_bytes)( &initial_particles_buffer ) ) );

        success &= ( NS(Particles_buffers_have_same_structure)(
            &initial_particles_buffer, &particles_buffer ) );

        success &= ( !NS(Particles_buffers_map_to_same_memory)(
            &initial_particles_buffer, &particles_buffer ) );

        success &= ( NS(Particles_buffers_have_same_structure)(
            &result_particles_buffer, &particles_buffer ) );

        success &= ( !NS(Particles_buffers_map_to_same_memory)(
            &result_particles_buffer, &particles_buffer ) );

        if( NS(Blocks_get_num_of_blocks)( &elem_by_elem ) > 0 )
        {
            int const elem_by_elem_succes =
                NS(Blocks_init_from_serialized_data)(
                    &calculated_elem_by_elem,
                    NS(Blocks_get_const_data_begin)( &elem_by_elem ),
                    NS(Blocks_get_total_num_bytes)( &elem_by_elem ) );

            success &= ( elem_by_elem_succes == 0 );

            NS(Particles_buffer_preset_values)( &calculated_elem_by_elem );
        }

        /* ----------------------------------------------------------------- */

        NS(BlockInfo)* io_block_it = 0;

        NS(block_size_t) const NUM_IO_ELEMENTS_PER_TURN =
            NUM_OF_BEAM_ELEMENTS * NUM_OF_PARTICLE_BLOCKS;

        NS(block_size_t) const AVAILABLE_ELEM_BY_ELEM_BLOCKS =
            NS(Blocks_get_num_of_blocks)( &elem_by_elem );

        if( ( NUM_OF_TURNS * NUM_IO_ELEMENTS_PER_TURN )
                <= AVAILABLE_ELEM_BY_ELEM_BLOCKS )
        {
            io_block_it = NS(Blocks_get_block_infos_begin)(
                &calculated_elem_by_elem );
        }

        bool const use_elem_by_elem_buffer = ( io_block_it != 0 );

        for( NS(block_size_t) ii = 0 ; ii < NUM_OF_TURNS ; ++ii )
        {
            NS(BlockInfo)* part_block_it =
                NS(Blocks_get_block_infos_begin)( &particles_buffer );

            NS(BlockInfo)* part_block_end =
                NS(Blocks_get_block_infos_end)( &particles_buffer );

            success &= ( part_block_it  != 0 );
            success &= ( part_block_end != 0 );

            if( !success )
            {
                break;
            }

            for( ; part_block_it != part_block_end ; ++part_block_it )
            {
                NS(Particles)* io_particles =
                    NS(Blocks_get_particles)( io_block_it );

                NS(Particles)* particles =
                    NS(Blocks_get_particles)( part_block_it );

                NS(block_num_elements_t) const num_of_particles =
                    NS(Particles_get_num_particles)( particles );

                if( ( particles == 0 ) ||
                    ( ( io_block_it  != 0 ) && ( io_particles == 0 ) ) ||
                    ( num_of_particles <= 0 ) )
                {
                    success = false;
                    break;
                }

                if( 0 != NS(Track_beam_elements)( particles, 0,
                        num_of_particles, &beam_elements, io_block_it ) )
                {
                    success = false;
                    break;
                }

                if( ( success ) && ( use_elem_by_elem_buffer ) )
                {
                    SIXTRL_ASSERT( io_block_it != 0 );
                    io_block_it = io_block_it + NUM_IO_ELEMENTS_PER_TURN;
                }
            }
        }

        /* ----------------------------------------------------------------- */

        if( success )
        {
            int cmp_result_particles = NS(Particles_buffer_compare_values)(
                    &result_particles_buffer, &particles_buffer );

            success = ( cmp_result_particles == 0 );

            if( ( success ) && ( use_elem_by_elem_buffer ) )
            {
                success = ( NS(Particles_buffers_have_same_structure)(
                    &elem_by_elem, &calculated_elem_by_elem ) );

                success &= ( !NS(Particles_buffers_map_to_same_memory)(
                    &elem_by_elem, &calculated_elem_by_elem ) );

                int const elem_by_elem_cmp_result =
                    NS(Particles_buffer_compare_values)(
                        &elem_by_elem, &calculated_elem_by_elem );

                success &= ( elem_by_elem_cmp_result == 0 );
            }
        }

        NS(Blocks_free)( &calculated_elem_by_elem );
        NS(Blocks_free)( &particles_buffer );
    }

    /* --------------------------------------------------------------------- */

    NS(Blocks_free)( &elem_by_elem );
    NS(Blocks_free)( &beam_elements );
    NS(Blocks_free)( &initial_particles_buffer );
    NS(Blocks_free)( &result_particles_buffer );

    return success;
}

/* end: sixtracklib/common/tests/test_track_tools.c */
