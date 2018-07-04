#ifndef SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TEST_TRACK_TOOLS_H__
#define SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TEST_TRACK_TOOLS_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
    #else
        #include <stdbool.h>
        #include <stddef.h>
        #include <stdint.h>
        #include <stdlib.h>
    #endif /* __cplusplus */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/blocks.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/impl/particles_api.h"

    #include "sixtracklib/testlib/test_particles_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

bool NS(Tracks_store_testdata_to_binary_file)(
        uint64_t const num_of_turns,
        NS(Blocks) const* SIXTRL_RESTRICT initial_particles_buffer,
        NS(Blocks) const* SIXTRL_RESTRICT result_particles_buffer,
        NS(Blocks) const* SIXTRL_RESTRICT beam_elements_buffer,
        NS(Blocks) const* SIXTRL_RESTRICT ptr_elem_by_elem_buffer,
        char const path_to_bin_testdata_file[] );

bool NS(Tracks_restore_testdata_from_binary_file)(
    char const path_to_bin_testdata_file[],
    uint64_t* ptr_num_of_turns,
    NS(Blocks)* SIXTRL_RESTRICT initial_particles_buffer,
    NS(Blocks)* SIXTRL_RESTRICT result_particles_buffer,
    NS(Blocks)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Blocks)* SIXTRL_RESTRICT elem_by_elem_buffer );

bool NS(TestData_test_tracking_single_particle)(
    char const *const SIXTRL_RESTRICT path_to_testdata_file );

bool NS(TestData_test_tracking_particles)(
    char const *const SIXTRL_RESTRICT path_to_testdata_file );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}

template< class BeamElement >
bool NS(TestData_test_tracking_single_particle_over_specific_be_type)(
    char const *const SIXTRL_RESTRICT path_to_testdata_file,
    SIXTRL_TRACK_RETURN (*tracking_fn)(
        NS(Particles)* SIXTRL_RESTRICT,
        NS(block_num_elements_t) const,
        const BeamElement *const SIXTRL_RESTRICT ) );

template< class BeamElement >
bool NS(TestData_test_tracking_particles_over_specific_be_type)(
    char const *const SIXTRL_RESTRICT path_to_testdata_file,
    SIXTRL_TRACK_RETURN (*tracking_fn)(
        NS(Particles)* SIXTRL_RESTRICT,
        NS(block_num_elements_t) const,
        NS(block_num_elements_t) const,
        const BeamElement *const SIXTRL_RESTRICT,
        NS(Particles)* SIXTRL_RESTRICT io_particles ) );


template< class BeamElement >
bool NS(TestData_test_tracking_single_particle_over_specific_be_type)(
    char const *const SIXTRL_RESTRICT path_to_testdata_file,
    SIXTRL_TRACK_RETURN (*tracking_fn)(
        NS(Particles)* SIXTRL_RESTRICT,
        NS(block_num_elements_t) const,
        const BeamElement *const SIXTRL_RESTRICT ) )
{
    bool success = false;

    uint64_t NUM_OF_TURNS = uint64_t{ 0 };

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
        success &= ( NUM_OF_BEAM_ELEMENTS > NS(block_size_t){ 0 } );

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

        NS(BlockInfo)* io_block_it = nullptr;

        NS(block_size_t) const AVAILABLE_ELEM_BY_ELEM_BLOCKS =
            NS(Blocks_get_num_of_blocks)( &elem_by_elem );

        NS(block_size_t) const NUM_IO_ELEMENTS_PER_TURN =
            NUM_OF_BEAM_ELEMENTS * NUM_OF_PARTICLE_BLOCKS;

        if( ( NUM_OF_TURNS * NUM_IO_ELEMENTS_PER_TURN )
                <= AVAILABLE_ELEM_BY_ELEM_BLOCKS )
        {
            io_block_it = NS(Blocks_get_block_infos_begin)(
                &calculated_elem_by_elem );
        }

        bool const use_elem_by_elem_buffer = ( io_block_it != nullptr );

        for( NS(block_size_t) ii = 0 ; ii < NUM_OF_TURNS ; ++ii )
        {
            NS(BlockInfo) const* be_info_it =
                NS(Blocks_get_const_block_infos_begin)( &beam_elements );

            if( ( !success ) || ( be_info_it == nullptr ) )
            {
                success = false;
                break;
            }

            NS(block_size_t) jj = 0;

            for( ; jj < NUM_OF_BEAM_ELEMENTS ; ++jj, ++be_info_it )
            {
                NS(BlockInfo)* particles_block_it =
                    NS(Blocks_get_block_infos_begin)( &particles_buffer );

                NS(BlockInfo)* particles_block_end =
                    NS(Blocks_get_block_infos_end)( &particles_buffer );

                success &= ( particles_block_it  != nullptr );
                success &= ( particles_block_end != nullptr );

                BeamElement const* beam_element =
                    static_cast< BeamElement const* >(
                        NS(BlockInfo_get_const_ptr_begin)( be_info_it ) );

                if( ( !success ) || ( beam_element == nullptr ) )
                {
                    success = false;
                    break;
                }

                for( ; particles_block_it != particles_block_end ;
                        ++particles_block_it )
                {
                    NS(Particles)* io_particles =
                        NS(Blocks_get_particles)( io_block_it );

                    NS(Particles)* particles =
                        NS(Blocks_get_particles)( particles_block_it );

                    NS(block_num_elements_t) const num_of_particles =
                        NS(Particles_get_num_particles)( particles );

                    if( ( particles == nullptr ) ||
                        ( ( io_block_it  != nullptr ) &&
                        ( io_particles == nullptr ) ) ||
                        ( num_of_particles <= 0 ) )
                    {
                        success = false;
                        break;
                    }

                    NS(block_num_elements_t) ll = 0;

                    for(  ; ll < num_of_particles ; ++ll )
                    {
                        if( 0 != tracking_fn( particles, ll, beam_element ) )
                        {
                            success = false;
                            break;
                        }
                    }

                    if( ( success ) && ( io_particles != 0 ) )
                    {
                        SIXTRL_ASSERT( io_block_it != nullptr );

                        NS(Particles_copy_all_unchecked)(
                            io_particles, particles );

                        ++io_block_it;
                    }
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

template< class BeamElement >
bool NS(TestData_test_tracking_particles_over_specific_be_type)(
    char const *const SIXTRL_RESTRICT path_to_testdata_file,
    SIXTRL_TRACK_RETURN (*tracking_fn)(
        NS(Particles)* SIXTRL_RESTRICT,
        NS(block_num_elements_t) const,
        NS(block_num_elements_t) const,
        const BeamElement *const SIXTRL_RESTRICT,
        NS(Particles)* SIXTRL_RESTRICT io_particles ) )
{
    bool success = false;

    uint64_t NUM_OF_TURNS = uint64_t{ 0 };

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
        success &= ( NUM_OF_BEAM_ELEMENTS > NS(block_size_t){ 0 } );

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

        NS(BlockInfo)* io_block_it = nullptr;

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

        bool const use_elem_by_elem_buffer = ( io_block_it != nullptr );

        for( NS(block_size_t) ii = 0 ; ii < NUM_OF_TURNS ; ++ii )
        {
            NS(BlockInfo) const* be_info_it =
                NS(Blocks_get_const_block_infos_begin)( &beam_elements );

            if( ( !success ) || ( be_info_it == nullptr ) )
            {
                success = false;
                break;
            }

            NS(block_size_t) jj = 0;

            for( ; jj < NUM_OF_BEAM_ELEMENTS ; ++jj, ++be_info_it )
            {
                NS(BlockInfo)* particles_block_it =
                    NS(Blocks_get_block_infos_begin)( &particles_buffer );

                NS(BlockInfo)* particles_block_end =
                    NS(Blocks_get_block_infos_end)( &particles_buffer );

                success &= ( particles_block_it  != nullptr );
                success &= ( particles_block_end != nullptr );

                BeamElement const* beam_element =
                    static_cast< BeamElement const* >(
                        NS(BlockInfo_get_const_ptr_begin)( be_info_it ) );

                if( ( !success ) || ( beam_element == nullptr ) )
                {
                    success = false;
                    break;
                }

                for( ; particles_block_it != particles_block_end ;
                        ++particles_block_it )
                {
                    NS(Particles)* io_particles =
                        NS(Blocks_get_particles)( io_block_it );

                    NS(Particles)* particles =
                        NS(Blocks_get_particles)( particles_block_it );

                    NS(block_num_elements_t) const num_of_particles =
                        NS(Particles_get_num_particles)( particles );

                    if( ( particles == nullptr ) ||
                        ( ( io_block_it  != nullptr ) &&
                        ( io_particles == nullptr ) ) ||
                        ( num_of_particles <= 0u ) )
                    {
                        success = false;
                        break;
                    }

                    if( 0 != tracking_fn( particles, 0, num_of_particles,
                                        beam_element, io_particles ) )
                    {
                        success = false;
                        break;
                    }

                    if( io_block_it != 0 )
                    {
                        ++io_block_it;
                    }
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
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TEST_TRACK_TOOLS_H__ */

/* end: tests/sixtracklib/testlib/test_track_tools.h */

