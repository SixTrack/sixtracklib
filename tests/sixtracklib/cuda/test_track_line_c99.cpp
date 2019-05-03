#include "sixtracklib/cuda/argument.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/cuda/controller.h"
#include "sixtracklib/cuda/wrappers/track_particles.h"
#include "sixtracklib/common/track.h"


TEST( C99_CudaTrackLineTests, SendDataTrackSingleLineRecvDataCompare )
{
    using buffer_t    = ::NS(Buffer)*;
    using particles_t = ::NS(Particles)*;
    using buf_size_t  = ::NS(buffer_size_t);
    using status_t    = ::NS(controller_status_t);

    buffer_t pb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    buffer_t eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    buffer_t diff_buffer  = ::NS(Buffer_new)( buf_size_t{ 0 } );
    buffer_t track_pb     = ::NS(Buffer_new)( buf_size_t{ 0 } );
    buffer_t cmp_track_pb = ::NS(Buffer_new)( buf_size_t{ 0 } );

    particles_t particles = ::NS(Particles_add_copy)( track_pb,
        ::NS(Particles_buffer_get_const_particles)( pb, 0 ) );

    particles_t cmp_particles = ::NS(Particles_add_copy)( cmp_track_pb,
        ::NS(Particles_buffer_get_const_particles)( pb, 0 ) );

    SIXTRL_ASSERT( particles != nullptr );
    SIXTRL_ASSERT( cmp_particles != nullptr );

    buf_size_t const until_turn = 10;
    int status = ::NS(Track_all_particles_until_turn)(
        cmp_particles, eb, until_turn );

    SIXTRL_ASSERT( status == 0 );

    ::NS(CudaController)* controller = ::NS(CudaController_create)();
    ASSERT_TRUE( controller != nullptr );

    ::NS(CudaArgument)* lattice_arg = ::NS(CudaArgument_new)( controller );
    ASSERT_TRUE( lattice_arg != nullptr );

    status_t success = ::NS(CudaArgument_send_buffer)( lattice_arg, eb );
    ASSERT_TRUE( success == ::NS(CONTROLLER_STATUS_SUCCESS) );

    ::NS(CudaArgument)* particles_arg = ::NS(CudaArgument_new)( controller );
    ASSERT_TRUE( particles_arg != nullptr );

    success = ::NS(CudaArgument_send_buffer)( particles_arg, track_pb );
    ASSERT_TRUE( success == ::NS(CONTROLLER_STATUS_SUCCESS) );

    buf_size_t const num_beam_elements = ::NS(Buffer_get_num_of_objects)( eb );
    buf_size_t const num_lattice_parts = buf_size_t{ 10 };
    buf_size_t const num_elem_per_part = num_beam_elements / num_lattice_parts;

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < until_turn ; ++ii )
    {
        for( buf_size_t jj = buf_size_t{ 0 } ; jj < num_lattice_parts ; ++jj )
        {
            bool const is_last_in_turn = ( jj == ( num_lattice_parts - 1 ) );
            buf_size_t const begin_idx =  jj * num_elem_per_part;
            buf_size_t const end_idx   = ( !is_last_in_turn ) ?
                begin_idx + num_elem_per_part : num_beam_elements;

            status = NS(Track_particles_line_cuda_on_grid)(
                NS(CudaArgument_get_cuda_arg_buffer)( particles_arg ),
                NS(CudaArgument_get_cuda_arg_buffer)( lattice_arg ),
                begin_idx, end_idx, is_last_in_turn, 1, 1 );

            ASSERT_TRUE( status == 0 );
        }
    }

    success = NS(CudaArgument_receive_buffer)( particles_arg, track_pb );
    ASSERT_TRUE( success == ::NS(CONTROLLER_STATUS_SUCCESS) );

    particles = ::NS(Particles_buffer_get_particles)( track_pb, 0 );

    double const ABS_DIFF = double{ 2e-14 };

    if( ( 0 != ::NS(Particles_compare_values)( cmp_particles, particles ) ) &&
        ( 0 != ::NS(Particles_compare_values_with_treshold)(
            cmp_particles, particles, ABS_DIFF ) ) )
    {
        particles_t diff = ::NS(Particles_new)( diff_buffer,
            NS(Particles_get_num_of_particles)( cmp_particles ) );

        ::NS(Particles_calculate_difference)( particles, cmp_particles, diff );

        printf( "particles: \r\n" );
        ::NS(Particles_print_out)( particles );

        printf( "cmp_particles: \r\n" );
        ::NS(Particles_print_out)( cmp_particles );

        printf( "diff: \r\n" );
        ::NS(Particles_print_out)( diff );
    }

    ASSERT_TRUE(
        ( 0 == ::NS(Particles_compare_values)( cmp_particles, particles ) ) ||
        ( 0 == ::NS(Particles_compare_values_with_treshold)(
            cmp_particles, particles, ABS_DIFF ) ) );

    ::NS(CudaArgument_delete)( lattice_arg );
    ::NS(CudaArgument_delete)( particles_arg );
    ::NS(CudaController_delete)( controller );

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( track_pb );
    ::NS(Buffer_delete)( cmp_track_pb );
}

/* end: tests/sixtracklib/cuda/test_track_line_c99.cpp */
