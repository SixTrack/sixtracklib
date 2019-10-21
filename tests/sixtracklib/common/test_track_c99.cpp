#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"

#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/track/track.h"
#include "sixtracklib/testlib.h"

namespace sixtrack
{
    namespace tests
    {
        bool perform_tracking_test_from_data_file(
            std::string const& path_to_datafile,
            SIXTRL_REAL_T const treshold = SIXTRL_REAL_T{ 0.0 } )
        {
            using real_t          = ::NS(particle_real_t);
            using buffer_t        = ::NS(Buffer);
            using particles_t     = ::NS(Particles);
            using object_t        = ::NS(Object);
            using size_t          = ::NS(buffer_size_t);
            using num_particles_t = ::NS(particle_num_elements_t);

            bool success = false;

            buffer_t* buffer = ::NS(Buffer_new_from_file)(
                path_to_datafile.c_str() );

            size_t const buffer_size = ::NS(Buffer_get_size)( buffer );
            size_t const num_objs = ::NS(Buffer_get_num_of_objects)( buffer );

            success  = ( buffer != nullptr );
            success &= ( buffer_size > size_t{ 0 } );
            success &= ( num_objs    > size_t{ 2 } );

            size_t const num_beam_elements = ( num_objs > size_t{ 2 }  )
                ? ( num_objs - size_t{ 2 } ) : size_t{ 0 };

            if( success )
            {
                object_t* obj_it  = ::NS(Buffer_get_objects_begin)( buffer );
                object_t* obj_end = ::NS(Buffer_get_objects_end)( buffer );

                success  = ( obj_it  != nullptr );
                success &= ( obj_end != nullptr );
                success &= ( std::distance( obj_it, obj_end ) ==
                             static_cast< std::ptrdiff_t >( num_objs ) );

                particles_t* particles = nullptr;
                num_particles_t num_particles = num_particles_t{ 0 };

                if( success )
                {
                    object_t* init_particle_obj = nullptr;
                    init_particle_obj = obj_it++;

                    success = ( ::NS(Object_get_type_id)( init_particle_obj ) ==
                                 ::NS(OBJECT_TYPE_PARTICLE) );

                    success &= ( ::NS(Object_get_begin_addr)(
                        init_particle_obj ) != uintptr_t{ 0 } );

                    if( success )
                    {
                        particles = reinterpret_cast< particles_t* >(
                            static_cast< uintptr_t >( ::NS(Object_get_begin_addr)(
                                init_particle_obj ) ) );

                        success = ( particles != nullptr );

                        num_particles = ::NS(Particles_get_num_of_particles)(
                            particles );

                        success &= ( num_particles > num_particles_t{ 0 } );
                    }
                }

                object_t const*  be_begin = nullptr;
                object_t const*  be_end   = nullptr;

                if( success )
                {
                    be_begin = obj_it;
                    std::advance( obj_it, num_beam_elements );

                    be_end = obj_it;
                }

                particles_t const* cmp_particles = nullptr;

                if( success )
                {
                    object_t const* cmp_particles_obj = obj_it;

                    success = ( ::NS(Object_get_type_id)( cmp_particles_obj ) ==
                                 ::NS(OBJECT_TYPE_PARTICLE) );

                    success &= ( ::NS(Object_get_begin_addr)(
                        cmp_particles_obj ) != uintptr_t{ 0 } );

                    if( success )
                    {
                        cmp_particles = reinterpret_cast< particles_t const* >(
                            static_cast< uintptr_t >( ::NS(Object_get_begin_addr)(
                                cmp_particles_obj ) ) );

                        success  = ( cmp_particles != nullptr );
                        success &= ( num_particles ==
                            ::NS(Particles_get_num_of_particles)(
                                cmp_particles ) );
                    }
                }

                if( success )
                {
                    success = ( ::NS(Particles_have_same_structure)(
                        cmp_particles, particles ) ) ? 1 : 0;

                    success &= ( ::NS(Particles_map_to_same_memory)(
                        cmp_particles, particles ) ) ? 0 : 1;
                }

                if( success )
                {
                    num_particles_t const npart =
                        ::NS(Particles_get_num_of_particles)( particles );

                    for( num_particles_t idx = num_particles_t{ 0 } ;
                            idx < npart ; ++idx )
                    {
                        success = ( ::NS(TRACK_SUCCESS) ==
                            NS(Track_particle_until_turn_objs)(
                                particles, idx, be_begin, be_end, 1u ) );
                    }
                }

                if( success )
                {
                    success =
                    ( ( ::NS(Particles_compare_values)( cmp_particles,
                                                       particles ) == 0 ) ||
                      ( ( treshold > ( real_t )0.0 ) &&
                        ( 0 == ::NS(Particles_compare_values_with_treshold)(
                            cmp_particles, particles, treshold ) ) ) );
                }
            }

            ::NS(Buffer_delete)( buffer );
            return success;

        }

        bool performElementByElementTrackCheck(
            ::NS(Buffer)* SIXTRL_RESTRICT cmp_particles_buffer,
            ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
            double const abs_tolerance )
        {
            using buf_size_t = ::NS(buffer_size_t);

            buf_size_t const num_particle_blocks =
                ::NS(Buffer_get_num_of_objects)( cmp_particles_buffer );

            buf_size_t const num_beam_elements =
                ::NS(Buffer_get_num_of_objects)( beam_elements_buffer );

            bool success = false;
            ::NS(Buffer)* particles_buffer = ::NS(Buffer_new)( 0u );

            if( ( cmp_particles_buffer != nullptr ) &&
                ( beam_elements_buffer != nullptr ) &&
                ( particles_buffer     != nullptr ) &&
                ( num_beam_elements    >  buf_size_t{ 0 } ) &&
                ( num_particle_blocks  >  buf_size_t{ 0 } ) &&
                ( num_particle_blocks >= (
                    num_beam_elements + buf_size_t{ 1 } ) ) &&
                ( abs_tolerance > double{ 0 } ) )
            {
                ::NS(Particles)* particles = nullptr;

                ::NS(Particles) const* input_particles =
                    ::NS(Particles_buffer_get_const_particles)(
                        cmp_particles_buffer, 0u );

                buf_size_t const input_num_particles =
                    ::NS(Particles_get_num_of_particles)( input_particles );

                if( ( input_particles != nullptr ) &&
                    ( input_num_particles > buf_size_t{ 0 } ) )
                {
                    particles = ::NS(Particles_add_copy)(
                        particles_buffer, input_particles );
                }

                if( particles != nullptr )
                {
                    buf_size_t ii = buf_size_t{ 0 };
                    success = true;

                    for(  ; ii < num_beam_elements ; ++ii )
                    {
                        ::NS(Particles) const* cmp_particles =
                            ::NS(Particles_buffer_get_const_particles)(
                                cmp_particles_buffer, ii );

                        if( ( 0 != ::NS(Particles_compare_real_values)(
                                particles, cmp_particles ) ) &&
                            ( 0 != ::NS(Particles_compare_real_values_with_treshold)(
                                particles, cmp_particles, abs_tolerance ) ) )
                        {
                            ::NS(Buffer)* diff_buffer = ::NS(Buffer_new)( 0u );
                            ::NS(Particles)* diff = ::NS(Particles_new)( diff_buffer,
                                ::NS(Particles_get_num_of_particles)( particles ) );

                            ::NS(Particles_calculate_difference)(
                                particles, cmp_particles, diff );

                            ::NS(Object) const* ptr_obj = ::NS(Buffer_get_const_object)(
                                beam_elements_buffer, ii - 1 );

                            std::cout << "ii = " << ii - 1 << std::endl;
                            std::cout << "type_id = " << ::NS(Object_get_type_id)( ptr_obj ) << std::endl;


                            std::cout << "particles: " << std::endl;
                            ::NS(Particles_print_out)( particles );
                            std::cout << std::endl;

                            std::cout << "cmp_particles: " << std::endl;
                            ::NS(Particles_print_out)( cmp_particles );
                            std::cout << std::endl;

                            std::cout << "diff = " << std::endl;
                            ::NS(Particles_print_out)( diff );
                            std::cout << std::endl;

                            ::NS(Buffer_delete)( diff_buffer );

                            success = false;
                            break;
                        }

                        ::NS(Particles_copy)( particles, cmp_particles );

                        ::NS(particle_num_elements_t) const npart =
                            ::NS(Particles_get_num_of_particles)( particles );

                        ::NS(particle_num_elements_t) idx = 0;

                        while( ( success ) && ( idx < npart ) )
                        {
                            ::NS(track_status_t) const status =
                            ::NS(Track_particle_beam_element_obj)(
                                particles, idx++, ::NS(Buffer_get_const_object)(
                                    beam_elements_buffer, ii ) );

                            success = ( status == ::NS(TRACK_SUCCESS) );
                        }
                    }
                }

                if( success )
                {
                    ::NS(Particles) const* cmp_particles =
                        ::NS(Particles_buffer_get_const_particles)(
                            cmp_particles_buffer, num_beam_elements );

                    success = (
                        ( cmp_particles != nullptr ) &&
                        ( ( 0 == ::NS(Particles_compare_real_values)(
                                particles, cmp_particles ) ) ||
                          ( 0 == ::NS(Particles_compare_real_values_with_treshold)(
                                particles, cmp_particles, abs_tolerance ) ) ) );
                }
            }

            ::NS(Buffer_delete)( particles_buffer );

            return success;
        }
    }
}


TEST( C99_CommonTrackTests, TrackParticlesOverDriftBeamElements )
{
    using real_t = SIXTRL_REAL_T;

    static real_t const EPS  = std::numeric_limits< real_t >::epsilon();


    std::string const path_to_datafile =
        ::NS(PATH_TO_TEST_TRACKING_BE_DRIFT_DATA);

    ::FILE* fp = fopen( path_to_datafile.c_str(), "rb" );

    if( fp != nullptr )
    {
        fclose( fp );
        fp = nullptr;

        ASSERT_TRUE( sixtrack::tests::perform_tracking_test_from_data_file(
            path_to_datafile, EPS ) );
    }
    else
    {
        std::cerr << "!!! --> Warning :: tracking dataset "
                  << path_to_datafile << " not available -> "
                  << "skipping tracking unit-test <-- !!!"
                  << std::endl;
    }
}

/* ========================================================================= */

TEST( C99_CommonTrackTests, TrackParticlesOverDriftExactBeamElements )
{
    using real_t = SIXTRL_REAL_T;

    static real_t const EPS  = 5e-14; //std::numeric_limits< real_t >::epsilon();

    std::string const path_to_datafile =
        ::NS(PATH_TO_TEST_TRACKING_BE_DRIFTEXACT_DATA);

    ::FILE* fp = fopen( path_to_datafile.c_str(), "rb" );

    if( fp != nullptr )
    {
        fclose( fp );
        fp = nullptr;

        ASSERT_TRUE( sixtrack::tests::perform_tracking_test_from_data_file(
            path_to_datafile, EPS ) );
    }
    else
    {
        std::cerr << "!!! --> Warning :: tracking dataset "
                  << path_to_datafile << " not available -> "
                  << "skipping tracking unit-test <-- !!!"
                  << std::endl;
    }
}

/* ========================================================================= */

TEST( C99_CommonTrackTests, TrackParticlesOverMultiPoleBeamElements )
{
    using real_t = SIXTRL_REAL_T;

    static real_t const EPS  = std::numeric_limits< real_t >::epsilon();

    std::string const path_to_datafile =
        ::NS(PATH_TO_TEST_TRACKING_BE_MULTIPOLE_DATA);

    ::FILE* fp = fopen( path_to_datafile.c_str(), "rb" );

    if( fp != nullptr )
    {
        fclose( fp );
        fp = nullptr;

        ASSERT_TRUE( sixtrack::tests::perform_tracking_test_from_data_file(
            path_to_datafile, EPS ) );
    }
    else
    {
        std::cerr << "!!! --> Warning :: tracking dataset "
                  << path_to_datafile << " not available -> "
                  << "skipping tracking unit-test <-- !!!"
                  << std::endl;
    }
}

/* ========================================================================= */

TEST( C99_CommonTrackTests, LHCReproduceSixTrackSingleTurnNoBeamBeam )
{
    using size_t   = ::NS(buffer_size_t);
    using object_t = ::NS(Object);

    using ptr_to_cpart_t  = ::NS(Particles) const*;
    using ptr_to_part_t   = ::NS(Particles)*;
    using num_particles_t = ::NS(particle_num_elements_t);
    using real_t          = ::NS(particle_real_t);
    using index_t         = ::NS(particle_index_t);

    static real_t const ABS_TOLERANCE = real_t{ 1e-13 };

    ::NS(Buffer)* pb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_LHC_NO_BB_PARTICLES_SIXTRACK_DUMP) );

    ::NS(Buffer)* eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_LHC_NO_BB_BEAM_ELEMENTS_SIXTRACK) );

    ::NS(Buffer)* track_pb   = ::NS(Buffer_new)( size_t{ 1u << 20u } );
    ::NS(Buffer)* compare_pb = ::NS(Buffer_new)( size_t{ 1u << 20u } );
    ::NS(Buffer)* diff_pb    = ::NS(Buffer_new)( size_t{ 1u << 20u } );

    ASSERT_TRUE( pb != nullptr );
    ASSERT_TRUE( eb != nullptr );

    index_t const num_beam_elements = ::NS(Buffer_get_num_of_objects)( eb );
    index_t const num_particle_sets = ::NS(Buffer_get_num_of_objects)( pb );

    ASSERT_TRUE( num_beam_elements > index_t{ 0 } );
    ASSERT_TRUE( num_particle_sets > index_t{ 0 } );

    object_t const* be_begin = ::NS(Buffer_get_const_objects_begin)( eb );
    object_t const* be_end   = ::NS(Buffer_get_const_objects_end)( eb );

    object_t const* pb_begin = ::NS(Buffer_get_const_objects_begin)( pb );
    object_t const* pb_end   = ::NS(Buffer_get_const_objects_end)( pb );

    ASSERT_TRUE( be_begin != nullptr );
    ASSERT_TRUE( be_end   != nullptr );

    ASSERT_TRUE( pb_begin != nullptr );
    ASSERT_TRUE( pb_end   != nullptr );

    object_t const* pb_it = pb_begin;

    ASSERT_TRUE( ::NS(Object_get_type_id)( pb_it ) == ::NS(OBJECT_TYPE_PARTICLE) );

    ptr_to_cpart_t in_particles = reinterpret_cast< ptr_to_cpart_t >(
        ::NS(Object_get_const_begin_ptr)( pb_it ) );

    ASSERT_TRUE( in_particles != nullptr );

    num_particles_t const in_num_particles =
        ::NS(Particles_get_num_of_particles)( in_particles );

    ASSERT_TRUE( in_num_particles > num_particles_t{ 0 } );

    ptr_to_part_t particles =
        ::NS(Particles_new)( track_pb, in_num_particles );

    ptr_to_part_t cmp_particles =
        ::NS(Particles_new)( compare_pb, in_num_particles );

    ptr_to_part_t diff_particles =
        ::NS(Particles_new)( diff_pb, in_num_particles );

    ASSERT_TRUE( particles      != nullptr );
    ASSERT_TRUE( cmp_particles  != nullptr );
    ASSERT_TRUE( diff_particles != nullptr );

    ASSERT_TRUE( ::NS(Particles_get_num_of_particles)( particles ) ==
                 in_num_particles );

    ASSERT_TRUE( ::NS(Particles_get_num_of_particles)( cmp_particles ) ==
                 in_num_particles );

    ASSERT_TRUE( ::NS(Particles_get_num_of_particles)( diff_particles ) ==
                 in_num_particles );

    object_t const* prev_pb = pb_it++;
    ptr_to_cpart_t  prev_in_particles = nullptr;

    size_t cnt = size_t{ 0 };

    for( ; pb_it != pb_end ; ++pb_it, ++prev_pb, ++cnt )
    {
        ASSERT_TRUE( ::NS(Object_get_const_begin_ptr)( pb_it ) != nullptr );
        ASSERT_TRUE( ::NS(Object_get_size)( pb_it ) >= sizeof( ::NS(Particles) ) );
        ASSERT_TRUE( ::NS(Object_get_type_id)( pb_it ) ==
                     ::NS(OBJECT_TYPE_PARTICLE) );

        prev_in_particles = in_particles;

        in_particles = reinterpret_cast< ptr_to_cpart_t >(
            ::NS(Object_get_const_begin_ptr)( pb_it ) );

        ASSERT_TRUE( ::NS(Particles_get_num_of_particles)( in_particles ) ==
                     in_num_particles );

        ::NS(Particles_copy)( particles, prev_in_particles );
        ::NS(Particles_copy)( cmp_particles,  in_particles );

        for( num_particles_t ii = 0 ; ii < in_num_particles ; ++ii )
        {
            ASSERT_TRUE(
                ( ii == 0 ) ||
                ( ::NS(Particles_get_particle_id_value)( particles,  0 ) !=
                  ::NS(Particles_get_particle_id_value)( particles, ii ) ) );

            ASSERT_TRUE(
                ::NS(Particles_get_at_element_id_value)( particles,  0 ) ==
                ::NS(Particles_get_at_element_id_value)( particles, ii ) );

            ASSERT_TRUE(
                ::NS(Particles_get_at_element_id_value)( cmp_particles, 0 ) ==
                ::NS(Particles_get_at_element_id_value)( cmp_particles, ii ) );


            ASSERT_TRUE( ::NS(Particles_get_at_turn_value)( particles,  0 ) ==
                         ::NS(Particles_get_at_turn_value)( particles, ii ) );

            ASSERT_TRUE(
                ::NS(Particles_get_particle_id_value)( particles, ii ) ==
                ::NS(Particles_get_particle_id_value)( cmp_particles, ii ) );

            ASSERT_TRUE(
                ::NS(Particles_get_at_turn_value)( particles, 0 ) ==
                ::NS(Particles_get_at_turn_value)( cmp_particles, ii ) );

            ASSERT_TRUE(
                ::NS(Particles_get_at_turn_value)( cmp_particles, 0 ) ==
                ::NS(Particles_get_at_turn_value)( cmp_particles, ii ) );
        }

        index_t const begin_elem_id = ::NS(Particles_get_at_element_id_value)(
            particles, num_particles_t{ 0 } );

        index_t const end_elem_id   = ::NS(Particles_get_at_element_id_value)(
            cmp_particles, num_particles_t{ 0 } );

        object_t const* line_begin = be_begin;
        std::advance( line_begin, begin_elem_id + index_t{ 1 } );

        object_t const* line_end = be_begin;
        std::advance( line_end, end_elem_id + index_t{ 1 } );

        ::NS(particle_num_elements_t) const npart =
            ::NS(Particles_get_num_of_particles)( particles );

        for( ::NS(particle_num_elements_t) idx = 0 ; idx < npart ; ++idx )
        {
            ::NS(track_status_t) const status = ::NS(Track_particle_line_objs)(
                particles, idx, line_begin, line_end, false );

            ASSERT_TRUE( status == ::NS(TRACK_SUCCESS) );
        }

        ::NS(Particles_calculate_difference)(
            cmp_particles, particles, diff_particles );

        bool is_equal = true;

        for( num_particles_t ii = 0 ; ii < in_num_particles ; ++ii )
        {
            if( ( ABS_TOLERANCE < std::fabs( ::NS(Particles_get_s_value)( diff_particles, ii ) ) ) ||
                ( ABS_TOLERANCE < std::fabs( ::NS(Particles_get_x_value)( diff_particles, ii ) ) ) ||
                ( ABS_TOLERANCE < std::fabs( ::NS(Particles_get_y_value)( diff_particles, ii ) ) ) ||
                ( ABS_TOLERANCE < std::fabs( ::NS(Particles_get_px_value)( diff_particles, ii ) ) ) ||
                ( ABS_TOLERANCE < std::fabs( ::NS(Particles_get_py_value)( diff_particles, ii ) ) ) ||
                ( ABS_TOLERANCE < std::fabs( ::NS(Particles_get_zeta_value)( diff_particles, ii ) ) ) ||
                ( ABS_TOLERANCE < std::fabs( ::NS(Particles_get_psigma_value)( diff_particles, ii ) ) ) ||
                ( ABS_TOLERANCE < std::fabs( ::NS(Particles_get_delta_value)( diff_particles, ii ) ) ) ||
                ( ABS_TOLERANCE < std::fabs( ::NS(Particles_get_rpp_value)( diff_particles, ii ) ) ) ||
                ( ABS_TOLERANCE < std::fabs( ::NS(Particles_get_rvv_value)( diff_particles, ii ) ) ) ||
                ( ABS_TOLERANCE < std::fabs( ::NS(Particles_get_chi_value)( diff_particles, ii ) ) ) )
            {
                is_equal = false;
                break;
            }
        }

        if( !is_equal )
        {
            std::cout << "Difference between tracked particles and "
                         "reference particle data detected: \r\n"
                      << "at beam-element block #" << cnt
                      << ", concerning beam-elements [ "
                      << std::setw( 6 ) << begin_elem_id + 1 << " - "
                      << std::setw( 6 ) << end_elem_id + 1 << " ):\r\n"
                      << "absolute tolerance : " << ABS_TOLERANCE << "\r\n"
                      << "\r\n"
                      << "beam-elements: \r\n";

            object_t const* line_it  = line_begin;
            size_t jj = begin_elem_id + index_t{ 1 };

            for( ; line_it != line_end ; ++line_it )
            {
                std::cout << "be id = " << std::setw( 6 ) << jj ;
                NS(BeamElement_print_out)( line_it );
            }

            std::cout << "\r\n"
                      << "diff_particles = |cmp_particles - particles| :\r\n";

            NS(Particles_print)( stdout, diff_particles );

            std::cout << std::endl;
        }

        for( num_particles_t ii = 0 ; ii < in_num_particles ; ++ii )
        {
            ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                ::NS(Particles_get_s_value)( diff_particles, ii ) ) );

            ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                ::NS(Particles_get_x_value)( diff_particles, ii ) ) );

            ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                ::NS(Particles_get_y_value)( diff_particles, ii ) ) );

            ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                ::NS(Particles_get_px_value)( diff_particles, ii ) ) );

            ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                ::NS(Particles_get_py_value)( diff_particles, ii ) ) );

            ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                ::NS(Particles_get_zeta_value)( diff_particles, ii ) ) );

            ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                ::NS(Particles_get_psigma_value)( diff_particles, ii ) ) );

            ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                ::NS(Particles_get_delta_value)( diff_particles, ii ) ) );

            ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                ::NS(Particles_get_rpp_value)( diff_particles, ii ) ) );

            ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                ::NS(Particles_get_rvv_value)( diff_particles, ii ) ) );

            ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                ::NS(Particles_get_chi_value)( diff_particles, ii ) ) );

            ASSERT_TRUE( ::NS(Particles_get_particle_id_value)(
                diff_particles, ii ) == index_t{ 0 } );
        }
    }

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( diff_pb );
    ::NS(Buffer_delete)( track_pb );
    ::NS(Buffer_delete)( compare_pb );
}

TEST( C99_CommonTrackTests, LHCReproducePySixTrackSingleTurnBBSimple )
{
    using real_t = ::NS(particle_real_t);
    namespace stests = sixtrack::tests;

    static real_t const ABS_TOLERANCE = real_t{ 1e-15 };

    ::NS(Buffer)* cmp_particles_buffer =
        ::NS(Buffer_new_from_file)( ::NS(PATH_TO_BBSIMPLE_PARTICLES_DUMP) );

    ::NS(Buffer)* beam_elements_buffer =
        ::NS(Buffer_new_from_file)( ::NS(PATH_TO_BBSIMPLE_BEAM_ELEMENTS) );

    ASSERT_TRUE( sixtrack::tests::performElementByElementTrackCheck(
        cmp_particles_buffer, beam_elements_buffer, ABS_TOLERANCE ) );

    ::NS(Buffer_delete)( cmp_particles_buffer );
    ::NS(Buffer_delete)( beam_elements_buffer );
}

TEST( C99_CommonTrackTests, LHCReproducePySixTrackSingleTurnBeamBeam )
{
    using real_t = ::NS(particle_real_t);
    namespace stests = SIXTRL_CXX_NAMESPACE::tests;

    static real_t const ABS_TOLERANCE = real_t{ 1e-15 };

    ::NS(Buffer)* cmp_particles_buffer =
        ::NS(Buffer_new_from_file)( ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    ::NS(Buffer)* beam_elements_buffer =
        ::NS(Buffer_new_from_file)( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    ASSERT_TRUE( sixtrack::tests::performElementByElementTrackCheck(
        cmp_particles_buffer, beam_elements_buffer, ABS_TOLERANCE ) );

    ::NS(Buffer_delete)( cmp_particles_buffer );
    ::NS(Buffer_delete)( beam_elements_buffer );
}

TEST( C99_CommonTrackTests, LHCReproducePySixTrackSingleTurnLhcNoBB )
{
    using real_t = ::NS(particle_real_t);
    namespace stests = SIXTRL_CXX_NAMESPACE::tests;

    static real_t const ABS_TOLERANCE = real_t{ 1e-15 };

    ::NS(Buffer)* cmp_particles_buffer =
        ::NS(Buffer_new_from_file)( ::NS(PATH_TO_LHC_NO_BB_PARTICLES_DUMP) );

    ::NS(Buffer)* beam_elements_buffer =
        ::NS(Buffer_new_from_file)( ::NS(PATH_TO_LHC_NO_BB_BEAM_ELEMENTS) );

    ASSERT_TRUE( sixtrack::tests::performElementByElementTrackCheck(
        cmp_particles_buffer, beam_elements_buffer, ABS_TOLERANCE ) );

    ::NS(Buffer_delete)( cmp_particles_buffer );
    ::NS(Buffer_delete)( beam_elements_buffer );
}

/* end: tests/sixtracklib/common/test_track_c99.cpp */
