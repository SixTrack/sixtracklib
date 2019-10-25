#define _USE_MATH_DEFINES

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/opencl/context.h"
#include "sixtracklib/opencl/argument.h"

/* ************************************************************************* */

namespace SIXTRL_CXX_NAMESPACE
{
    namespace tests
    {
        bool ClContext_perform_tracking_tests(
            ::NS(ClContext)* SIXTRL_RESTRICT context,
            ::NS(Buffer) const* SIXTRL_RESTRICT in_particles_buffer,
            ::NS(Buffer) const* SIXTRL_RESTRICT in_beam_elements_buffer,
            double const abs_treshold = double{ 1e-16 } );
    }
}

/* ************************************************************************* */

TEST( C99_OpenCL_TrackParticlesTests, LHCReproduceSixTrackSingleTurnNoBeamBeamDebug )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    static double const ABS_TOLERANCE = double{ 1e-13 };

    ::NS(Buffer)* lhc_particles_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_LHC_NO_BB_PARTICLES_SIXTRACK_DUMP) );

    ::NS(Buffer)* lhc_beam_elements_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_LHC_NO_BB_BEAM_ELEMENTS_SIXTRACK) );

    ::NS(ClContext)* context = ::NS(ClContext_create)();

    ASSERT_TRUE( context != nullptr );

    std::size_t const num_available_nodes =
        ::NS(ClContextBase_get_num_available_nodes)( context );

    ::NS(ClContext_delete)( context );
    context = nullptr;

    for( std::size_t nn = std::size_t{ 0 } ; nn < num_available_nodes ; ++nn )
    {
        context = ::NS(ClContext_create)();
        ::NS(ClContextBase_enable_debug_mode)( context );
        ::NS(ClContext_disable_optimized_tracking_by_default)( context );

        ASSERT_TRUE(  ::NS(ClContextBase_is_debug_mode_enabled)( context ) );
        ASSERT_TRUE( !::NS(ClContext_uses_optimized_tracking_by_default)( context ) );

        ASSERT_TRUE(  ::NS(ClContextBase_select_node_by_index)( context, nn ) );
        ASSERT_TRUE(  ::NS(ClContextBase_has_selected_node)( context ) );

        ::NS(context_node_info_t) const* node_info =
            ::NS(ClContextBase_get_selected_node_info)( context );

        ASSERT_TRUE( node_info != nullptr );
        ASSERT_TRUE( ::NS(ClContext_has_track_until_kernel)( context ) );
        ASSERT_TRUE( ::NS(ClContextBase_has_remapping_kernel)( context ) );

        char id_str[ 32 ];
        ::NS(ClContextBase_get_selected_node_id_str)( context, id_str, 32 );

        std::cout << "# ------------------------------------------------------"
                  << "--------------------------------------------------------"
                  << "\r\n"
                  << "# Run Test on :: \r\n"
                  << "# ID          :: " << id_str << "\r\n"
                  << "# NAME        :: "
                  << ::NS(ComputeNodeInfo_get_name)( node_info ) << "\r\n"
                  << "# PLATFORM    :: "
                  << ::NS(ComputeNodeInfo_get_platform)( node_info ) << "\r\n"
                  << "# "
                  << std::endl;

        ASSERT_TRUE( st::tests::ClContext_perform_tracking_tests(
            context, lhc_particles_buffer, lhc_beam_elements_buffer,
                ABS_TOLERANCE ) );

        ::NS(ClContext_delete)( context );
        context = nullptr;
    }

    if( num_available_nodes == std::size_t{ 0 } )
    {
        std::cout << "Skipping unit-test because no "
                  << "OpenCL platforms have been found --> "
                  << "NEITHER PASSED NOR FAILED!"
                  << std::endl;
    }

    ::NS(Buffer_delete)( lhc_particles_buffer );
    ::NS(Buffer_delete)( lhc_beam_elements_buffer );
}

/* ************************************************************************* */

TEST( C99_OpenCL_TrackParticlesTests,
      LHCReproduceSixTrackSingleTurnNoBeamBeamPrivParticlesOptimizedDebug )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    static double const ABS_TOLERANCE = double{ 1e-13 };

    ::NS(Buffer)* lhc_particles_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_LHC_NO_BB_PARTICLES_SIXTRACK_DUMP) );

    ::NS(Buffer)* lhc_beam_elements_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_LHC_NO_BB_BEAM_ELEMENTS_SIXTRACK) );

    ::NS(ClContext)* context = ::NS(ClContext_create)();

    ASSERT_TRUE( context != nullptr );

    std::size_t const num_available_nodes =
        ::NS(ClContextBase_get_num_available_nodes)( context );

    ::NS(ClContext_delete)( context );
    context = nullptr;

    for( std::size_t nn = std::size_t{ 0 } ; nn < num_available_nodes ; ++nn )
    {
        context = ::NS(ClContext_create)();
        ::NS(ClContextBase_enable_debug_mode)( context );
        ::NS(ClContext_enable_optimized_tracking_by_default)( context );

        ASSERT_TRUE( ::NS(ClContextBase_is_debug_mode_enabled)( context ) );
        ASSERT_TRUE( ::NS(ClContext_uses_optimized_tracking_by_default)( context ) );

        ASSERT_TRUE( ::NS(ClContextBase_select_node_by_index)( context, nn ) );
        ASSERT_TRUE( ::NS(ClContextBase_has_selected_node)( context ) );

        ::NS(context_node_info_t) const* node_info =
            ::NS(ClContextBase_get_selected_node_info)( context );

        ASSERT_TRUE( node_info != nullptr );
        ASSERT_TRUE( ::NS(ClContext_has_track_until_kernel)( context ) );
        ASSERT_TRUE( ::NS(ClContextBase_has_remapping_kernel)( context ) );

        char id_str[ 32 ];
        ::NS(ClContextBase_get_selected_node_id_str)( context, id_str, 32 );

        std::cout << "# ------------------------------------------------------"
                  << "--------------------------------------------------------"
                  << "\r\n"
                  << "# Run Test on :: \r\n"
                  << "# ID          :: " << id_str << "\r\n"
                  << "# NAME        :: "
                  << ::NS(ComputeNodeInfo_get_name)( node_info ) << "\r\n"
                  << "# PLATFORM    :: "
                  << ::NS(ComputeNodeInfo_get_platform)( node_info ) << "\r\n"
                  << "# "
                  << std::endl;

        ASSERT_TRUE( st::tests::ClContext_perform_tracking_tests(
            context, lhc_particles_buffer, lhc_beam_elements_buffer,
                ABS_TOLERANCE ) );

        ::NS(ClContext_delete)( context );
        context = nullptr;
    }

    if( num_available_nodes == std::size_t{ 0 } )
    {
        std::cout << "Skipping unit-test because no "
                  << "OpenCL platforms have been found --> "
                  << "NEITHER PASSED NOR FAILED!"
                  << std::endl;
    }

    ::NS(Buffer_delete)( lhc_particles_buffer );
    ::NS(Buffer_delete)( lhc_beam_elements_buffer );
}


TEST( C99_OpenCL_TrackParticlesTests, LHCReproduceSixTrackSingleTurnNoBeamBeam )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    static double const ABS_TOLERANCE = double{ 1e-13 };

    ::NS(Buffer)* lhc_particles_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_LHC_NO_BB_PARTICLES_SIXTRACK_DUMP) );

    ::NS(Buffer)* lhc_beam_elements_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_LHC_NO_BB_BEAM_ELEMENTS_SIXTRACK) );

    ::NS(ClContext)* context = ::NS(ClContext_create)();

    ASSERT_TRUE( context != nullptr );

    std::size_t const num_available_nodes =
        ::NS(ClContextBase_get_num_available_nodes)( context );

    ::NS(ClContext_delete)( context );
    context = nullptr;

    for( std::size_t nn = std::size_t{ 0 } ; nn < num_available_nodes ; ++nn )
    {
        context = ::NS(ClContext_create)();
        ::NS(ClContextBase_disable_debug_mode)( context );
        ::NS(ClContext_disable_optimized_tracking_by_default)( context );

        ASSERT_TRUE( !::NS(ClContextBase_is_debug_mode_enabled)( context ) );
        ASSERT_TRUE( !::NS(ClContext_uses_optimized_tracking_by_default)( context ) );

        ASSERT_TRUE(  ::NS(ClContextBase_select_node_by_index)( context, nn ) );
        ASSERT_TRUE(  ::NS(ClContextBase_has_selected_node)( context ) );

        ::NS(context_node_info_t) const* node_info =
            ::NS(ClContextBase_get_selected_node_info)( context );

        ASSERT_TRUE( node_info != nullptr );
        ASSERT_TRUE( ::NS(ClContext_has_track_until_kernel)( context ) );
        ASSERT_TRUE( ::NS(ClContextBase_has_remapping_kernel)( context ) );

        char id_str[ 32 ];
        ::NS(ClContextBase_get_selected_node_id_str)( context, id_str, 32 );

        std::cout << "# ------------------------------------------------------"
                  << "--------------------------------------------------------"
                  << "\r\n"
                  << "# Run Test on :: \r\n"
                  << "# ID          :: " << id_str << "\r\n"
                  << "# NAME        :: "
                  << ::NS(ComputeNodeInfo_get_name)( node_info ) << "\r\n"
                  << "# PLATFORM    :: "
                  << ::NS(ComputeNodeInfo_get_platform)( node_info ) << "\r\n"
                  << "# "
                  << std::endl;

        ASSERT_TRUE( st::tests::ClContext_perform_tracking_tests(
            context, lhc_particles_buffer, lhc_beam_elements_buffer,
                ABS_TOLERANCE ) );

        ::NS(ClContext_delete)( context );
        context = nullptr;
    }

    if( num_available_nodes == std::size_t{ 0 } )
    {
        std::cout << "Skipping unit-test because no "
                  << "OpenCL platforms have been found --> "
                  << "NEITHER PASSED NOR FAILED!"
                  << std::endl;
    }

    ::NS(Buffer_delete)( lhc_particles_buffer );
    ::NS(Buffer_delete)( lhc_beam_elements_buffer );
}

/* ************************************************************************* */

TEST( C99_OpenCL_TrackParticlesTests,
      LHCReproduceSixTrackSingleTurnNoBeamBeamPrivParticlesOptimized )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    static double const ABS_TOLERANCE = double{ 1e-13 };

    ::NS(Buffer)* lhc_particles_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_LHC_NO_BB_PARTICLES_SIXTRACK_DUMP) );

    ::NS(Buffer)* lhc_beam_elements_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_LHC_NO_BB_BEAM_ELEMENTS_SIXTRACK) );

    ::NS(ClContext)* context = ::NS(ClContext_create)();

    ASSERT_TRUE( context != nullptr );

    std::size_t const num_available_nodes =
        ::NS(ClContextBase_get_num_available_nodes)( context );

    ::NS(ClContext_delete)( context );
    context = nullptr;

    for( std::size_t nn = std::size_t{ 0 } ; nn < num_available_nodes ; ++nn )
    {
        context = ::NS(ClContext_create)();
        ::NS(ClContextBase_disable_debug_mode)( context );
        ::NS(ClContext_enable_optimized_tracking_by_default)( context );

        ASSERT_TRUE( !::NS(ClContextBase_is_debug_mode_enabled)( context ) );
        ASSERT_TRUE(  ::NS(ClContext_uses_optimized_tracking_by_default)( context ) );

        ASSERT_TRUE(  ::NS(ClContextBase_select_node_by_index)( context, nn ) );
        ASSERT_TRUE(  ::NS(ClContextBase_has_selected_node)( context ) );

        ::NS(context_node_info_t) const* node_info =
            ::NS(ClContextBase_get_selected_node_info)( context );

        ASSERT_TRUE( node_info != nullptr );
        ASSERT_TRUE( ::NS(ClContext_has_track_until_kernel)( context ) );
        ASSERT_TRUE( ::NS(ClContextBase_has_remapping_kernel)( context ) );

        char id_str[ 32 ];
        ::NS(ClContextBase_get_selected_node_id_str)( context, id_str, 32 );

        std::cout << "# ------------------------------------------------------"
                  << "--------------------------------------------------------"
                  << "\r\n"
                  << "# Run Test on :: \r\n"
                  << "# ID          :: " << id_str << "\r\n"
                  << "# NAME        :: "
                  << ::NS(ComputeNodeInfo_get_name)( node_info ) << "\r\n"
                  << "# PLATFORM    :: "
                  << ::NS(ComputeNodeInfo_get_platform)( node_info ) << "\r\n"
                  << "# "
                  << std::endl;

        ASSERT_TRUE( st::tests::ClContext_perform_tracking_tests(
            context, lhc_particles_buffer, lhc_beam_elements_buffer,
                ABS_TOLERANCE ) );

        ::NS(ClContext_delete)( context );
        context = nullptr;
    }

    if( num_available_nodes == std::size_t{ 0 } )
    {
        std::cout << "Skipping unit-test because no "
                  << "OpenCL platforms have been found --> "
                  << "NEITHER PASSED NOR FAILED!"
                  << std::endl;
    }

    ::NS(Buffer_delete)( lhc_particles_buffer );
    ::NS(Buffer_delete)( lhc_beam_elements_buffer );
}

/* ************************************************************************* */
/* * IMPLEMENTATION OF HELPER FUNCTIONS                                    * */
/* ************************************************************************* */

namespace SIXTRL_CXX_NAMESPACE
{
    namespace tests
    {
        bool ClContext_perform_tracking_tests(
            ::NS(ClContext)* SIXTRL_RESTRICT context,
            ::NS(Buffer) const* SIXTRL_RESTRICT in_particles_buffer,
            ::NS(Buffer) const* SIXTRL_RESTRICT in_beam_elements_buffer,
            double const abs_treshold )
        {
            bool success = false;

            using size_t          = ::NS(buffer_size_t);
            using object_t        = ::NS(Object);
            using particles_t     = ::NS(Particles);
            using index_t         = ::NS(particle_index_t);
            using num_particles_t = ::NS(particle_num_elements_t);

            index_t const in_num_sequences =
                ::NS(Buffer_get_num_of_objects)( in_particles_buffer );

            index_t const in_num_beam_elements =
                ::NS(Buffer_get_num_of_objects)( in_beam_elements_buffer );

            object_t const* be_begin = ::NS(Buffer_get_const_objects_begin)(
                in_beam_elements_buffer );

            object_t const* pb_begin = ::NS(Buffer_get_const_objects_begin)(
                in_particles_buffer );

            object_t const* pb_end = ::NS(Buffer_get_const_objects_end)(
                in_particles_buffer );

            ::NS(Buffer)* pb      = ::NS(Buffer_new)( size_t{ 0xffff } );
            ::NS(Buffer)* eb      = ::NS(Buffer_new)( size_t{ 0xffff } );
            ::NS(Buffer)* diff_pb = ::NS(Buffer_new)( size_t{ 0xffff } );

            if( ( context != nullptr ) &&
                ( in_particles_buffer != nullptr ) &&
                ( in_beam_elements_buffer != nullptr ) &&
                ( ::NS(ClContextBase_has_selected_node)(  context ) ) &&
                ( ::NS(ClContext_has_track_until_kernel)( context ) ) &&
                ( in_num_sequences > index_t{ 0 } ) &&
                ( in_num_beam_elements > index_t{ 0 } ) &&
                ( be_begin != nullptr ) &&
                ( pb_begin != nullptr ) && ( pb_end != nullptr ) &&
                ( pb_begin != pb_end  ) &&
                ( pb != nullptr ) && ( eb != nullptr ) &&
                ( diff_pb != nullptr ) )
            {
                object_t const* pb_it = pb_begin;

                particles_t const* in_particles =
                    ::NS(BufferIndex_get_const_particles)( pb_it );

                object_t const* prev_pb = pb_it++;

                particles_t const* prev_in_particles = nullptr;

                num_particles_t num_particles =
                    ::NS(Particles_get_num_of_particles)( in_particles );

                num_particles_t prev_num_particles = num_particles_t{ 0 };
                size_t cnt = size_t{ 0 };

                success = true;

                for( ; pb_it != pb_end ; ++pb_it, ++prev_pb, ++cnt )
                {
                    prev_in_particles  = in_particles;
                    in_particles = ::NS(BufferIndex_get_const_particles)( pb_it );

                    prev_num_particles = num_particles;
                    num_particles = ::NS(Particles_get_num_of_particles)(
                        in_particles );

                    success &= ( num_particles == prev_num_particles );
                    success &= ( in_particles != nullptr );

                    if( !success ) break;

                    /* ------------------------------------------------- */
                    /* build OpenCL argument from particles buffer: */

                    ::NS(Buffer_reset)( pb );

                    particles_t* particles =
                        ::NS(Particles_add_copy)( pb, prev_in_particles );

                    success  = ( ::NS(Buffer_get_num_of_objects)( pb ) == 1u );
                    success &= ( ::NS(Buffer_get_size)( pb ) > size_t{ 0 } );

                    ::NS(ClArgument)* particles_arg =
                        ::NS(ClArgument_new_from_buffer)( pb, context );

                    success &= ( particles_arg != nullptr );
                    success &= ( ::NS(ClArgument_uses_cobj_buffer)( particles_arg ) );

                    success &= ( ::NS(ClArgument_get_ptr_cobj_buffer)(
                        particles_arg ) == pb );

                    success &= ( ::NS(ClArgument_get_argument_size)( particles_arg )
                        == ::NS(Buffer_get_size)( pb ) );

                    success &= ( ::NS(ClArgument_get_ptr_to_context)(
                        particles_arg ) == context );

                    if( !success ) break;

                    /* ------------------------------------------------- */
                    /* build Ocl arg from line in beam-elements buffer: */

                    index_t const begin_elem_id =
                        ::NS(Particles_get_at_element_id_value)(
                            particles, num_particles_t{ 0 } );

                    index_t const end_elem_id =
                        ::NS(Particles_get_at_element_id_value)(
                            in_particles, num_particles_t{ 0 } );

                    index_t const at_turn = ::NS(Particles_get_at_turn_value)(
                        particles, num_particles_t{ 0 } );

                    SIXTRL_ASSERT( ::NS(Particles_get_at_turn_value)(
                        in_particles, num_particles_t{ 0 } ) );

                    object_t const* line_begin = be_begin;
                    object_t const* line_end   = be_begin;

                    std::advance( line_begin, begin_elem_id + index_t{ 1 } );
                    std::advance( line_end,   end_elem_id   + index_t{ 1 } );

                    ::NS(Buffer_reset)( eb );
                    ::NS(BeamElements_copy_to_buffer)( eb, line_begin, line_end );

                    success &= ( static_cast< std::ptrdiff_t >(
                        ::NS(Buffer_get_num_of_objects)( eb ) ) ==
                        std::distance( line_begin, line_end ) );

                    success &= ( ::NS(Buffer_get_size)( eb ) > size_t{ 0 } );

                    ::NS(ClArgument)* beam_elements_arg =
                        ::NS(ClArgument_new_from_buffer)( eb, context );

                    success &= ( beam_elements_arg != nullptr );
                    success &= ( ::NS(ClArgument_uses_cobj_buffer)(
                        beam_elements_arg ) );

                    success &= ( ::NS(ClArgument_get_ptr_cobj_buffer)(
                        beam_elements_arg ) == eb );

                    success &= ( ::NS(ClArgument_get_argument_size)(
                        beam_elements_arg ) == ::NS(Buffer_get_size)( eb ) );

                    success &= ( ::NS(ClArgument_get_ptr_to_context)(
                        beam_elements_arg ) == context );

                    if( !success ) break;

                    /* ------------------------------------------------- */
                    /* Perform tracking of particles over line: */

                    success &= ( ::NS(TRACK_SUCCESS) ==
                        ::NS(ClContext_track_until)( context, at_turn + 1 ) );

                    if( !success )  break;

                    /* ------------------------------------------------- */
                    /* Read back particles and compare  values: */

                    success &= ( ::NS(ClArgument_read)( particles_arg, pb ) );

                    if( !success )  break;

                    success &= ( 0 == ::NS(Buffer_remap)( pb ) );

                    particles_t const* cmp_particles = in_particles;
                    particles = ::NS(Particles_buffer_get_particles)( pb, 0u );

                    success &= ( particles != nullptr );
                    if( !success ) break;

                    if( 0 != ::NS(Particles_compare_real_values_with_treshold)(
                            cmp_particles, particles, abs_treshold ) )
                    {
                        ::NS(Buffer_reset)( diff_pb );

                        particles_t* diff_particles = ::NS(Particles_new)(
                            diff_pb, num_particles );

                        success &= ( diff_particles != nullptr );

                        ::NS(Particles_calculate_difference)(
                            cmp_particles, particles, diff_particles );

                        std::cout << "Diff. between tracked particles and "
                                     "reference particle data detected: \r\n"
                                  << "at beam-element block #" << cnt
                                  << ", concerning beam-elements [ "
                                  << std::setw( 6 ) << begin_elem_id + 1
                                  << " - "
                                  << std::setw( 6 ) << end_elem_id   + 1
                                  << " ):\r\n"
                                  << "absolute tolerance : "
                                  << abs_treshold << "\r\n"
                                  << "\r\n"
                                  << "beam-elements: \r\n";

                        object_t const* line_it  = line_begin;
                        size_t jj = begin_elem_id + index_t{ 1 };

                        for( ; line_it != line_end ; ++line_it )
                        {
                            std::cout << "be id = " << std::setw( 6 ) << jj ;
                            ::NS(BeamElement_print_out)( line_it );
                        }

                        std::cout << "\r\n"
                                  << "diff_particles = "
                                  << "|cmp_particles - particles| :\r\n";

                        ::NS(Particles_print_out)( diff_particles );

                        std::cout << std::endl;
                    }

                    success &= ( ::NS(Particles_compare_real_values_with_treshold)(
                            cmp_particles, particles, abs_treshold ) == 0 );

                    ::NS(ClArgument_delete)( particles_arg );
                    ::NS(ClArgument_delete)( beam_elements_arg );

                    if( !success ) break;
                }
            }

            ::NS(Buffer_delete)( pb );
            ::NS(Buffer_delete)( eb );
            ::NS(Buffer_delete)( diff_pb );

            return success;
        }
    }
}

/* end: tests/sixtracklib/opencl/test_track_opencl_c99.cpp */
