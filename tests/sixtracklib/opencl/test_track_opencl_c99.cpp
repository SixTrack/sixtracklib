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
            ::st_ClContext* SIXTRL_RESTRICT context,
            ::st_Buffer const* SIXTRL_RESTRICT in_particles_buffer,
            ::st_Buffer const* SIXTRL_RESTRICT in_beam_elements_buffer,
            double const abs_treshold = double{ 1e-16 } );
    }
}

/* ************************************************************************* */

TEST( C99_OpenCL_TrackParticlesTests, LHCReproduceSixTrackSingleTurnNoBeamBeam )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    static double const ABS_TOLERANCE = double{ 1e-13 };

    ::st_Buffer* lhc_particles_buffer = ::st_Buffer_new_from_file(
        ::st_PATH_TO_TEST_LHC_PARTICLES_DATA_T1_P2_NO_BEAM_BEAM );

    ::st_Buffer* lhc_beam_elements_buffer = ::st_Buffer_new_from_file(
        ::st_PATH_TO_TEST_LHC_BEAM_ELEMENTS_DATA_NO_BEAM_BEAM );

    ::st_ClContext* context = ::st_ClContext_create();

    ASSERT_TRUE( context != nullptr );

    std::size_t const num_available_nodes =
        ::st_ClContextBase_get_num_available_nodes( context );

    for( std::size_t nn = std::size_t{ 0 } ; nn < num_available_nodes ; ++nn )
    {
        ASSERT_TRUE( ::st_ClContextBase_select_node_by_index( context, nn ) );
        ASSERT_TRUE( ::st_ClContextBase_has_selected_node( context ) );

        /* ----------------------------------------------------------------- */
        /* Add and select optimized tracking kernel: */

        std::string path_to_tracking_program( ::st_PATH_TO_BASE_DIR );
        path_to_tracking_program += "sixtracklib/opencl/kernels/";
        path_to_tracking_program += "track_particles_kernel.cl";

        std::string tracking_program_compile_options( "-D_GPUCODE=1" );
        tracking_program_compile_options += " -cl-strict-aliasing";
        tracking_program_compile_options += " -DSIXTRL_BUFFER_ARGPTR_DEC=__private";
        tracking_program_compile_options += " -DSIXTRL_BUFFER_DATAPTR_DEC=__global";
        tracking_program_compile_options += " -I";
        tracking_program_compile_options += ::st_PATH_TO_BASE_DIR;

        int const tracking_program_id = ::st_ClContextBase_add_program_file(
            context, path_to_tracking_program.c_str(),
            tracking_program_compile_options.c_str() );

        ASSERT_TRUE( tracking_program_id >= 0 );
        ASSERT_TRUE( static_cast< std::size_t >( tracking_program_id ) <
                     ::st_ClContextBase_get_num_available_programs( context ) );

        std::string tracking_kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
        tracking_kernel_name += "Track_particles_beam_elements_opencl";

        int const tracking_kernel_id = ::st_ClContextBase_enable_kernel(
            context, tracking_kernel_name.c_str(), tracking_program_id );

        ASSERT_TRUE( tracking_kernel_id >= 0 );
        ASSERT_TRUE( static_cast< std::size_t >( tracking_kernel_id ) <
                     ::st_ClContextBase_get_num_available_kernels( context ) );

        ASSERT_TRUE( tracking_kernel_name.compare(
            ::st_ClContextBase_get_kernel_function_name(
                context, tracking_kernel_id ) ) == 0 );

        ASSERT_TRUE( ::st_ClContext_set_tracking_kernel_id(
            context, tracking_kernel_id ) );

        ASSERT_TRUE( ::st_ClContext_has_tracking_kernel( context ) );
        ASSERT_TRUE( ::st_ClContext_get_tracking_kernel_id( context ) ==
                     tracking_kernel_id );

        ::st_context_node_info_t const* node_info =
            ::st_ClContextBase_get_selected_node_info( context );

        ASSERT_TRUE( node_info != nullptr );
        ASSERT_TRUE( ::st_ClContext_has_tracking_kernel( context ) );
        ASSERT_TRUE( ::st_ClContextBase_has_remapping_kernel( context ) );

        char id_str[ 32 ];
        ::st_ClContextBase_get_selected_node_id_str( context, id_str, 32 );

        std::cout << "# ------------------------------------------------------"
                  << "--------------------------------------------------------"
                  << "\r\n"
                  << "# Run Test on :: \r\n"
                  << "# ID          :: " << id_str << "\r\n"
                  << "# NAME        :: "
                  << ::st_ComputeNodeInfo_get_name( node_info ) << "\r\n"
                  << "# PLATFORM    :: "
                  << ::st_ComputeNodeInfo_get_platform( node_info ) << "\r\n"
                  << "# "
                  << std::endl;

        ASSERT_TRUE( st::tests::ClContext_perform_tracking_tests(
            context, lhc_particles_buffer, lhc_beam_elements_buffer,
                ABS_TOLERANCE ) );

        ::st_ClContext_clear( context );
    }

    if( num_available_nodes == std::size_t{ 0 } )
    {
        std::cout << "Skipping unit-test because no "
                  << "OpenCL platforms have been found --> "
                  << "NEITHER PASSED NOR FAILED!"
                  << std::endl;
    }

    ::st_Buffer_delete( lhc_particles_buffer );
    ::st_Buffer_delete( lhc_beam_elements_buffer );
    ::st_ClContext_delete( context );
}

/* ************************************************************************* */

TEST( C99_OpenCL_TrackParticlesTests,
      LHCReproduceSixTrackSingleTurnNoBeamBeamPrivParticlesOptimized )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    static double const ABS_TOLERANCE = double{ 1e-13 };

    ::st_Buffer* lhc_particles_buffer = ::st_Buffer_new_from_file(
        ::st_PATH_TO_TEST_LHC_PARTICLES_DATA_T1_P2_NO_BEAM_BEAM );

    ::st_Buffer* lhc_beam_elements_buffer = ::st_Buffer_new_from_file(
        ::st_PATH_TO_TEST_LHC_BEAM_ELEMENTS_DATA_NO_BEAM_BEAM );

    ::st_ClContext* context = ::st_ClContext_create();

    ASSERT_TRUE( context != nullptr );

    std::size_t const num_available_nodes =
        ::st_ClContextBase_get_num_available_nodes( context );

    for( std::size_t nn = std::size_t{ 0 } ; nn < num_available_nodes ; ++nn )
    {
        ASSERT_TRUE( ::st_ClContextBase_select_node_by_index( context, nn ) );
        ASSERT_TRUE( ::st_ClContextBase_has_selected_node( context ) );

        /* ----------------------------------------------------------------- */
        /* Add and select optimized tracking kernel: */

        std::string path_to_tracking_program( ::st_PATH_TO_BASE_DIR );
        path_to_tracking_program += "sixtracklib/opencl/kernels/";
        path_to_tracking_program += "track_particles_priv_particles_optimized_kernel.cl";

        std::string tracking_program_compile_options( "-D_GPUCODE=1" );
        tracking_program_compile_options += " -cl-strict-aliasing";
        tracking_program_compile_options += " -DSIXTRL_BUFFER_ARGPTR_DEC=__private";
        tracking_program_compile_options += " -DSIXTRL_BUFFER_DATAPTR_DEC=__global";
        tracking_program_compile_options += " -DSIXTRL_PARTICLE_ARGPTR_DEC=__private";
        tracking_program_compile_options += " -DSIXTRL_PARTICLE_DATAPTR_DEC=__private";
        tracking_program_compile_options += " -I";
        tracking_program_compile_options += ::st_PATH_TO_BASE_DIR;

        int const tracking_program_id = ::st_ClContextBase_add_program_file(
            context, path_to_tracking_program.c_str(),
            tracking_program_compile_options.c_str() );

        ASSERT_TRUE( tracking_program_id >= 0 );
        ASSERT_TRUE( static_cast< std::size_t >( tracking_program_id ) <
                     ::st_ClContextBase_get_num_available_programs( context ) );

        std::string tracking_kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
        tracking_kernel_name +=
            "Track_particles_beam_elements_priv_particles_optimized_opencl";

        int const tracking_kernel_id = ::st_ClContextBase_enable_kernel(
            context, tracking_kernel_name.c_str(), tracking_program_id );

        ASSERT_TRUE( tracking_kernel_id >= 0 );
        ASSERT_TRUE( static_cast< std::size_t >( tracking_kernel_id ) <
                     ::st_ClContextBase_get_num_available_kernels( context ) );

        ASSERT_TRUE( ::st_ClContext_set_tracking_kernel_id(
            context, tracking_kernel_id ) );

        ASSERT_TRUE( ::st_ClContext_has_tracking_kernel( context ) );
        ASSERT_TRUE( ::st_ClContext_get_tracking_kernel_id( context ) ==
                     tracking_kernel_id );

        /* ----------------------------------------------------------------- */

        ::st_context_node_info_t const* node_info =
            ::st_ClContextBase_get_selected_node_info( context );

        ASSERT_TRUE( node_info != nullptr );
        ASSERT_TRUE( ::st_ClContext_has_tracking_kernel( context ) );
        ASSERT_TRUE( ::st_ClContextBase_has_remapping_kernel( context ) );

        char id_str[ 32 ];
        ::st_ClContextBase_get_selected_node_id_str( context, id_str, 32 );

        std::cout << "# ------------------------------------------------------"
                  << "--------------------------------------------------------"
                  << "\r\n"
                  << "# Run Test on :: \r\n"
                  << "# ID          :: " << id_str << "\r\n"
                  << "# NAME        :: "
                  << ::st_ComputeNodeInfo_get_name( node_info ) << "\r\n"
                  << "# PLATFORM    :: "
                  << ::st_ComputeNodeInfo_get_platform( node_info ) << "\r\n"
                  << "# "
                  << std::endl;

        ASSERT_TRUE( st::tests::ClContext_perform_tracking_tests(
            context, lhc_particles_buffer, lhc_beam_elements_buffer,
                ABS_TOLERANCE ) );

        ::st_ClContext_clear( context );
    }

    if( num_available_nodes == std::size_t{ 0 } )
    {
        std::cout << "Skipping unit-test because no "
                  << "OpenCL platforms have been found --> "
                  << "NEITHER PASSED NOR FAILED!"
                  << std::endl;
    }

    ::st_Buffer_delete( lhc_particles_buffer );
    ::st_Buffer_delete( lhc_beam_elements_buffer );
    ::st_ClContext_delete( context );
}

/* ************************************************************************* */
/* * IMPLEMENTATION OF HELPER FUNCTIONS                                    * */
/* ************************************************************************* */

namespace SIXTRL_CXX_NAMESPACE
{
    namespace tests
    {
        bool ClContext_perform_tracking_tests(
            ::st_ClContext* SIXTRL_RESTRICT context,
            ::st_Buffer const* SIXTRL_RESTRICT in_particles_buffer,
            ::st_Buffer const* SIXTRL_RESTRICT in_beam_elements_buffer,
            double const abs_treshold )
        {
            bool success = false;

            using size_t          = ::st_buffer_size_t;
            using object_t        = ::st_Object;
            using particles_t     = ::st_Particles;
            using index_t         = ::st_particle_index_t;
            using num_particles_t = ::st_particle_num_elements_t;

            index_t const in_num_sequences =
                ::st_Buffer_get_num_of_objects( in_particles_buffer );

            index_t const in_num_beam_elements =
                ::st_Buffer_get_num_of_objects( in_beam_elements_buffer );

            object_t const* be_begin = ::st_Buffer_get_const_objects_begin(
                in_beam_elements_buffer );

            object_t const* pb_begin = ::st_Buffer_get_const_objects_begin(
                in_particles_buffer );

            object_t const* pb_end = ::st_Buffer_get_const_objects_end(
                in_particles_buffer );

            ::st_Buffer* pb      = ::st_Buffer_new( size_t{ 0xffff } );
            ::st_Buffer* eb      = ::st_Buffer_new( size_t{ 0xffff } );
            ::st_Buffer* diff_pb = ::st_Buffer_new( size_t{ 0xffff } );

            if( ( context != nullptr ) &&
                ( in_particles_buffer != nullptr ) &&
                ( in_beam_elements_buffer != nullptr ) &&
                ( ::st_ClContextBase_has_selected_node(  context ) ) &&
                ( ::st_ClContext_has_tracking_kernel( context ) ) &&
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
                    ::st_BufferIndex_get_const_particles( pb_it );

                object_t const* prev_pb = pb_it++;

                particles_t const* prev_in_particles = nullptr;

                num_particles_t num_particles =
                    ::st_Particles_get_num_of_particles( in_particles );

                num_particles_t prev_num_particles = num_particles_t{ 0 };
                size_t cnt = size_t{ 0 };

                success = true;

                for( ; pb_it != pb_end ; ++pb_it, ++prev_pb, ++cnt )
                {
                    prev_in_particles  = in_particles;
                    in_particles = ::st_BufferIndex_get_const_particles( pb_it );

                    prev_num_particles = num_particles;
                    num_particles = ::st_Particles_get_num_of_particles(
                        in_particles );

                    success &= ( num_particles == prev_num_particles );
                    success &= ( in_particles != nullptr );

                    if( !success ) break;

                    /* ------------------------------------------------- */
                    /* build OpenCL argument from particles buffer: */

                    ::st_Buffer_reset( pb );

                    particles_t* particles =
                        ::st_Particles_add_copy( pb, prev_in_particles );

                    success  = ( ::st_Buffer_get_num_of_objects( pb ) == 1u );
                    success &= ( ::st_Buffer_get_size( pb ) > size_t{ 0 } );

                    ::st_ClArgument* particles_arg =
                        ::st_ClArgument_new_from_buffer( pb, context );

                    success &= ( particles_arg != nullptr );
                    success &= ( ::st_ClArgument_uses_cobj_buffer( particles_arg ) );

                    success &= ( ::st_ClArgument_get_ptr_cobj_buffer(
                        particles_arg ) == pb );

                    success &= ( ::st_ClArgument_get_argument_size( particles_arg )
                        == ::st_Buffer_get_size( pb ) );

                    success &= ( ::st_ClArgument_get_ptr_to_context(
                        particles_arg ) == context );

                    if( !success ) break;

                    /* ------------------------------------------------- */
                    /* build Ocl arg from line in beam-elements buffer: */

                    index_t const begin_elem_id =
                        ::st_Particles_get_at_element_id_value(
                            particles, num_particles_t{ 0 } );

                    index_t const end_elem_id =
                        ::st_Particles_get_at_element_id_value(
                            in_particles, num_particles_t{ 0 } );

                    object_t const* line_begin = be_begin;
                    object_t const* line_end   = be_begin;

                    std::advance( line_begin, begin_elem_id + index_t{ 1 } );
                    std::advance( line_end,   end_elem_id   + index_t{ 1 } );

                    ::st_Buffer_reset( eb );
                    ::st_BeamElements_copy_to_buffer( eb, line_begin, line_end );

                    success &= ( static_cast< std::ptrdiff_t >(
                        ::st_Buffer_get_num_of_objects( eb ) ) ==
                        std::distance( line_begin, line_end ) );

                    success &= ( ::st_Buffer_get_size( eb ) > size_t{ 0 } );

                    ::st_ClArgument* beam_elements_arg =
                        ::st_ClArgument_new_from_buffer( eb, context );

                    success &= ( beam_elements_arg != nullptr );
                    success &= ( ::st_ClArgument_uses_cobj_buffer(
                        beam_elements_arg ) );

                    success &= ( ::st_ClArgument_get_ptr_cobj_buffer(
                        beam_elements_arg ) == eb );

                    success &= ( ::st_ClArgument_get_argument_size(
                        beam_elements_arg ) == ::st_Buffer_get_size( eb ) );

                    success &= ( ::st_ClArgument_get_ptr_to_context(
                        beam_elements_arg ) == context );

                    if( !success ) break;

                    /* ------------------------------------------------- */
                    /* Perform tracking of particles over line: */

                    success &= ( 0 == ::st_ClContext_track(
                        context, particles_arg, beam_elements_arg ) );

                    if( !success )  break;

                    /* ------------------------------------------------- */
                    /* Read back particles and compare  values: */

                    success &= ( ::st_ClArgument_read( particles_arg, pb ) );

                    if( !success )  break;

                    success &= ( 0 == ::st_Buffer_remap( pb ) );

                    particles_t const* cmp_particles = in_particles;
                    particles = ::st_Particles_buffer_get_particles( pb, 0u );

                    success &= ( particles != nullptr );
                    if( !success ) break;

                    if( 0 != ::st_Particles_compare_real_values_with_treshold(
                            cmp_particles, particles, abs_treshold ) )
                    {
                        ::st_Buffer_reset( diff_pb );

                        particles_t* diff_particles = ::st_Particles_new(
                            diff_pb, num_particles );

                        success &= ( diff_particles != nullptr );

                        ::st_Particles_calculate_difference(
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
                            ::st_BeamElement_print( line_it );
                        }

                        std::cout << "\r\n"
                                  << "diff_particles = "
                                  << "|cmp_particles - particles| :\r\n";

                        ::st_Particles_print_out( diff_particles );

                        std::cout << std::endl;
                    }

                    success &= ( ::st_Particles_compare_real_values_with_treshold(
                            cmp_particles, particles, abs_treshold ) == 0 );

                    ::st_ClArgument_delete( particles_arg );
                    ::st_ClArgument_delete( beam_elements_arg );

                    if( !success ) break;
                }
            }

            ::st_Buffer_delete( pb );
            ::st_Buffer_delete( eb );
            ::st_Buffer_delete( diff_pb );

            return success;
        }
    }
}

/* end: tests/sixtracklib/opencl/test_track_opencl_c99.cpp */
