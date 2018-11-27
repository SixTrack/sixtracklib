#define _USE_MATH_DEFINES

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/be_drift/be_drift.h"

#include "sixtracklib/opencl/internal/base_context.h"
#include "sixtracklib/opencl/argument.h"

TEST( C99_OpenCL_ParticlesTests, CopyParticlesHostToDeviceThenBackCompare )
{
    using size_t = ::st_buffer_size_t;

    ::st_Buffer* in_buffer = ::st_Buffer_new( 0u );

    size_t const NUM_PARTICLE_BLOCKS = size_t{ 16 };
    size_t const PARTICLES_PER_BLOCK = size_t{ 1000 };
    size_t const TOTAL_NUM_PARTICLES = NUM_PARTICLE_BLOCKS * PARTICLES_PER_BLOCK;

    size_t jj = size_t{ 0 };

    for( size_t ii = size_t{ 0 } ; ii < NUM_PARTICLE_BLOCKS ; ++ii )
    {
        ::st_Particles* particles =
            ::st_Particles_new( in_buffer, PARTICLES_PER_BLOCK );

        ASSERT_TRUE( particles != nullptr );
        ASSERT_TRUE( ::st_Particles_get_num_of_particles( particles ) ==
                         PARTICLES_PER_BLOCK );

        ::st_Particles_realistic_init( particles );

        for( size_t kk = size_t{ 0 } ; kk < PARTICLES_PER_BLOCK ; ++kk, ++jj )
        {
            ::st_Particles_set_particle_id_value( particles, kk, jj );
        }
    }

    ASSERT_TRUE( ::st_Buffer_is_particles_buffer( in_buffer ) );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( in_buffer  ) ==
                 NUM_PARTICLE_BLOCKS );

    ASSERT_TRUE( ::st_Particles_buffer_get_total_num_of_particles( in_buffer )
                 == TOTAL_NUM_PARTICLES );

    /* ---------------------------------------------------------------------- */
    /* Prepare path to test GPU kernel program file and compile options */

    std::ostringstream a2str;

    a2str <<  ::st_PATH_TO_BASE_DIR
          << "tests/sixtracklib/testlib/opencl/kernels/"
          << "opencl_particles_kernel.cl";

    std::string const path_to_program = a2str.str();
    a2str.str( "" );

    a2str << " -D_GPUCODE=1"
          << " -DSIXTRL_BUFFER_ARGPTR_DEC=__private"
          << " -DSIXTRL_BUFFER_DATAPTR_DEC=__global"
          << " -I" << ::st_PATH_TO_SIXTRL_INCLUDE_DIR;

    if( std::strcmp( ::st_PATH_TO_SIXTRL_INCLUDE_DIR,
                     ::st_PATH_TO_SIXTRL_TESTLIB_INCLUDE_DIR ) != 0 )
    {
        a2str << " -I" << ::st_PATH_TO_SIXTRL_TESTLIB_INCLUDE_DIR;
    }

    std::string const program_compile_options( a2str.str() );
    a2str.str( "" );

    a2str << SIXTRL_C99_NAMESPACE_PREFIX_STR
          << "Particles_copy_buffer_opencl";

    std::string const kernel_name( a2str.str() );
    a2str.str( "" );

    /* ---------------------------------------------------------------------- */
    /* Get number of available devices */


    ::st_ClContextBase* context = ::st_ClContextBase_create();

    ASSERT_TRUE( context != nullptr );

    size_t const num_available_nodes =
        ::st_ClContextBase_get_num_available_nodes( context );

    ::st_ClContextBase_delete( context );
    context = nullptr;

    for( size_t ii = size_t{ 0 } ; ii < num_available_nodes ; ++ii )
    {
        ::st_Buffer* out_buffer = ::st_Buffer_new( 0u );

        for( jj = size_t{ 0 } ; jj < NUM_PARTICLE_BLOCKS ; ++jj )
        {
            ::st_Particles* particles =
                ::st_Particles_new( out_buffer, PARTICLES_PER_BLOCK );

            ASSERT_TRUE( particles != nullptr );
            ASSERT_TRUE( ::st_Particles_get_num_of_particles( particles ) ==
                         PARTICLES_PER_BLOCK );
        }

        ASSERT_TRUE( ::st_Buffer_is_particles_buffer( out_buffer ) );
        ASSERT_TRUE( ::st_Buffer_get_num_of_objects( out_buffer  ) ==
                     NUM_PARTICLE_BLOCKS );

        ASSERT_TRUE( ::st_Particles_buffer_get_total_num_of_particles( out_buffer )
                     == TOTAL_NUM_PARTICLES );

        context = ::st_ClContextBase_create();
        ::st_ClContextBase_enable_debug_mode( context );

        ASSERT_TRUE(  ::st_ClContextBase_is_debug_mode_enabled( context ) );
        ASSERT_TRUE(  ::st_ClContextBase_select_node_by_index( context, ii ) );
        ASSERT_TRUE(  ::st_ClContextBase_has_selected_node( context ) );

        ::st_context_node_info_t const* node_info =
            ::st_ClContextBase_get_selected_node_info( context );

        ASSERT_TRUE( node_info != nullptr );
        ASSERT_TRUE( ::st_ClContextBase_has_remapping_kernel( context ) );

        char id_str[ 32 ];
        ::st_ClContextBase_get_selected_node_id_str( context, id_str, 32 );

        int program_id = ::st_ClContextBase_add_program_file(
            context, path_to_program.c_str(), program_compile_options.c_str() );

        ASSERT_TRUE( program_id >= 0 );

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

        ASSERT_TRUE( ::st_ClContextBase_is_program_compiled(
            context, program_id ) );

        int const kernel_id = ::st_ClContextBase_enable_kernel(
            context, kernel_name.c_str(), program_id );

        ASSERT_TRUE( kernel_id >= 0 );

        ::st_ClArgument* in_buffer_arg = ::st_ClArgument_new_from_buffer(
            in_buffer, context );

        ASSERT_TRUE( ( in_buffer_arg != nullptr ) &&
            ( ::st_ClArgument_get_argument_size( in_buffer_arg ) ==
              ::st_Buffer_get_size( in_buffer ) ) &&
            ( ::st_ClArgument_uses_cobj_buffer( in_buffer_arg ) ) &&
            ( ::st_ClArgument_get_ptr_cobj_buffer( in_buffer_arg ) == in_buffer ) );

        ::st_ClArgument* out_buffer_arg =
            ::st_ClArgument_new_from_buffer( out_buffer, context );

        ASSERT_TRUE( ( out_buffer_arg != nullptr ) &&
            ( ::st_ClArgument_get_argument_size( out_buffer_arg ) ==
              ::st_Buffer_get_size( out_buffer ) ) &&
            ( ::st_ClArgument_uses_cobj_buffer( out_buffer_arg ) ) &&
            ( ::st_ClArgument_get_ptr_cobj_buffer( out_buffer_arg ) == out_buffer ) );

        int32_t success_flag = int32_t{ 0 };

        ::st_ClArgument* success_flag_arg = ::st_ClArgument_new_from_memory(
            &success_flag, sizeof( success_flag ), context );

        ASSERT_TRUE( ( success_flag_arg != nullptr ) &&
            ( ::st_ClArgument_get_argument_size( success_flag_arg ) ==
                sizeof( success_flag ) ) &&
            ( !::st_ClArgument_uses_cobj_buffer( success_flag_arg ) ) );

        ::st_ClContextBase_assign_kernel_argument(
            context, kernel_id, 0u, in_buffer_arg );

        ::st_ClContextBase_assign_kernel_argument(
            context, kernel_id, 1u, out_buffer_arg );

        ::st_ClContextBase_assign_kernel_argument(
            context, kernel_id, 2u, success_flag_arg );

        success_flag = int32_t{ -1 };

        ASSERT_TRUE( ::st_ClContextBase_run_kernel(
            context, kernel_id, TOTAL_NUM_PARTICLES ) );

        ASSERT_TRUE( ::st_ClArgument_read( out_buffer_arg, out_buffer ) );
        ASSERT_TRUE( ::st_ClArgument_read_memory( success_flag_arg,
                &success_flag, sizeof( success_flag ) ) );

        ASSERT_TRUE( success_flag == 0 );
        ASSERT_TRUE( ::st_Buffer_get_num_of_objects( out_buffer ) ==
                     NUM_PARTICLE_BLOCKS );

        for( jj = size_t{ 0 } ; jj < NUM_PARTICLE_BLOCKS ; ++jj )
        {
            ::st_Particles const* in_particles =
                ::st_Particles_buffer_get_const_particles( in_buffer, jj );

            ::st_Particles const* out_particles =
                ::st_Particles_buffer_get_const_particles( out_buffer, jj );

            ASSERT_TRUE( in_particles != nullptr );
            ASSERT_TRUE( out_particles != nullptr );
            ASSERT_TRUE( in_particles  != out_particles );
            ASSERT_TRUE( ::st_Particles_have_same_structure(
                in_particles, out_particles ) );

            ASSERT_TRUE( !::st_Particles_map_to_same_memory(
                in_particles, out_particles ) );

            ASSERT_TRUE( 0 == ::st_Particles_compare_values(
                in_particles, out_particles ) );
        }

        ::st_ClArgument_delete( in_buffer_arg );
        ::st_ClArgument_delete( out_buffer_arg );
        ::st_ClArgument_delete( success_flag_arg );
        ::st_Buffer_delete( out_buffer );

        ::st_ClContextBase_delete( context );
        context = nullptr;
    }

    ::st_Buffer_delete( in_buffer );
}

/* end: tests/sixtracklib/opencl/test_particles_opencl_c99.cpp */
