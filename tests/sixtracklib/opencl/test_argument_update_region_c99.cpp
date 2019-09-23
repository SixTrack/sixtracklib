#include "sixtracklib/opencl/argument.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <utility>

#include <gtest/gtest.h>

#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/be_cavity/be_cavity.h"
#include "sixtracklib/opencl/context.h"
#include "sixtracklib/testlib.h"

TEST( C99_OpenCL_ClArgumentUpdateRegionTest, BasicUsage )
{
    using buffer_t     = ::NS(Buffer);
    using argument_t   = ::NS(ClArgument);
    using controller_t = ::NS(ClContext);
    using cavity_t     = ::NS(Cavity);
    using status_t     = ::NS(arch_status_t);
    using size_t       = ::NS(ctrl_size_t);

    /* Load lattice from binary dump */
    buffer_t* lattice = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    SIXTRL_ASSERT( lattice != nullptr );

    /* Prepare list for cavities */
    std::vector< size_t >   cavity_indices;
    std::vector< size_t >   cavity_offsets;
    std::vector< size_t >   cavity_lengths;
    std::vector< cavity_t > cavities;

    /* Find all offsets, lengths and initial cavity values */

    size_t const num_beam_elements =
        ::NS(Buffer_get_num_of_objects)( lattice );

    for( size_t ii = size_t{ 0 } ; ii < num_beam_elements ; ++ii )
    {
        cavity_t const* ptr_cavity =
            ::NS(BeamElements_buffer_get_const_cavity)( lattice, ii );

        if( ptr_cavity != nullptr )
        {
            cavity_indices.push_back( ii );
            cavity_lengths.push_back( sizeof( cavity_t ) );
            cavity_offsets.push_back( reinterpret_cast< uintptr_t >(
                ptr_cavity ) - ::NS(Buffer_get_data_begin_addr)( lattice ) );

            cavities.push_back( *ptr_cavity );
        }
    }

    size_t const num_cavities = cavities.size();
    ASSERT_TRUE( num_cavities > size_t{ 0 } );

    /* Initialize an OpenCL controller */

    controller_t* controller = ::NS(ClContext_create)();
    SIXTRL_ASSERT( controller != nullptr );

    size_t const num_nodes =
        ::NS(ClContextBase_get_num_available_nodes)( controller );

    if( num_nodes > size_t{ 0 } )
    {
        ::NS(context_node_id_t) default_node_id =
            ::NS(ClContextBase_get_default_node_id)( controller );

        ::NS(ClContextBase_select_node_by_node_id)(
            controller, &default_node_id );

        argument_t* arg =
            ::NS(ClArgument_new_from_buffer)( lattice, controller );

        SIXTRL_ASSERT( arg != nullptr );

        /* Change voltage on host side */

        for( size_t ii = size_t{ 0 } ; ii < num_cavities ; ++ii )
        {
            size_t const idx = cavity_indices[ ii ];
            cavity_t const* ptr_orig_cavity =
                ::NS(BeamElements_buffer_get_const_cavity)( lattice, idx );
            ASSERT_TRUE( ptr_orig_cavity != nullptr );

            cavity_t* ptr_cavity = &cavities[ ii ];
            ASSERT_TRUE( ptr_cavity != nullptr );
            ASSERT_TRUE( 0 == ::NS(Cavity_compare_values)(
                ptr_orig_cavity, ptr_cavity ) );

            ptr_cavity->voltage *= double{ 1.05 };

            if( ptr_cavity->voltage < double{ 1e-16 } )
                    ptr_cavity->voltage += 1e3;

            ASSERT_TRUE( 0 != ::NS(Cavity_compare_values)(
                ptr_orig_cavity, ptr_cavity ) );

            status_t status = NS(ClArgument_update_region)( arg,
                cavity_offsets[ ii ], cavity_lengths[ ii ], ptr_cavity );

            ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
        }

        /* Read back the values from the device to lattice */
        ::NS(ClArgument_read)( arg, lattice );

        /* Compare the values with the expected state */
        for( size_t ii = size_t{ 0 } ; ii < num_cavities ; ++ii )
        {
            size_t const idx = cavity_indices[ ii ];
            cavity_t const* ptr_orig_cavity =
                NS(BeamElements_buffer_get_const_cavity)( lattice, idx );
            ASSERT_TRUE( ptr_orig_cavity != nullptr );

            cavity_t* ptr_cavity = &cavities[ ii ];
            ASSERT_TRUE( ptr_cavity != nullptr );
            ASSERT_TRUE( 0 == ::NS(Cavity_compare_values)(
                ptr_orig_cavity, ptr_cavity ) );
        }

        /* Update all cavities again */
        std::vector< void const* > ptr_cavities_begins;

        for( size_t ii = size_t{ 0 } ; ii < num_cavities ; ++ii )
        {
            cavity_t* ptr_cavity = &cavities[ ii ];
            ASSERT_TRUE( ptr_cavity != nullptr );
            ptr_cavity->voltage /= double{ 1.05 };
            ptr_cavities_begins.push_back( ptr_cavity );
        }

        status_t status = ::NS(ClArgument_update_regions)( arg, num_cavities,
            cavity_offsets.data(), cavity_lengths.data(),
                ptr_cavities_begins.data() );

        ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

        /* Read back the modified lattice */
        ::NS(ClArgument_read)( arg, lattice );

        /* Compare the values with the expected state */
        for( size_t ii = size_t{ 0 } ; ii < num_cavities ; ++ii )
        {
            size_t const idx = cavity_indices[ ii ];
            cavity_t const* ptr_orig_cavity =
                ::NS(BeamElements_buffer_get_const_cavity)( lattice, idx );
            ASSERT_TRUE( ptr_orig_cavity != nullptr );

            cavity_t* ptr_cavity = &cavities[ ii ];
            ASSERT_TRUE( ptr_cavity != nullptr );
            ASSERT_TRUE( 0 == ::NS(Cavity_compare_values)(
                ptr_orig_cavity, ptr_cavity ) );
        }

        ::NS(ClArgument_delete)( arg );
        arg = nullptr;
    }

    /* Cleanup */

    NS(ClContext_delete)( controller );
    NS(Buffer_delete)( lattice );

    controller = nullptr;
    lattice = nullptr;
}

/* end: tests/sixtracklib/opencl/test_argument_update_region_cxx.cpp */
