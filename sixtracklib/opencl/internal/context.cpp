#include "sixtracklib/opencl/context.h"

#if !defined( __CUDACC__ )

#include <chrono>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iterator>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/context/compute_arch.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/opencl/internal/base_context.h"
#include "sixtracklib/opencl/opencl.h"

#if defined( __cplusplus )
#include "sixtracklib/opencl/opencl.hpp"

namespace SIXTRL_CXX_NAMESPACE
{
    namespace
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        using base_ctx_t = st::ClContextBase;
        using ctx_t = st::ClContext;
        using st_size_t    = ctx_t::size_type;
        using st_kernel_id_t  = ctx_t::kernel_id_t;
        using st_program_id_t = ctx_t::program_id_t;
        using st_status_t  = ctx_t::status_t;
    }

    constexpr st_size_t ctx_t::MIN_NUM_TRACK_UNTIL_ARGS;
    constexpr st_size_t ctx_t::MIN_NUM_TRACK_LINE_ARGS;
    constexpr st_size_t ctx_t::MIN_NUM_TRACK_ELEM_ARGS;
    constexpr st_size_t ctx_t::MIN_NUM_ASSIGN_BE_MON_ARGS;
    constexpr st_size_t ctx_t::MIN_NUM_CLEAR_BE_MON_ARGS;
    constexpr st_size_t ctx_t::MIN_NUM_ASSIGN_ELEM_ARGS;
    constexpr st_size_t ctx_t::MIN_NUM_FETCH_PARTICLES_ADDR_ARGS;

    ctx_t::ClContext( char const* SIXTRL_RESTRICT config_str ) :
        base_ctx_t( config_str ),
        m_num_particles_in_pset( st_size_t{ 0 } ), m_pset_index( st_size_t{ 0 } ),
        m_elem_by_elem_config_index(
            ctx_t::DEFAULT_ELEM_BY_ELEM_CONFIG_INDEX ),
        m_track_until_turn_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_line_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_elem_by_elem_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_be_mon_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_fetch_particles_addr_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_clear_be_mon_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_track_line_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_elem_by_elem_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_be_mon_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_fetch_particles_addr_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_clear_be_mon_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_use_optimized_tracking( false )
    {
        ctx_t::status_t const status =
            this->doInitDefaultFeatureFlagsPrivImpl();
        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;

        this->doInitDefaultProgramsPrivImpl();
    }

    ctx_t::ClContext( st_size_t const node_index,
            char const* SIXTRL_RESTRICT config_str ) :
        base_ctx_t( config_str ),
        m_num_particles_in_pset( st_size_t{ 0 } ),
        m_pset_index( st_size_t{ 0 } ),
        m_elem_by_elem_config_index( ctx_t::DEFAULT_ELEM_BY_ELEM_CONFIG_INDEX ),
        m_track_until_turn_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_line_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_elem_by_elem_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_be_mon_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_fetch_particles_addr_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_clear_be_mon_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_track_line_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_elem_by_elem_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_be_mon_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_fetch_particles_addr_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_clear_be_mon_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_use_optimized_tracking( false )
    {
        if( config_str != nullptr )
        {
            this->doSetConfigStr( config_str );
            ClContextBase::doParseConfigString( this->configStr() );
        }

        /* WARNING: Workaround for AMD Heisenbug */
        if( !this->isAvailableNodeAMDPlatform( node_index ) )
        {
            this->m_use_optimized_tracking = true;
        }

        ctx_t::status_t const status =
            this->doInitDefaultFeatureFlagsPrivImpl();
        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;

        this->doInitDefaultProgramsPrivImpl();

        if( ( node_index < this->numAvailableNodes() ) &&
            ( ClContextBase::doSelectNode( node_index ) ) )
        {
            ClContextBase::doInitDefaultKernels();
            this->doInitDefaultKernelsPrivImpl();

            this->doAssignSlotSizeArgPrivImpl(
                st::BUFFER_DEFAULT_SLOT_SIZE );

            this->doAssignElemByElemConfigIndexArgPrivImpl(
                ctx_t::DEFAULT_ELEM_BY_ELEM_CONFIG_INDEX );

            this->doAssignStatusFlagsArgPrivImpl(
                this->internalStatusFlagsBuffer() );
        }
    }

    ctx_t::ClContext( ctx_t::node_id_t const node_id,
                          const char *const SIXTRL_RESTRICT config_str ) :
        base_ctx_t( config_str ),
        m_num_particles_in_pset( st_size_t{ 0 } ),
        m_pset_index( st_size_t{ 0 } ),
        m_elem_by_elem_config_index( ctx_t::DEFAULT_ELEM_BY_ELEM_CONFIG_INDEX ),
        m_track_until_turn_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_line_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_elem_by_elem_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_be_mon_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_fetch_particles_addr_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_clear_be_mon_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_addr_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_track_line_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_elem_by_elem_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_be_mon_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_fetch_particles_addr_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_clear_be_mon_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_addr_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_use_optimized_tracking( false )
    {
        ctx_t::status_t const status =
            this->doInitDefaultFeatureFlagsPrivImpl();
        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;

        this->doInitDefaultProgramsPrivImpl();

        st_size_t  const node_index = this->findAvailableNodesIndex(
            NS(ComputeNodeId_get_platform_id)( &node_id ),
            NS(ComputeNodeId_get_device_id)( &node_id ) );

        /* WARNING: Workaround for AMD Heisenbug */
        if( !this->isAvailableNodeAMDPlatform( node_index ) )
        {
            this->m_use_optimized_tracking = true;
        }

        if( ( node_index < this->numAvailableNodes() ) &&
            ( base_ctx_t::doSelectNode( node_index ) ) )
        {
            base_ctx_t::doInitDefaultKernels();
            this->doInitDefaultKernelsPrivImpl();

            this->doAssignSlotSizeArgPrivImpl(
                st::BUFFER_DEFAULT_SLOT_SIZE );

            this->doAssignElemByElemConfigIndexArgPrivImpl(
                ctx_t::DEFAULT_ELEM_BY_ELEM_CONFIG_INDEX );

            this->doAssignStatusFlagsArgPrivImpl(
                this->internalStatusFlagsBuffer() );
        }
    }

    ctx_t::ClContext( char const* node_id_str,
                          const char *const SIXTRL_RESTRICT config_str ) :
        base_ctx_t( config_str ),
        m_num_particles_in_pset( st_size_t{ 0 } ),
        m_pset_index( st_size_t{ 0 } ),
        m_elem_by_elem_config_index( ctx_t::DEFAULT_ELEM_BY_ELEM_CONFIG_INDEX ),
        m_track_until_turn_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_line_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_elem_by_elem_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_be_mon_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_fetch_particles_addr_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_clear_be_mon_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_addr_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_track_line_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_elem_by_elem_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_be_mon_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_fetch_particles_addr_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_clear_be_mon_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_addr_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_use_optimized_tracking( false )
    {
        st_size_t  node_index = this->findAvailableNodesIndex( node_id_str );

        /* WARNING: Workaround for AMD Heisenbug */
        if( !this->isAvailableNodeAMDPlatform( node_index ) )
        {
            this->m_use_optimized_tracking = true;
        }

        ctx_t::status_t const status =
            this->doInitDefaultFeatureFlagsPrivImpl();
        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;

        this->doInitDefaultProgramsPrivImpl();

        if( node_index >= this->numAvailableNodes() )
        {
            ctx_t::node_id_t const default_node_id = this->defaultNodeId();

            ctx_t::platform_id_t const platform_index =
                NS(ComputeNodeId_get_platform_id)( &default_node_id );

            ctx_t::device_id_t const device_index =
                NS(ComputeNodeId_get_device_id)( &default_node_id );

            node_index = this->findAvailableNodesIndex(
                platform_index, device_index );
        }

        if( ( node_index < this->numAvailableNodes() ) &&
            ( base_ctx_t::doSelectNode( node_index ) ) )
        {
            base_ctx_t::doInitDefaultKernels();
            this->doInitDefaultKernelsPrivImpl();

            this->doAssignSlotSizeArgPrivImpl(
                st::BUFFER_DEFAULT_SLOT_SIZE );

            this->doAssignElemByElemConfigIndexArgPrivImpl(
                ctx_t::DEFAULT_ELEM_BY_ELEM_CONFIG_INDEX );

            this->doAssignStatusFlagsArgPrivImpl(
                this->internalStatusFlagsBuffer() );
        }
    }

    ctx_t::ClContext(
        ctx_t::platform_id_t const platform_idx,
        ctx_t::device_id_t const device_idx,
        const char *const SIXTRL_RESTRICT config_str ) :
        base_ctx_t( config_str ),
        m_num_particles_in_pset( st_size_t{ 0 } ),
        m_pset_index( st_size_t{ 0 } ),
        m_elem_by_elem_config_index( ctx_t::DEFAULT_ELEM_BY_ELEM_CONFIG_INDEX ),
        m_track_until_turn_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_line_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_elem_by_elem_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_be_mon_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_fetch_particles_addr_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_clear_be_mon_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_addr_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_track_line_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_elem_by_elem_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_be_mon_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_fetch_particles_addr_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_clear_be_mon_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_addr_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_use_optimized_tracking( false )
    {
        st_size_t const node_index =
            this->findAvailableNodesIndex( platform_idx, device_idx );

        /* WARNING: Workaround for AMD Heisenbug */
        if( !this->isAvailableNodeAMDPlatform( node_index ) )
        {
            this->m_use_optimized_tracking = true;
        }

        ctx_t::status_t const status =
            this->doInitDefaultFeatureFlagsPrivImpl();
        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;

        this->doInitDefaultProgramsPrivImpl();

        if( ( node_index < this->numAvailableNodes() ) &&
            ( base_ctx_t::doSelectNode( node_index ) ) )
        {
            base_ctx_t::doInitDefaultKernels();
            this->doInitDefaultKernelsPrivImpl();

            this->doAssignSlotSizeArgPrivImpl(
                st::BUFFER_DEFAULT_SLOT_SIZE );

            this->doAssignElemByElemConfigIndexArgPrivImpl(
                this->m_elem_by_elem_config_index );

            this->doAssignStatusFlagsArgPrivImpl(
                this->internalStatusFlagsBuffer() );
        }
    }

    ctx_t::~ClContext() SIXTRL_NOEXCEPT {}

    st_status_t ctx_t::assign_particles_arg(
        ctx_t::cl_argument_t& SIXTRL_RESTRICT_REF particles_arg )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( !particles_arg.usesCObjectBuffer() ) ||
            (  particles_arg.ptrCObjectBuffer() == nullptr ) )
        {
            return status;
        }

        constexpr st_size_t NUM_KERNELS = st_size_t{ 4 };
        status = st::ARCH_STATUS_SUCCESS;

        st_kernel_id_t kernel_ids[ NUM_KERNELS ] =
        {
            st::ARCH_ILLEGAL_KERNEL_ID, st::ARCH_ILLEGAL_KERNEL_ID,
            st::ARCH_ILLEGAL_KERNEL_ID, st::ARCH_ILLEGAL_KERNEL_ID
        };

        kernel_ids[ 0 ] = this->track_until_kernel_id();
        kernel_ids[ 1 ] = this->track_line_kernel_id();
        kernel_ids[ 2 ] = this->track_elem_by_elem_kernel_id();
        kernel_ids[ 3 ] = this->fetch_particles_addr_kernel_id();

        size_t const min_num_kernel_args[ NUM_KERNELS ] =
        {
            ctx_t::MIN_NUM_TRACK_UNTIL_ARGS, ctx_t::MIN_NUM_TRACK_LINE_ARGS,
            ctx_t::MIN_NUM_TRACK_ELEM_ARGS,
            ctx_t::MIN_NUM_FETCH_PARTICLES_ADDR_ARGS
        };

        st_size_t const particles_arg_idx[ NUM_KERNELS ] =
        {
            st_size_t{ 0 }, // track_until
            st_size_t{ 0 }, // track_line
            st_size_t{ 0 }, // track_elem_elem
            st_size_t{ 0 }  // fetch_particles_addr
        };

        for( st_size_t ii = st_size_t{ 0 } ; ii < NUM_KERNELS ; ++ii )
        {
            st_kernel_id_t const kernel_id = kernel_ids[ ii ];
            st_size_t const min_num_args = min_num_kernel_args[ ii ];

            if( kernel_id == st::ARCH_ILLEGAL_KERNEL_ID ) continue;
            if( min_num_args == st_size_t{ 0 } ) continue;

            st_size_t const num_args = this->kernelNumArgs( kernel_id );
            st_size_t const arg_idx = particles_arg_idx[ ii ];

            if( ( num_args <= arg_idx ) || ( num_args < min_num_args ) )
            {
                status |= st::ARCH_STATUS_GENERAL_FAILURE;
                continue;
            }

            this->assignKernelArgument( kernel_id, arg_idx, particles_arg );
        }

        return status;
    }

    st_status_t ctx_t::assign_particle_set_arg(
        st_size_t const particle_set_index,
        st_size_t const num_particles_in_selected_set )
    {
        st_status_t status = st::ARCH_STATUS_SUCCESS;
        constexpr size_t NUM_KERNELS = size_t{ 3 };

        st_kernel_id_t kernel_ids[ NUM_KERNELS ] =
        {
            st::ARCH_ILLEGAL_KERNEL_ID, st::ARCH_ILLEGAL_KERNEL_ID,
            st::ARCH_ILLEGAL_KERNEL_ID
        };

        kernel_ids[ 0 ] = this->track_until_kernel_id();
        kernel_ids[ 1 ] = this->track_line_kernel_id();
        kernel_ids[ 2 ] = this->track_elem_by_elem_kernel_id();

        st_size_t const min_num_kernel_args[ NUM_KERNELS ] =
        {
            ctx_t::MIN_NUM_TRACK_UNTIL_ARGS, ctx_t::MIN_NUM_TRACK_LINE_ARGS,
            ctx_t::MIN_NUM_TRACK_ELEM_ARGS
        };

        st_size_t const pset_arg_idx[ NUM_KERNELS ] =
        {
            st_size_t{ 1 }, st_size_t{ 1 }, st_size_t{ 1 },
        };

        uint64_t const pset_idx_arg =
            static_cast< uint64_t >( particle_set_index );

        for( st_size_t ii = st_size_t{ 0 } ; ii < NUM_KERNELS ; ++ii )
        {
            st_kernel_id_t const kernel_id = kernel_ids[ ii ];
            st_size_t const min_num_args = min_num_kernel_args[ ii ];

            if( kernel_id == st::ARCH_ILLEGAL_KERNEL_ID ) continue;
            if( min_num_args == st_size_t{ 0 } ) continue;

            st_size_t const num_args = this->kernelNumArgs( kernel_id );
            st_size_t const arg_idx  = pset_arg_idx[ ii ];

            if( ( num_args <= arg_idx ) || ( num_args < min_num_args ) )
            {
                status |= st::ARCH_STATUS_GENERAL_FAILURE;
                continue;
            }

            this->assignKernelArgumentValue( kernel_id, arg_idx, pset_idx_arg );
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            this->m_pset_index = particle_set_index;
            this->m_num_particles_in_pset = num_particles_in_selected_set;
        }

        return status;
    }

    st_status_t ctx_t::assign_beam_elements_arg(
        ctx_t::cl_argument_t& SIXTRL_RESTRICT_REF beam_elements_arg )
    {
        using size_t       = st_size_t;
        using st_kernel_id_t  = st_kernel_id_t;

        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( !beam_elements_arg.usesCObjectBuffer() ) ||
            (  beam_elements_arg.ptrCObjectBuffer() == nullptr ) ||
            ( size_t{ 1 } > ::NS(Buffer_get_num_of_objects)(
                beam_elements_arg.ptrCObjectBuffer() ) ) )
        {
            return status;
        }

        constexpr size_t NUM_KERNELS = size_t{ 5 };

        status = st::ARCH_STATUS_SUCCESS;

        st_kernel_id_t kernel_ids[ NUM_KERNELS ];
        std::fill( &kernel_ids[ 0 ], &kernel_ids[ NUM_KERNELS ],
                   st::ARCH_ILLEGAL_KERNEL_ID );

        kernel_ids[ 0 ] = this->track_until_kernel_id();
        kernel_ids[ 1 ] = this->track_line_kernel_id();
        kernel_ids[ 2 ] = this->track_elem_by_elem_kernel_id();
        kernel_ids[ 3 ] = this->assign_beam_monitor_output_kernel_id();
        kernel_ids[ 4 ] = this->clear_beam_monitor_output_kernel_id();

        size_t const min_num_kernel_args[ NUM_KERNELS ] =
        {
            ctx_t::MIN_NUM_TRACK_UNTIL_ARGS, ctx_t::MIN_NUM_TRACK_LINE_ARGS,
            ctx_t::MIN_NUM_TRACK_ELEM_ARGS,
            ctx_t::MIN_NUM_ASSIGN_BE_MON_ARGS,
            ctx_t::MIN_NUM_CLEAR_BE_MON_ARGS
        };

        size_t const beam_elems_arg_idx[ NUM_KERNELS ] =
        {
            size_t{ 2 }, // track_until
            size_t{ 2 }, // track_line
            size_t{ 2 }, // track_elem_elem
            size_t{ 0 }, // assign_be_mon
            size_t{ 0 }, // clear_be_mon
        };

        for( size_t ii = size_t{ 0 } ; ii < NUM_KERNELS ; ++ii )
        {
            st_kernel_id_t const kernel_id = kernel_ids[ ii ];
            size_t const min_num_args = min_num_kernel_args[ ii ];

            if( kernel_id == st::ARCH_ILLEGAL_KERNEL_ID ) continue;
            if( min_num_args == size_t{ 0 } ) continue;

            size_t const num_args = this->kernelNumArgs( kernel_id );
            size_t const arg_idx = beam_elems_arg_idx[ ii ];

            if( ( num_args <= arg_idx ) || ( num_args < min_num_args ) )
            {
                status |= st::ARCH_STATUS_GENERAL_FAILURE;
                continue;
            }

            this->assignKernelArgument( kernel_id, arg_idx, beam_elements_arg );
        }

        return status;
    }

    st_status_t ctx_t::assign_output_buffer_arg(
        ctx_t::cl_argument_t& SIXTRL_RESTRICT_REF out_buffer_arg )
    {
        using size_t       = st_size_t;
        using st_kernel_id_t  = st_kernel_id_t;

        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( !out_buffer_arg.usesCObjectBuffer() ) ||
            (  out_buffer_arg.ptrCObjectBuffer() == nullptr ) )
        {
            return status;
        }

        status = st::ARCH_STATUS_SUCCESS;
        constexpr size_t NUM_KERNELS = size_t{ 2 };

        st_kernel_id_t kernel_ids[ NUM_KERNELS ] =
        {
            st::ARCH_ILLEGAL_KERNEL_ID, st::ARCH_ILLEGAL_KERNEL_ID
        };

        kernel_ids[ 0 ] = this->assign_beam_monitor_output_kernel_id();
        kernel_ids[ 1 ] = this->assign_elem_by_elem_output_kernel_id();

        size_t const min_num_kernel_args[ NUM_KERNELS ] =
        {
            ctx_t::MIN_NUM_ASSIGN_BE_MON_ARGS,
            ctx_t::MIN_NUM_ASSIGN_ELEM_ARGS
        };

        size_t const out_buffer_arg_idx[ NUM_KERNELS ] =
        {
            size_t{ 1 }, // assign_beam_monitor_output_kernel
            size_t{ 2 }, // assign_elem_by_elem_output_kernel
        };

        for( size_t ii = size_t{ 0 } ; ii < NUM_KERNELS ; ++ii )
        {
            st_kernel_id_t const kernel_id = kernel_ids[ ii ];
            size_t const min_num_args = min_num_kernel_args[ ii ];

            if( kernel_id == st::ARCH_ILLEGAL_KERNEL_ID ) continue;
            if( min_num_args == size_t{ 0 } ) continue;

            size_t const num_args = this->kernelNumArgs( kernel_id );
            size_t const arg_idx  = out_buffer_arg_idx[ ii ];

            if( ( num_args <= arg_idx ) || ( num_args < min_num_args ) )
            {
                status |= st::ARCH_STATUS_GENERAL_FAILURE;
                continue;
            }

            this->assignKernelArgument( kernel_id, arg_idx, out_buffer_arg );
        }

        return status;
    }

    st_status_t ctx_t::assign_elem_by_elem_config_buffer_arg(
            ctx_t::cl_argument_t& SIXTRL_RESTRICT_REF config_buffer_arg )
    {
        constexpr st_size_t NUM_KERNELS = st_size_t{ 2 };
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( !config_buffer_arg.usesCObjectBuffer() ) ||
            (  config_buffer_arg.ptrCObjectBuffer() == nullptr ) )
        {
            return status;
        }

        status = st::ARCH_STATUS_SUCCESS;

        st_kernel_id_t kernel_ids[ NUM_KERNELS ] =
        {
            st::ARCH_ILLEGAL_KERNEL_ID, st::ARCH_ILLEGAL_KERNEL_ID
        };

        kernel_ids[ 0 ] = this->track_elem_by_elem_kernel_id();
        kernel_ids[ 1 ] = this->assign_elem_by_elem_output_kernel_id();

        st_size_t const min_num_kernel_args[ NUM_KERNELS ] =
        {
            ctx_t::MIN_NUM_TRACK_ELEM_ARGS,
            ctx_t::MIN_NUM_ASSIGN_ELEM_ARGS
        };

        st_size_t const elem_by_elem_arg_idx[ NUM_KERNELS ] =
        {
            st_size_t{ 3 }, // track_elem_by_elem_kernel
            st_size_t{ 0 }, // assign_elem_by_elem_output_kernel
        };

        for( st_size_t ii = st_size_t{ 0 } ; ii < NUM_KERNELS ; ++ii )
        {
            st_kernel_id_t const kernel_id = kernel_ids[ ii ];
            st_size_t const min_num_args = min_num_kernel_args[ ii ];

            if( kernel_id == st::ARCH_ILLEGAL_KERNEL_ID ) continue;
            if( min_num_args == st_size_t{ 0 } ) continue;

            st_size_t const num_args = this->kernelNumArgs( kernel_id );
            st_size_t const arg_idx  = elem_by_elem_arg_idx[ ii ];

            if( ( num_args <= arg_idx ) || ( num_args < min_num_args ) )
            {
                status |= st::ARCH_STATUS_GENERAL_FAILURE;
                continue;
            }

            this->assignKernelArgument( kernel_id, arg_idx, config_buffer_arg );
        }

        return status;
    }

    st_status_t ctx_t::assign_particles_addr_buffer_arg(
        ClArgument& SIXTRL_RESTRICT_REF particles_addr_arg )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( !particles_addr_arg.usesCObjectBuffer() ) ||
            (  particles_addr_arg.ptrCObjectBuffer() == nullptr ) )
        {
            return status;
        }

        status = st::ARCH_STATUS_SUCCESS;
        st_kernel_id_t const kernel_id = this->fetch_particles_addr_kernel_id();

        if( ( kernel_id != st::ARCH_ILLEGAL_KERNEL_ID ) &&
            ( st_size_t{ 1 } <= ctx_t::MIN_NUM_FETCH_PARTICLES_ADDR_ARGS ) &&
            ( ctx_t::MIN_NUM_FETCH_PARTICLES_ADDR_ARGS <=
                this->kernelNumArgs( kernel_id ) ) )
        {
            this->assignKernelArgument( kernel_id, st_size_t{ 1 },
                                        particles_addr_arg );

            status = st::ARCH_STATUS_SUCCESS;
        }

        return status;
    }

    st_status_t ctx_t::assign_elem_by_elem_config_index_arg(
        st_size_t const elem_by_elem_config_index )
    {
        return this->doAssignElemByElemConfigIndexArg(
            elem_by_elem_config_index );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ctx_t::has_track_until_kernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
                 ( this->m_track_until_turn_kernel_id >= st_kernel_id_t{ 0 } ) &&
                 ( static_cast< st_size_t >( this->m_track_until_turn_kernel_id )
                    < this->numAvailableKernels() ) );
    }

    st_kernel_id_t ctx_t::track_until_kernel_id() const SIXTRL_NOEXCEPT
    {
        return ( this->has_track_until_kernel() )
            ? this->m_track_until_turn_kernel_id : st::ARCH_ILLEGAL_KERNEL_ID;
    }

    st_status_t ctx_t::set_track_until_kernel_id(
        st_kernel_id_t const kernel_id )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->hasSelectedNode() ) && ( kernel_id >= st_kernel_id_t{ 0 } ) &&
            ( static_cast< st_size_t >( kernel_id ) <
                this->numAvailableKernels() ) )
        {
            st_program_id_t const program_id =
                this->programIdByKernelId( kernel_id );

            if( ( program_id >= st_program_id_t{ 0 } ) &&
                ( static_cast< st_size_t >( program_id ) <
                    this->numAvailablePrograms() ) )
            {
                this->m_track_until_turn_kernel_id  = kernel_id;
                this->m_track_until_turn_program_id = program_id;
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    ctx_t::track_status_t ctx_t::track_until(
        ctx_t::num_turns_t const until_turn )
    {
        ctx_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;
        st_kernel_id_t const kernel_id = this->track_until_kernel_id();
        st_size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= ctx_t::MIN_NUM_TRACK_UNTIL_ARGS ) )
        {
            SIXTRL_ASSERT( ctx_t::MIN_NUM_TRACK_UNTIL_ARGS >= st_size_t{ 4 } );
            SIXTRL_ASSERT( this->m_num_particles_in_pset > st_size_t{ 0 } );
            SIXTRL_ASSERT( kernel_id >= st_kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< st_size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            int64_t const until_turn_arg = static_cast< int64_t >( until_turn );

            this->assignKernelArgumentValue(
                kernel_id, st_size_t{ 3 }, until_turn_arg );

            if( !this->debugMode() )
            {
                if( this->runKernel( kernel_id, this->m_num_particles_in_pset,
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = st::TRACK_SUCCESS;
                }
            }
            else if( this->prepare_status_flags_for_use() ==
                     st::ARCH_STATUS_SUCCESS )
            {
                if( this->runKernel( kernel_id, this->m_num_particles_in_pset,
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = static_cast< ctx_t::track_status_t >(
                        this->eval_status_flags_after_use() );
                }
            }
        }

        return status;
    }

    ctx_t::track_status_t ctx_t::track_until(
        ctx_t::num_turns_t const until_turn, st_size_t const pset_index,
        st_size_t const num_particles_in_set, bool const restore_pset_index )
    {
        ctx_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;
        st_kernel_id_t const kernel_id = this->track_until_kernel_id();
        st_size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= ctx_t::MIN_NUM_TRACK_UNTIL_ARGS ) )
        {
            SIXTRL_ASSERT( ctx_t::MIN_NUM_TRACK_UNTIL_ARGS >= st_size_t{ 4 } );
            SIXTRL_ASSERT( num_particles_in_set != st_size_t{ 0 } );
            SIXTRL_ASSERT( this->m_num_particles_in_pset > st_size_t{ 0 } );
            SIXTRL_ASSERT( kernel_id >= st_kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< st_size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            if( this->m_pset_index != pset_index )
            {
                uint64_t const pset_index_arg =
                    static_cast< uint64_t >( pset_index );

                this->assignKernelArgumentValue(
                    kernel_id, st_size_t{ 1 }, pset_index_arg );
            }

            int64_t const until_turn_arg = static_cast< int64_t >( until_turn );

            this->assignKernelArgumentValue(
                kernel_id, st_size_t{ 3 }, until_turn_arg );

            if( !this->debugMode() )
            {
                if( this->runKernel( kernel_id, num_particles_in_set,
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = st::TRACK_SUCCESS;
                }
            }
            else if( this->prepare_status_flags_for_use() ==
                     st::ARCH_STATUS_SUCCESS )
            {
                if( this->runKernel( kernel_id, num_particles_in_set,
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = static_cast< ctx_t::track_status_t >(
                        this->eval_status_flags_after_use() );
                }
            }

            if( ( this->m_pset_index != pset_index ) && ( restore_pset_index ) )
            {
                uint64_t const pset_index_arg =
                    static_cast< uint64_t >( this->m_pset_index );

                this->assignKernelArgumentValue(
                    kernel_id, st_size_t{ 1 }, pset_index_arg );
            }
        }

        return status;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ctx_t::has_track_line_kernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
            ( this->m_track_line_kernel_id >= st_kernel_id_t{ 0 } ) &&
                ( static_cast< st_size_t >( this->m_track_line_kernel_id ) <
                    this->numAvailableKernels() ) );
    }

    st_kernel_id_t ctx_t::track_line_kernel_id() const SIXTRL_NOEXCEPT
    {
        return ( this->has_track_until_kernel() )
            ? this->m_track_line_kernel_id : st::ARCH_ILLEGAL_KERNEL_ID;
    }

    st_status_t ctx_t::set_track_line_kernel_id(
        st_kernel_id_t const kernel_id )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->hasSelectedNode() ) && ( kernel_id >= st_kernel_id_t{ 0 } ) &&
            ( static_cast< st_size_t >( kernel_id ) < this->numAvailableKernels() ) )
        {
            st_program_id_t const program_id =
                this->programIdByKernelId( kernel_id );

            if( ( program_id >= st_program_id_t{ 0 } ) &&
                ( static_cast< st_size_t >( program_id ) <
                    this->numAvailablePrograms() ) )
            {
                this->m_track_line_kernel_id  = kernel_id;
                this->m_track_line_program_id = program_id;
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    ctx_t::track_status_t ctx_t::track_line( st_size_t const line_begin_idx,
        st_size_t const line_end_idx, bool const finish_turn )
    {
        ctx_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;
        st_kernel_id_t const kernel_id = this->track_line_kernel_id();
        st_size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= ctx_t::MIN_NUM_TRACK_LINE_ARGS ) )
        {
            SIXTRL_ASSERT( ctx_t::MIN_NUM_TRACK_LINE_ARGS >= st_size_t{ 7 } );
            SIXTRL_ASSERT( this->m_num_particles_in_pset > st_size_t{ 0 } );
            SIXTRL_ASSERT( kernel_id >= st_kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< st_size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            uint64_t const line_begin_idx_arg =
                static_cast< uint64_t >( line_begin_idx );

            uint64_t const line_end_idx_arg =
                static_cast< uint64_t >( line_end_idx );

            uint64_t const finish_turn_arg = ( finish_turn )
                ? uint64_t{ 1 } : uint64_t{ 0 };

            this->assignKernelArgumentValue(
                kernel_id, st_size_t{ 3 }, line_begin_idx_arg );

            this->assignKernelArgumentValue(
                kernel_id, st_size_t{ 4 }, line_end_idx_arg );

            this->assignKernelArgumentValue(
                kernel_id, st_size_t{ 5 }, finish_turn_arg );

            if( !this->debugMode() )
            {
                if( this->runKernel( kernel_id, this->m_num_particles_in_pset,
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = st::TRACK_SUCCESS;
                }
            }
            else if( this->prepare_status_flags_for_use() ==
                     st::ARCH_STATUS_SUCCESS )
            {
                if( this->runKernel( kernel_id, this->m_num_particles_in_pset,
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = static_cast< ctx_t::track_status_t >(
                        this->eval_status_flags_after_use() );
                }
            }
        }

        return status;
    }

    ctx_t::track_status_t ctx_t::track_line(
        ctx_t::size_type const line_begin_idx,
        ctx_t::size_type const line_end_idx,
        bool const finish_turn,
        ctx_t::size_type const pset_index,
        ctx_t::size_type const num_particles_in_set,
        bool const restore_pset_index )
    {
        ctx_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;
        st_kernel_id_t const kernel_id = this->track_line_kernel_id();
        st_size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= ctx_t::MIN_NUM_TRACK_LINE_ARGS ) )
        {
            SIXTRL_ASSERT( ctx_t::MIN_NUM_TRACK_LINE_ARGS >= st_size_t{ 7 } );
            SIXTRL_ASSERT( num_particles_in_set != st_size_t{ 0 } );
            SIXTRL_ASSERT( kernel_id >= st_kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< st_size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            if( this->m_pset_index != pset_index )
            {
                uint64_t const pset_index_arg =
                    static_cast< uint64_t >( pset_index );

                this->assignKernelArgumentValue(
                    kernel_id, st_size_t{ 1 }, pset_index_arg );
            }

            uint64_t const line_begin_idx_arg =
                static_cast< uint64_t >( line_begin_idx );

            uint64_t const line_end_idx_arg =
                static_cast< uint64_t >( line_end_idx );

            uint64_t const finish_turn_arg = ( finish_turn )
                ? uint64_t{ 1 } : uint64_t{ 0 };

            this->assignKernelArgumentValue(
                kernel_id, st_size_t{ 3 }, line_begin_idx_arg );

            this->assignKernelArgumentValue(
                kernel_id, st_size_t{ 4 }, line_end_idx_arg );

            this->assignKernelArgumentValue(
                kernel_id, st_size_t{ 5 }, finish_turn_arg );

            if( !this->debugMode() )
            {
                if( this->runKernel( kernel_id, num_particles_in_set,
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = st::TRACK_SUCCESS;
                }
            }
            else if( this->prepare_status_flags_for_use() ==
                     st::ARCH_STATUS_SUCCESS )
            {
                if( this->runKernel( kernel_id, num_particles_in_set,
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = static_cast< ctx_t::track_status_t >(
                        this->eval_status_flags_after_use() );
                }
            }

            if( ( this->m_pset_index != pset_index ) &&
                ( restore_pset_index ) )
            {
                uint64_t const pset_index_arg =
                    static_cast< uint64_t >( this->m_pset_index );

                this->assignKernelArgumentValue(
                    kernel_id, st_size_t{ 1 }, pset_index_arg );
            }
        }

        return status;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ctx_t::has_track_elem_by_elem_kernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
            ( this->m_track_elem_by_elem_kernel_id >= st_kernel_id_t{ 0 } ) &&
            ( static_cast< st_size_t >( this->m_track_elem_by_elem_kernel_id ) <
                         this->numAvailableKernels() ) );
    }

    st_kernel_id_t ctx_t::track_elem_by_elem_kernel_id() const SIXTRL_NOEXCEPT
    {
        return ( this->has_track_elem_by_elem_kernel() )
            ? this->m_track_elem_by_elem_kernel_id : st::ARCH_ILLEGAL_KERNEL_ID;
    }

    st_status_t ctx_t::set_track_elem_by_elem_kernel_id(
        st_kernel_id_t const kernel_id )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->hasSelectedNode() ) &&
            ( kernel_id >= st_kernel_id_t{ 0 } ) &&
            ( static_cast< st_size_t >( kernel_id ) <
                this->numAvailableKernels() ) )
        {
            st_program_id_t const program_id =
                this->programIdByKernelId( kernel_id );

            if( ( program_id >= st_program_id_t{ 0 } ) &&
                ( static_cast< st_size_t >( program_id ) <
                  this->numAvailablePrograms() ) )
            {
                this->m_track_elem_by_elem_kernel_id  = kernel_id;
                this->m_track_elem_by_elem_program_id = program_id;
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    ctx_t::track_status_t ctx_t::track_elem_by_elem(
        st_size_t const until_turn )
    {
        ctx_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        st_kernel_id_t const kernel_id =
            this->track_elem_by_elem_kernel_id();

        st_size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= ctx_t::MIN_NUM_TRACK_ELEM_ARGS ) )
        {
            SIXTRL_ASSERT( ctx_t::MIN_NUM_TRACK_ELEM_ARGS >= st_size_t{ 6 } );
            SIXTRL_ASSERT( this->m_num_particles_in_pset > st_size_t{ 0 } );
            SIXTRL_ASSERT( kernel_id >= st_kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< st_size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            int64_t const until_turn_arg = static_cast< int64_t >( until_turn );

            this->assignKernelArgumentValue(
                kernel_id, st_size_t{ 5 }, until_turn_arg );

            if( !this->debugMode() )
            {
                if( this->runKernel( kernel_id, this->m_num_particles_in_pset,
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = st::TRACK_SUCCESS;
                }
            }
            else if( this->prepare_status_flags_for_use() ==
                     st::ARCH_STATUS_SUCCESS )
            {
                if( this->runKernel( kernel_id, this->m_num_particles_in_pset,
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = static_cast< ctx_t::track_status_t >(
                        this->eval_status_flags_after_use() );
                }
            }
        }

        return status;
    }

    ctx_t::track_status_t ctx_t::track_elem_by_elem(
        st_size_t const until_turn, st_size_t const pset_index,
        st_size_t const num_particles_in_set, bool const restore_pset_index )
    {
        ctx_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        st_kernel_id_t const kernel_id =
            this->track_elem_by_elem_kernel_id();

        st_size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= ctx_t::MIN_NUM_TRACK_ELEM_ARGS ) )
        {
            SIXTRL_ASSERT( ctx_t::MIN_NUM_TRACK_ELEM_ARGS >= st_size_t{ 6 } );
            SIXTRL_ASSERT( num_particles_in_set != st_size_t{ 0 } );
            SIXTRL_ASSERT( kernel_id >= st_kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< st_size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            if( this->m_pset_index != pset_index )
            {
                uint64_t const pset_index_arg =
                    static_cast< uint64_t >( pset_index );

                this->assignKernelArgumentValue(
                    kernel_id, st_size_t{ 1 }, pset_index_arg );
            }

            int64_t const until_turn_arg = static_cast< int64_t >( until_turn );

            this->assignKernelArgumentValue(
                kernel_id, st_size_t{ 5 }, until_turn_arg );

            if( !this->debugMode() )
            {
                if( this->runKernel( kernel_id, this->m_num_particles_in_pset,
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = st::TRACK_SUCCESS;
                }
            }
            else if( this->prepare_status_flags_for_use() ==
                     st::ARCH_STATUS_SUCCESS )
            {
                if( this->runKernel( kernel_id, this->m_num_particles_in_pset,
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = static_cast< ctx_t::track_status_t >(
                        this->eval_status_flags_after_use() );
                }
            }

            if( ( this->m_pset_index != pset_index ) &&
                ( restore_pset_index ) )
            {
                uint64_t const pset_index_arg =
                    static_cast< uint64_t >( this->m_pset_index );

                this->assignKernelArgumentValue(
                    kernel_id, st_size_t{ 1 }, pset_index_arg );
            }
        }

        return status;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ctx_t::has_assign_beam_monitor_output_kernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
            ( this->m_assign_be_mon_out_buffer_kernel_id >=
                st_kernel_id_t{ 0 } ) &&
            ( static_cast< st_size_t >(
                this->m_assign_be_mon_out_buffer_kernel_id ) <
                this->numAvailableKernels() ) );
    }

    st_kernel_id_t ctx_t::assign_beam_monitor_output_kernel_id(
        ) const SIXTRL_NOEXCEPT
    {
        return ( this->has_assign_beam_monitor_output_kernel() )
            ? this->m_assign_be_mon_out_buffer_kernel_id
            : st::ARCH_ILLEGAL_KERNEL_ID;
    }

    st_status_t ctx_t::set_assign_beam_monitor_output_kernel_id(
        st_kernel_id_t const kernel_id )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->hasSelectedNode() ) && ( kernel_id >= st_kernel_id_t{ 0 } ) &&
            ( this->numAvailableKernels() >
                static_cast< st_size_t >( kernel_id ) ) )
        {
            st_program_id_t const program_id =
                this->programIdByKernelId( kernel_id );

            if( ( program_id >= st_program_id_t{ 0 } ) &&
                ( static_cast< st_size_t >( program_id ) <
                    this->numAvailablePrograms() ) )
            {
                this->m_assign_be_mon_out_buffer_kernel_id  = kernel_id;
                this->m_assign_be_mon_out_buffer_program_id = program_id;
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    st_status_t ctx_t::assign_beam_monitor_output(
        ctx_t::particle_index_t const min_turn_id,
        st_size_t const out_buffer_index_offset  )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        st_kernel_id_t const kernel_id =
            this->assign_beam_monitor_output_kernel_id();

        st_size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= ctx_t::MIN_NUM_ASSIGN_BE_MON_ARGS ) )
        {
            SIXTRL_ASSERT( ctx_t::MIN_NUM_ASSIGN_BE_MON_ARGS >= st_size_t{ 5 } );
            SIXTRL_ASSERT( kernel_id >= st_kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< st_size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            int64_t const min_turn_id_arg =
                static_cast< int64_t >( min_turn_id );

            uint64_t const out_buffer_index_offset_arg =
                static_cast< uint64_t >( out_buffer_index_offset );

            this->assignKernelArgumentValue(
                kernel_id, st_size_t{ 2 }, min_turn_id_arg );

            this->assignKernelArgumentValue(
                kernel_id, st_size_t{ 3 }, out_buffer_index_offset_arg );

            if( !this->debugMode() )
            {
                if( this->runKernel( kernel_id, st_size_t{ 1 },
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = st::TRACK_SUCCESS;
                }
            }
            else if( this->prepare_status_flags_for_use() ==
                     st::ARCH_STATUS_SUCCESS )
            {
                if( this->runKernel( kernel_id, st_size_t{ 1 },
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = this->eval_status_flags_after_use();
                }
            }
        }

        return status;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ctx_t::has_assign_elem_by_elem_output_kernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
            ( this->m_assign_elem_by_elem_out_buffer_kernel_id >=
                st_kernel_id_t{ 0 } ) &&
            ( static_cast< st_size_t >(
                this->m_assign_elem_by_elem_out_buffer_kernel_id ) <
                this->numAvailableKernels() ) );
    }

    st_kernel_id_t ctx_t::assign_elem_by_elem_output_kernel_id() const SIXTRL_NOEXCEPT
    {
        return ( this->has_assign_elem_by_elem_output_kernel() )
            ? this->m_assign_elem_by_elem_out_buffer_kernel_id
            : st::ARCH_ILLEGAL_KERNEL_ID;
    }

    st_status_t ctx_t::set_assign_elem_by_elem_output_kernel_id(
        st_kernel_id_t const kernel_id )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->hasSelectedNode() ) && ( kernel_id >= st_kernel_id_t{ 0 } ) &&
            ( this->numAvailableKernels() >
                static_cast< st_size_t >( kernel_id ) ) )
        {
            st_program_id_t const program_id = this->programIdByKernelId(
                kernel_id );

            if( ( program_id >= st_program_id_t{ 0 } ) &&
                ( static_cast< st_size_t >( program_id ) <
                    this->numAvailablePrograms() ) )
            {
                this->m_assign_elem_by_elem_out_buffer_kernel_id  = kernel_id;
                this->m_assign_elem_by_elem_out_buffer_program_id = program_id;
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    st_status_t ctx_t::assign_elem_by_elem_output(
        ctx_t::size_type const out_buffer_index_offset )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        st_kernel_id_t const kernel_id =
            this->assign_elem_by_elem_output_kernel_id();

        st_size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= ctx_t::MIN_NUM_ASSIGN_ELEM_ARGS ) )
        {
            SIXTRL_ASSERT( ctx_t::MIN_NUM_ASSIGN_ELEM_ARGS >= st_size_t{ 4 } );
            SIXTRL_ASSERT( kernel_id >= st_kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< st_size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            uint64_t const elem_by_elem_config_index_arg =
                static_cast< uint64_t >( this->m_elem_by_elem_config_index );

            this->assignKernelArgumentValue(
                kernel_id, st_size_t{ 1 }, elem_by_elem_config_index_arg );

            uint64_t const out_buffer_index_offset_arg =
                static_cast< uint64_t >( out_buffer_index_offset );

            this->assignKernelArgumentValue(
                kernel_id, st_size_t{ 3 }, out_buffer_index_offset_arg );

            if( !this->debugMode() )
            {
                if( this->runKernel( kernel_id, st_size_t{ 1 },
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = st::TRACK_SUCCESS;
                }
            }
            else if( this->prepare_status_flags_for_use() ==
                     st::ARCH_STATUS_SUCCESS )
            {
                if( this->runKernel( kernel_id, st_size_t{ 1 },
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = this->eval_status_flags_after_use();
                }
            }
        }

        return status;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ctx_t::has_clear_beam_monitor_output_kernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
            ( this->m_clear_be_mon_kernel_id >= st_kernel_id_t{ 0 } ) &&
            ( static_cast< st_size_t >( this->m_clear_be_mon_kernel_id ) <
                this->numAvailableKernels() ) );
    }

    st_kernel_id_t ctx_t::clear_beam_monitor_output_kernel_id() const SIXTRL_NOEXCEPT
    {
        return ( this->has_clear_beam_monitor_output_kernel() )
            ? this->m_clear_be_mon_kernel_id : st::ARCH_ILLEGAL_KERNEL_ID;
    }

    st_status_t ctx_t::set_clear_beam_monitor_output_kernel_id(
        st_kernel_id_t const kernel_id )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->hasSelectedNode() ) && ( kernel_id >= st_kernel_id_t{ 0 } ) &&
            ( static_cast< st_size_t >( kernel_id ) <
                this->numAvailableKernels() ) )
        {
            st_program_id_t const program_id =
                this->programIdByKernelId( kernel_id );

            if( ( program_id >= st_program_id_t{ 0 } ) &&
                ( static_cast< st_size_t >( program_id ) <
                  this->numAvailablePrograms() ) )
            {
                this->m_clear_be_mon_kernel_id  = kernel_id;
                this->m_clear_be_mon_program_id = program_id;
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    st_status_t ctx_t::clear_beam_monitor_output()
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        st_kernel_id_t const kernel_id =
            this->clear_beam_monitor_output_kernel_id();

        st_size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= ctx_t::MIN_NUM_CLEAR_BE_MON_ARGS ) )
        {
            SIXTRL_ASSERT( ctx_t::MIN_NUM_CLEAR_BE_MON_ARGS >= st_size_t{ 2 } );
            SIXTRL_ASSERT( kernel_id >= st_kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< st_size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            if( !this->debugMode() )
            {
                if( this->runKernel( kernel_id, st_size_t{ 1 },
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = st::TRACK_SUCCESS;
                }
            }
            else if( this->prepare_status_flags_for_use() ==
                     st::ARCH_STATUS_SUCCESS )
            {
                if( this->runKernel( kernel_id, st_size_t{ 1 },
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = this->eval_status_flags_after_use();
                }
            }
        }

        return status;
    }

    /* --------------------------------------------------------------------- */

    bool ctx_t::has_fetch_particles_addr_kernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
            ( this->m_fetch_particles_addr_kernel_id >= st_kernel_id_t{ 0 } ) &&
            ( static_cast< st_size_t >( this->m_fetch_particles_addr_kernel_id
                ) < this->numAvailableKernels() ) );
    }

    st_kernel_id_t
    ctx_t::fetch_particles_addr_kernel_id() const SIXTRL_NOEXCEPT
    {
        return ( this->has_fetch_particles_addr_kernel() )
            ? this->m_fetch_particles_addr_kernel_id
            : st::ARCH_ILLEGAL_KERNEL_ID;
    }

    ctx_t::status_t ctx_t::set_fetch_particles_addr_kernel_id(
            st_kernel_id_t const kernel_id )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->hasSelectedNode() ) && ( kernel_id >= st_kernel_id_t{ 0 } ) &&
            ( static_cast< st_size_t >( kernel_id ) <
                this->numAvailableKernels() ) )
        {
            st_program_id_t const program_id =
                this->programIdByKernelId( kernel_id );

            if( ( program_id >= st_program_id_t{ 0 } ) &&
                ( static_cast< st_size_t >( program_id ) <
                  this->numAvailablePrograms() ) )
            {
                this->m_fetch_particles_addr_kernel_id  = kernel_id;
                this->m_fetch_particles_addr_program_id = program_id;
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    bool ctx_t::has_assign_addresses_kernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->m_assign_addr_kernel_id !=
            st::ARCH_ILLEGAL_KERNEL_ID ) &&
                 ( this->m_assign_addr_kernel_id <
                   this->numAvailableKernels() ) );
    }

    st_kernel_id_t ctx_t::assign_addresses_kernel_id() const SIXTRL_NOEXCEPT
    {
        return this->m_assign_addr_kernel_id;
    }

    ctx_t::status_t ctx_t::set_assign_addresses_kernel_id(
        st_kernel_id_t const kernel_id )
    {
        using size_t = st_size_t;
        using st_kernel_id_t = st_kernel_id_t;

        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->hasSelectedNode() ) && ( kernel_id >= st_kernel_id_t{ 0 } ) &&
            ( static_cast< size_t >( kernel_id ) < this->numAvailableKernels() ) )
        {
            st_program_id_t const program_id =
                this->programIdByKernelId( kernel_id );

            if( ( program_id >= st_program_id_t{ 0 } ) &&
                ( static_cast< st_size_t >( program_id ) <
                  this->numAvailablePrograms() ) )
            {
                this->m_assign_addr_kernel_id  = kernel_id;
                this->m_assign_addr_program_id = program_id;
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    ctx_t::status_t ctx_t::fetch_particles_addr()
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        st_kernel_id_t const kernel_id =
            this->fetch_particles_addr_kernel_id();

        if( this->hasSelectedNode() )
        {
            SIXTRL_ASSERT( kernel_id >= st_kernel_id_t{ 0 } );
            SIXTRL_ASSERT( ctx_t::MIN_NUM_FETCH_PARTICLES_ADDR_ARGS >=
                           this->kernelNumArgs( kernel_id ) );
            SIXTRL_ASSERT( static_cast< st_size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            if( this->runKernel( kernel_id, st_size_t{ 1 },
                this->lastExecWorkGroupSize( kernel_id ) ) )
            {
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    ctx_t::status_t ctx_t::assign_addresses(
        ctx_t::cl_argument_t& SIXTRL_RESTRICT_REF assign_items_arg,
        ctx_t::cl_argument_t& SIXTRL_RESTRICT_REF dest_buffer_arg,
        ctx_t::size_type const dest_buffer_id,
        ctx_t::cl_argument_t& SIXTRL_RESTRICT_REF src_buffer_arg,
        ctx_t::size_type const src_buffer_id )
    {
        using st_size_t = ctx_t::size_type;

        ctx_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        st_kernel_id_t const kernel_id =
            this->assign_addresses_kernel_id();

        if( ( kernel_id != st::ARCH_ILLEGAL_KERNEL_ID ) &&
            ( dest_buffer_id != st::ARCH_ILLEGAL_BUFFER_ID ) &&
            ( src_buffer_id  != st::ARCH_ILLEGAL_BUFFER_ID ) &&
            ( assign_items_arg.context() == this ) &&
            ( assign_items_arg.usesCObjectBuffer() ) &&
            ( assign_items_arg.ptrCObjectBuffer() != nullptr ) &&
            ( dest_buffer_arg.context() == this ) &&
            ( dest_buffer_arg.usesCObjectBuffer() ) &&
            ( dest_buffer_arg.ptrCObjectBuffer() != nullptr ) &&
            ( src_buffer_arg.context() == this ) &&
            ( src_buffer_arg.usesCObjectBuffer() ) &&
            ( src_buffer_arg.ptrCObjectBuffer() != nullptr ) )
        {
            st_size_t const num_items = ::NS(Buffer_get_num_of_objects)(
                assign_items_arg.ptrCObjectBuffer() );

            if( num_items == st_size_t{ 0 } )
            {
                return st::ARCH_STATUS_SUCCESS;
            }

            st_size_t const assign_slot_size = ::NS(Buffer_get_slot_size)(
                assign_items_arg.ptrCObjectBuffer() );

            st_size_t const dest_slot_size = ::NS(Buffer_get_slot_size)(
                dest_buffer_arg.ptrCObjectBuffer() );

            st_size_t const src_slot_size = ::NS(Buffer_get_slot_size)(
                dest_buffer_arg.ptrCObjectBuffer() );

            if( ( assign_slot_size > st_size_t{ 0 } ) &&
                ( dest_slot_size > st_size_t{ 0 } ) &&
                ( src_slot_size > st_size_t{ 0 } ) &&
                ( this->kernelNumArgs( kernel_id ) >= st_size_t{ 8 } ) )
            {
                this->assignKernelArgument(
                    kernel_id, st_size_t{ 0 }, assign_items_arg );

                this->assignKernelArgumentValue(
                    kernel_id, size_t{ 1 }, assign_slot_size );

                this->assignKernelArgument(
                    kernel_id, size_t{ 2 }, dest_buffer_arg );

                this->assignKernelArgumentValue(
                    kernel_id, size_t{ 3 }, dest_slot_size );

                this->assignKernelArgumentValue(
                    kernel_id, size_t{ 4 }, dest_buffer_id );

                this->assignKernelArgument(
                    kernel_id, size_t{ 5 }, src_buffer_arg );

                this->assignKernelArgumentValue(
                    kernel_id, size_t{ 6 }, src_slot_size );

                this->assignKernelArgumentValue(
                    kernel_id, size_t{ 7 }, src_buffer_id );

                if( this->runKernel( kernel_id, num_items ) )
                {
                    status = st::ARCH_STATUS_SUCCESS;
                }
            }
        }

        return status;
    }

    /* --------------------------------------------------------------------- */

    st_size_t ctx_t::selected_particle_set() const SIXTRL_NOEXCEPT
    {
        return this->m_pset_index;
    }

    st_size_t ctx_t::num_particles_in_selected_set() const SIXTRL_NOEXCEPT
    {
        return this->m_num_particles_in_pset;
    }

    /* --------------------------------------------------------------------- */

    bool ctx_t::use_optimized_tracking() const SIXTRL_NOEXCEPT
    {
        return this->m_use_optimized_tracking;
    }

    void ctx_t::enable_optimized_tracking()
    {
        if( ( !this->use_optimized_tracking() ) &&
            ( !this->hasSelectedNode() ) )
        {
            this->clear();
            this->m_use_optimized_tracking = true;
            this->doInitDefaultPrograms();
            this->doInitDefaultKernels();
        }
    }

    void ctx_t::disable_optimized_tracking()
    {
        if( ( this->use_optimized_tracking() ) &&
            ( !this->hasSelectedNode() ) )
        {
            this->clear();

            this->m_use_optimized_tracking = false;
            this->doInitDefaultPrograms();
            this->doInitDefaultKernels();
        }
    }

    /* --------------------------------------------------------------------- */

    bool ctx_t::is_beam_beam_tracking_enabled() const
    {
        return (
            ( ( !this->has_feature_flag( "SIXTRL_TRACK_BEAMBEAM4D" ) ) ||
              (  this->feature_flag_str( "SIXTRL_TRACK_BEAMBEAM4D" ).compare(
                  SIXTRL_TRACK_MAP_ENABLED_STR ) == 0 ) ) &&
            ( ( !this->has_feature_flag( "SIXTRL_TRACK_BEAMBEAM6D" ) ) ||
              (  this->feature_flag_str( "SIXTRL_TRACK_BEAMBEAM6D" ).compare(
                  SIXTRL_TRACK_MAP_ENABLED_STR ) == 0 ) ) );
    }

    void ctx_t::enable_beam_beam_tracking()
    {
        if( !this->hasSelectedNode() )
        {
            this->set_feature_flag(
                "SIXTRL_TRACK_BEAMBEAM4D", SIXTRL_TRACK_MAP_ENABLED_STR );
            this->set_feature_flag(
                "SIXTRL_TRACK_BEAMBEAM6D", SIXTRL_TRACK_MAP_ENABLED_STR );

            this->reinit_default_programs();
        }
    }

    void ctx_t::skip_beam_beam_tracking()
    {
        if( !this->hasSelectedNode() )
        {
            this->set_feature_flag(
                "SIXTRL_TRACK_BEAMBEAM4D", SIXTRL_TRACK_MAP_SKIP_STR );

            this->set_feature_flag(
                "SIXTRL_TRACK_BEAMBEAM6D", SIXTRL_TRACK_MAP_SKIP_STR );

            this->reinit_default_programs();
        }
    }

    /* --------------------------------------------------------------------- */

    void ctx_t::disable_beam_beam_tracking()
    {
        if( !this->hasSelectedNode() )
        {
            this->set_feature_flag(
                "SIXTRL_TRACK_BEAMBEAM4D", SIXTRL_TRACK_MAP_DISABLED_STR );

            this->set_feature_flag(
                "SIXTRL_TRACK_BEAMBEAM6D", SIXTRL_TRACK_MAP_DISABLED_STR );

            this->reinit_default_programs();
        }
    }

    /* --------------------------------------------------------------------- */

    bool ctx_t::doSelectNode( ctx_t::size_type node_index )
    {
        /* WARNING: Workaround for AMD Heisenbug */
        if( ( this->use_optimized_tracking() ) &&
            ( this->isAvailableNodeAMDPlatform( node_index ) ) )
        {
            this->disable_optimized_tracking();
        }

        return base_ctx_t::doSelectNode( node_index );
    }

    st_status_t ctx_t::doInitDefaultFeatureFlags()
    {
        st_status_t status = base_ctx_t::doInitDefaultFeatureFlags();

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doInitDefaultFeatureFlagsPrivImpl();
        }

        return status;
    }

    st_status_t ctx_t::doInitDefaultFeatureFlagsPrivImpl()
    {
        #if defined( SIXTRL_TRACK_BEAMBEAM4D )
        this->set_feature_flag( "SIXTRL_TRACK_BEAMBEAM4D",
            SIXTRL_TRACK_BEAMBEAM4D_STR );
        #endif /* SIXTRL_TRACK_BEAMBEAM4D */

        #if defined( SIXTRL_TRACK_BEAMBEAM6D )
        this->set_feature_flag( "SIXTRL_TRACK_BEAMBEAM6D",
            SIXTRL_TRACK_BEAMBEAM6D_STR );
        #endif /* SIXTRL_TRACK_BEAMBEAM6D */

        #if defined( SIXTRL_TRACK_SC_COASTING )
        this->set_feature_flag( "SIXTRL_TRACK_SC_COASTING",
            SIXTRL_TRACK_SC_COASTING_STR );
        #endif /* SIXTRL_TRACK_SC_COASTING */

        #if defined( SIXTRL_TRACK_SC_BUNCHED )
        this->set_feature_flag( "SIXTRL_TRACK_SC_BUNCHED",
            SIXTRL_TRACK_SC_BUNCHED_STR );
        #endif /* SIXTRL_TRACK_SC_BUNCHED */

        #if defined( SIXTRL_TRACK_TRICUB )
        this->set_feature_flag( "SIXTRL_TRACK_TRICUB",
            SIXTRL_TRACK_TRICUB_STR );
        #endif /* SIXTRL_TRACK_TRICUB */

        #if defined( SIXTRL_APERTURE_CHECK_AT_DRIFT )
        this->set_feature_flag( "SIXTRL_APERTURE_CHECK_AT_DRIFT",
            SIXTRL_APERTURE_CHECK_AT_DRIFT_STR );
        #endif /* defined( SIXTRL_APERTURE_CHECK_AT_DRIFT ) */

        #if defined( SIXTRL_APERTURE_CHECK_MIN_DRIFT_LENGTH )
        this->set_feature_flag( "SIXTRL_APERTURE_CHECK_MIN_DRIFT_LENGTH",
            SIXTRL_APERTURE_CHECK_MIN_DRIFT_LENGTH_STR );
        #endif /* SIXTRL_APERTURE_CHECK_MIN_DRIFT_LENGTH */

        #if defined( SIXTRL_APERTURE_X_LIMIT )
        this->set_feature_flag( "SIXTRL_APERTURE_X_LIMIT",
            SIXTRL_APERTURE_X_LIMIT_STR );
        #endif /* SIXTRL_APERTURE_X_LIMIT */

        #if defined( SIXTRL_APERTURE_Y_LIMIT )
        this->set_feature_flag( "SIXTRL_APERTURE_Y_LIMIT",
            SIXTRL_APERTURE_Y_LIMIT_STR );
        #endif /* SIXTRL_APERTURE_Y_LIMIT */

        return st::ARCH_STATUS_SUCCESS;
    }

    bool ctx_t::doInitDefaultPrograms()
    {
        return ( ( ClContextBase::doInitDefaultPrograms() ) &&
                 ( this->doInitDefaultProgramsPrivImpl() ) );
    }

    bool ctx_t::doInitDefaultKernels()
    {
        return ( ( ClContextBase::doInitDefaultKernels() ) &&
                 ( this->doInitDefaultKernelsPrivImpl() ) );
    }

    st_status_t ctx_t::doAssignStatusFlagsArgPrivImpl(
        ctx_t::cl_buffer_t& SIXTRL_RESTRICT_REF status_flags_arg )
    {
        using size_t = st_size_t;
        using st_kernel_id_t = st_kernel_id_t;

        st_status_t status = st::ARCH_STATUS_SUCCESS;
        if( !this->debugMode() ) return status;

        constexpr size_t NUM_KERNELS = size_t{ 6 };

        st_kernel_id_t kernel_ids[ NUM_KERNELS ];
        std::fill( &kernel_ids[ 0 ], &kernel_ids[ NUM_KERNELS ],
                   st::ARCH_ILLEGAL_KERNEL_ID );

        kernel_ids[ 0 ] = this->track_until_kernel_id();
        kernel_ids[ 1 ] = this->track_line_kernel_id();
        kernel_ids[ 2 ] = this->track_elem_by_elem_kernel_id();
        kernel_ids[ 3 ] = this->assign_beam_monitor_output_kernel_id();
        kernel_ids[ 4 ] = this->clear_beam_monitor_output_kernel_id();
        kernel_ids[ 5 ] = this->assign_elem_by_elem_output_kernel_id();

        size_t const min_num_kernel_args[ NUM_KERNELS ] =
        {
            ctx_t::MIN_NUM_TRACK_UNTIL_ARGS,
            ctx_t::MIN_NUM_TRACK_LINE_ARGS,
            ctx_t::MIN_NUM_TRACK_ELEM_ARGS,
            ctx_t::MIN_NUM_ASSIGN_BE_MON_ARGS,
            ctx_t::MIN_NUM_CLEAR_BE_MON_ARGS,
            ctx_t::MIN_NUM_ASSIGN_ELEM_ARGS
        };

        size_t const status_flags_arg_idx[ NUM_KERNELS ] =
        {
            size_t{ 5 }, // track_until
            size_t{ 7 }, // track_line
            size_t{ 7 }, // track_elem_elem
            size_t{ 5 }, // assign_be_mon
            size_t{ 2 }, // clear_be_mon
            size_t{ 5 }  // assign_elem_by_elem
        };

        for( size_t ii = size_t{ 0 } ; ii < NUM_KERNELS ; ++ii )
        {
            st_kernel_id_t const kernel_id = kernel_ids[ ii ];
            size_t const min_num_args = min_num_kernel_args[ ii ];

            if( kernel_id == st::ARCH_ILLEGAL_KERNEL_ID ) continue;
            if( min_num_args == size_t{ 0 } ) continue;

            size_t const num_args = this->kernelNumArgs( kernel_id );
            size_t const arg_idx = status_flags_arg_idx[ ii ];

            if( ( num_args <= arg_idx ) || ( num_args < min_num_args ) )
            {
                status |= st::ARCH_STATUS_GENERAL_FAILURE;
                continue;
            }

            this->assignKernelArgumentClBuffer(
                kernel_id, arg_idx, status_flags_arg );
        }

        return status;
    }

    st_status_t ctx_t::doAssignStatusFlagsArg(
        ctx_t::cl_buffer_t& SIXTRL_RESTRICT_REF status_flags_arg )
    {
        st_status_t status =
            base_ctx_t::doAssignStatusFlagsArg( status_flags_arg );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doAssignStatusFlagsArgPrivImpl( status_flags_arg );
        }

        return status;
    }

    st_status_t ctx_t::doAssignSlotSizeArgPrivImpl( st_size_t const slot_size )
    {
        using size_t = st_size_t;
        using st_kernel_id_t = st_kernel_id_t;

        st_status_t status = st::ARCH_STATUS_SUCCESS;
        constexpr size_t NUM_KERNELS = size_t{ 7 };

        st_kernel_id_t kernel_ids[ NUM_KERNELS ];
        std::fill( &kernel_ids[ 0 ], &kernel_ids[ NUM_KERNELS ],
                   st::ARCH_ILLEGAL_KERNEL_ID );

        kernel_ids[ 0 ] = this->track_until_kernel_id();
        kernel_ids[ 1 ] = this->track_line_kernel_id();
        kernel_ids[ 2 ] = this->track_elem_by_elem_kernel_id();
        kernel_ids[ 3 ] = this->assign_beam_monitor_output_kernel_id();
        kernel_ids[ 4 ] = this->clear_beam_monitor_output_kernel_id();
        kernel_ids[ 5 ] = this->assign_elem_by_elem_output_kernel_id();
        kernel_ids[ 6 ] = this->fetch_particles_addr_kernel_id();

        size_t const min_num_kernel_args[ NUM_KERNELS ] =
        {
            ctx_t::MIN_NUM_TRACK_UNTIL_ARGS,
            ctx_t::MIN_NUM_TRACK_LINE_ARGS,
            ctx_t::MIN_NUM_TRACK_ELEM_ARGS,
            ctx_t::MIN_NUM_ASSIGN_BE_MON_ARGS,
            ctx_t::MIN_NUM_CLEAR_BE_MON_ARGS,
            ctx_t::MIN_NUM_ASSIGN_ELEM_ARGS,
            ctx_t::MIN_NUM_FETCH_PARTICLES_ADDR_ARGS
        };

        size_t const slot_size_arg_idx[ NUM_KERNELS ] =
        {
            size_t{ 4 }, // track_until
            size_t{ 6 }, // track_line
            size_t{ 6 }, // track_elem_elem
            size_t{ 4 }, // assign_be_mon
            size_t{ 1 }, // clear_be_mon
            size_t{ 4 }, // assign_elem_by_elem
            size_t{ 2 }  // fetch_particles_addr
        };

        uint64_t const slot_size_arg = static_cast< uint64_t >( slot_size );

        for( size_t ii = size_t{ 0 } ; ii < NUM_KERNELS ; ++ii )
        {
            st_kernel_id_t const kernel_id = kernel_ids[ ii ];
            size_t const min_num_args = min_num_kernel_args[ ii ];

            if( kernel_id == st::ARCH_ILLEGAL_KERNEL_ID ) continue;
            if( min_num_args == size_t{ 0 } ) continue;

            size_t const num_args = this->kernelNumArgs( kernel_id );
            size_t const arg_idx = slot_size_arg_idx[ ii ];

            if( ( num_args <= arg_idx ) || ( num_args < min_num_args ) )
            {
                status |= st::ARCH_STATUS_GENERAL_FAILURE;
                continue;
            }

            this->assignKernelArgumentValue( kernel_id, arg_idx, slot_size_arg );
        }

        return status;
    }

    st_status_t ctx_t::doAssignElemByElemConfigIndexArgPrivImpl(
        st_size_t const elem_by_elem_config_index )
    {
        st_status_t status = st::ARCH_STATUS_SUCCESS;
        constexpr st_size_t NUM_KERNELS = st_size_t{ 2 };

        st_kernel_id_t kernel_ids[ NUM_KERNELS ] =
        {
            st::ARCH_ILLEGAL_KERNEL_ID, st::ARCH_ILLEGAL_KERNEL_ID
        };

        kernel_ids[ 0 ] = this->track_elem_by_elem_kernel_id();
        kernel_ids[ 1 ] = this->assign_elem_by_elem_output_kernel_id();

        st_size_t const min_num_kernel_args[ NUM_KERNELS ] =
        {
            ctx_t::MIN_NUM_TRACK_ELEM_ARGS,
            ctx_t::MIN_NUM_ASSIGN_ELEM_ARGS
        };

        st_size_t const elem_by_elem_config_index_arg_idx[ NUM_KERNELS ] =
        {
            st_size_t{ 4 }, // track_elem_by_elem_kernel
            st_size_t{ 1 }, // assign_elem_by_elem_output_kernel
        };

        uint64_t const conf_idx_arg =
            static_cast< uint64_t >( elem_by_elem_config_index );

        for( st_size_t ii = st_size_t{ 0 } ; ii < NUM_KERNELS ; ++ii )
        {
            st_kernel_id_t const kernel_id = kernel_ids[ ii ];
            st_size_t const min_num_args = min_num_kernel_args[ ii ];

            if( kernel_id == st::ARCH_ILLEGAL_KERNEL_ID ) continue;
            if( min_num_args == st_size_t{ 0 } ) continue;

            st_size_t const num_args = this->kernelNumArgs( kernel_id );
            st_size_t const arg_idx  = elem_by_elem_config_index_arg_idx[ ii ];

            if( ( num_args <= arg_idx ) || ( num_args < min_num_args ) )
            {
                status |= st::ARCH_STATUS_GENERAL_FAILURE;
                continue;
            }

            this->assignKernelArgumentValue(
                kernel_id, arg_idx, conf_idx_arg );
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            this->m_elem_by_elem_config_index = elem_by_elem_config_index;
        }

        return status;
    }

    st_status_t ctx_t::doAssignSlotSizeArg( st_size_t const slot_size )
    {
        st_status_t status = base_ctx_t::doAssignSlotSizeArg( slot_size );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doAssignSlotSizeArgPrivImpl( slot_size );
        }

        return status;
    }

    st_status_t ctx_t::doAssignElemByElemConfigIndexArg(
        st_size_t const elem_by_elem_config_index )
    {
        return this->doAssignElemByElemConfigIndexArgPrivImpl(
            elem_by_elem_config_index );
    }

    bool ctx_t::doInitDefaultProgramsPrivImpl()
    {
        bool success = false;

        std::string path_to_particles_track_prog =
            this->default_path_to_kernel_dir_str();

        std::string path_to_particles_track_opt_prog =
            this->default_path_to_kernel_dir_str();

        std::string path_to_assign_be_mon_out_buffer_prog =
            this->default_path_to_kernel_dir_str();

        std::string path_to_assign_elem_out_buffer_prog =
            this->default_path_to_kernel_dir_str();

        std::string path_to_fetch_particle_addr_prog =
            this->default_path_to_kernel_dir_str();

        std::string path_to_assign_addr_prog =
            this->default_path_to_kernel_dir_str();

        if( !this->debugMode() )
        {
            path_to_particles_track_prog += "track_particles.cl";

            path_to_particles_track_opt_prog +=
                "track_particles_optimized_priv_particles.cl";

            path_to_assign_be_mon_out_buffer_prog +=
                "be_monitors_assign_out_buffer.cl";

            path_to_assign_elem_out_buffer_prog +=
                "elem_by_elem_assign_out_buffer.cl";
        }
        else
        {
            path_to_particles_track_prog += "track_particles_debug.cl";

            path_to_particles_track_opt_prog +=
                "track_particles_optimized_priv_particles_debug.cl";

            path_to_assign_be_mon_out_buffer_prog +=
                "be_monitors_assign_out_buffer_debug.cl";

            path_to_assign_elem_out_buffer_prog +=
                "elem_by_elem_assign_out_buffer_debug.cl";
        }

        path_to_fetch_particle_addr_prog += "fetch_particles_addr.cl";
        path_to_assign_addr_prog += "assign_address_item.cl";

        std::ostringstream compile_options;

        if( ( this->defaultCompileOptions() != nullptr ) &&
            ( std::strlen( this->defaultCompileOptions() ) > 0u ) )
        {
            compile_options << this->defaultCompileOptions() << " ";
        }

        compile_options << this->feature_flag_repr( "_GPUCODE" ) << " "
            << this->feature_flag_repr( "SIXTRL_BUFFER_ARGPTR_DEC" ) << " "
            << this->feature_flag_repr( "SIXTRL_BUFFER_DATAPTR_DEC" ) << " "
            << "-I" << NS(PATH_TO_SIXTRL_INCLUDE_DIR);

        std::string const common_compile_options( compile_options.str() );

        compile_options.str( "" );
        std::vector< std::string > const OPTIONAL_FEATURES =
        {
            "SIXTRL_TRACK_BEAMBEAM4D", "SIXTRL_TRACK_BEAMBEAM6D",
            "SIXTRL_TRACK_SPACECHARGE", "SIXTRL_TRACK_TRICUB",
            "SIXTRL_APERTURE_CHECK_AT_DRIFT",
            "SIXTRL_APERTURE_CHECK_MIN_DRIFT_LENGTH",
            "SIXTRL_APERTURE_X_LIMIT", "SIXTRL_APERTURE_Y_LIMIT"
        };

        for( auto const& fkey : OPTIONAL_FEATURES )
        {
            if( this->has_feature_flag( fkey ) )
            {
                compile_options << this->feature_flag_repr( fkey ) << " ";
            }
        }

        std::string const feature_compile_options( compile_options.str() );
        compile_options.str( "" );

        compile_options << common_compile_options << " "
                        << feature_compile_options << " "
                        << "-DSIXTRL_PARTICLE_ARGPTR_DEC=__private "
                        << "-DSIXTRL_PARTICLE_DATAPTR_DEC=__private ";

        std::string const track_optimized_compile_options( compile_options.str() );
        compile_options.str( "" );

        compile_options << common_compile_options  << " "
                        << feature_compile_options << " "
                        << "-DSIXTRL_PARTICLE_ARGPTR_DEC=__global "
                        << "-DSIXTRL_PARTICLE_DATAPTR_DEC=__global ";

        std::string const track_compile_options( compile_options.str() );
        compile_options.str( "" );

        compile_options << common_compile_options << " "
            << "-DSIXTRL_PARTICLE_ARGPTR_DEC=__global "
            << "-DSIXTRL_PARTICLE_DATAPTR_DEC=__global ";

        std::string const fetch_paddrs_compile_options( compile_options.str() );
        std::string const assign_out_buffer_compile_options(
            fetch_paddrs_compile_options );


        st_program_id_t const track_program_id = this->addProgramFile(
            path_to_particles_track_prog, track_compile_options,
            ctx_t::PROGRAM_PATH_ABSOLUTE );

        st_program_id_t const track_optimized_program_id = this->addProgramFile(
            path_to_particles_track_opt_prog, track_optimized_compile_options,
            ctx_t::PROGRAM_PATH_ABSOLUTE );

        st_program_id_t const assign_be_mon_out_buffer_program_id =
        this->addProgramFile( path_to_assign_be_mon_out_buffer_prog,
            assign_out_buffer_compile_options, ctx_t::PROGRAM_PATH_ABSOLUTE );

        st_program_id_t const assign_elem_by_elem_out_buffer_program_id =
        this->addProgramFile( path_to_assign_elem_out_buffer_prog,
            assign_out_buffer_compile_options, ctx_t::PROGRAM_PATH_ABSOLUTE );

        st_program_id_t const fetch_particles_addr_program_id =
        this->addProgramFile( path_to_fetch_particle_addr_prog,
            fetch_paddrs_compile_options, ctx_t::PROGRAM_PATH_ABSOLUTE );

        st_program_id_t const assign_addr_program_id =
            this->addProgramFile( path_to_assign_addr_prog,
            assign_out_buffer_compile_options, ctx_t::PROGRAM_PATH_ABSOLUTE );

        if( ( track_program_id != st::ARCH_ILLEGAL_PROGRAM_ID ) &&
            ( track_optimized_program_id  != st::ARCH_ILLEGAL_PROGRAM_ID ) &&
            ( assign_be_mon_out_buffer_program_id != st::ARCH_ILLEGAL_PROGRAM_ID ) &&
            ( assign_elem_by_elem_out_buffer_program_id != st::ARCH_ILLEGAL_PROGRAM_ID ) &&
            ( assign_addr_program_id != st::ARCH_ILLEGAL_PROGRAM_ID ) )
        {
            if( !this->use_optimized_tracking() )
            {
                this->m_track_until_turn_program_id   = track_program_id;
                this->m_track_elem_by_elem_program_id = track_program_id;
                this->m_track_line_program_id         = track_program_id;
            }
            else
            {
                this->m_track_until_turn_program_id   = track_optimized_program_id;
                this->m_track_elem_by_elem_program_id = track_optimized_program_id;
                this->m_track_line_program_id         = track_optimized_program_id;
            }

            this->m_assign_be_mon_out_buffer_program_id =
                assign_be_mon_out_buffer_program_id;

            this->m_clear_be_mon_program_id =
                assign_be_mon_out_buffer_program_id;

            this->m_assign_elem_by_elem_out_buffer_program_id =
                assign_elem_by_elem_out_buffer_program_id;

            this->m_fetch_particles_addr_program_id =
                fetch_particles_addr_program_id;

            this->m_assign_addr_program_id = assign_addr_program_id;
            success = true;
        }

        return success;
    }

    bool ctx_t::doInitDefaultKernelsPrivImpl()
    {
        bool success = false;

        if( this->hasSelectedNode() )
        {
            st_program_id_t const max_program_id = static_cast< st_program_id_t >(
                this->numAvailablePrograms() );

            if( ( this->m_track_until_turn_program_id >= st_program_id_t{ 0 } ) &&
                ( this->m_track_until_turn_program_id <  max_program_id ) )
            {
                std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
                kernel_name += "Track_particles_until_turn";

                if( this->use_optimized_tracking() )
                {
                    kernel_name += "_opt_pp";
                }

                if( this->debugMode() )
                {
                    kernel_name += "_debug";
                }

                kernel_name += "_opencl";

                st_kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_track_until_turn_program_id );

                if( kernel_id >= st_kernel_id_t{ 0 } )
                {
                    success = ( this->set_track_until_kernel_id( kernel_id ) ==
                        st::ARCH_STATUS_SUCCESS );
                }
            }

            if( ( success ) &&
                ( this->m_track_elem_by_elem_program_id >= st_program_id_t{ 0 } ) &&
                ( this->m_track_elem_by_elem_program_id <  max_program_id ) )
            {
                std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
                kernel_name += "Track_particles_elem_by_elem";

                if( this->use_optimized_tracking() )
                {
                    kernel_name += "_opt_pp";
                }

                if( this->debugMode() )
                {
                    kernel_name += "_debug";
                }

                kernel_name += "_opencl";

                st_kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_track_elem_by_elem_program_id );

                if( kernel_id >= st_kernel_id_t{ 0 } )
                {
                    success = ( this->set_track_elem_by_elem_kernel_id(
                        kernel_id ) == st::ARCH_STATUS_SUCCESS );
                }
            }

            if( ( success ) &&
                ( this->m_track_line_program_id >= st_program_id_t{ 0 } ) &&
                ( this->m_track_line_program_id <  max_program_id ) )
            {
                std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
                kernel_name += "Track_particles_line";

                if( this->use_optimized_tracking() )
                {
                    kernel_name += "_opt_pp";
                }

                if( this->debugMode() )
                {
                    kernel_name += "_debug";
                }

                kernel_name += "_opencl";

                st_kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_track_line_program_id );

                if( kernel_id >= st_kernel_id_t{ 0 } )
                {
                    success = ( this->set_track_line_kernel_id( kernel_id ) ==
                        st::ARCH_STATUS_SUCCESS );
                }
            }

            if( ( success ) &&
                ( this->m_assign_be_mon_out_buffer_program_id >= st_program_id_t{ 0 } ) &&
                ( this->m_assign_be_mon_out_buffer_program_id <  max_program_id ) )
            {
                std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
                kernel_name += "BeamMonitor_assign_out_buffer_from_offset";

                if( this->debugMode() )
                {
                    kernel_name += "_debug";
                }

                kernel_name += "_opencl";

                st_kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_assign_be_mon_out_buffer_program_id );

                if( kernel_id >= st_kernel_id_t{ 0 } )
                {
                    success = ( this->set_assign_beam_monitor_output_kernel_id(
                        kernel_id ) == st::ARCH_STATUS_SUCCESS );
                }
            }

            if( ( success ) &&
                ( this->m_clear_be_mon_program_id >= st_program_id_t{ 0 } ) &&
                ( this->m_clear_be_mon_program_id <  max_program_id ) )
            {
                std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
                kernel_name += "BeamMonitor_clear_all_line_obj";

                if( this->debugMode() )
                {
                    kernel_name += "_debug";
                }

                kernel_name += "_opencl";

                st_kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_clear_be_mon_program_id );

                if( kernel_id >= st_kernel_id_t{ 0 } )
                {
                    success = ( this->set_clear_beam_monitor_output_kernel_id(
                        kernel_id ) == st::ARCH_STATUS_SUCCESS );
                }
            }

            if( ( success ) &&
                ( this->m_assign_elem_by_elem_out_buffer_program_id >=
                    st_program_id_t{ 0 } ) &&
                ( this->m_assign_elem_by_elem_out_buffer_program_id <
                    max_program_id ) )
            {
                std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
                kernel_name += "ElemByElem_assign_out_buffer_from_offset";

                if( this->debugMode() )
                {
                    kernel_name += "_debug";
                }

                kernel_name += "_opencl";

                st_kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(),
                    this->m_assign_elem_by_elem_out_buffer_program_id );

                if( kernel_id >= st_kernel_id_t{ 0 } )
                {
                    success = ( this->set_assign_elem_by_elem_output_kernel_id(
                        kernel_id ) == st::ARCH_STATUS_SUCCESS );
                }
            }

            if( ( success ) &&
                ( this->m_fetch_particles_addr_program_id >= st_program_id_t{ 0 } ) &&
                ( this->m_fetch_particles_addr_program_id < max_program_id ) )
            {
                std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
                kernel_name += "Particles_buffer_store_all_addresses";
                kernel_name += "_opencl";

                st_kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_fetch_particles_addr_program_id );

                if( kernel_id >= st_kernel_id_t{ 0 } )
                {
                    success = ( this->set_fetch_particles_addr_kernel_id(
                        kernel_id ) == st::ARCH_STATUS_SUCCESS );
                }
            }

            if( ( success ) &&
                ( this->m_assign_addr_program_id != st::ARCH_ILLEGAL_PROGRAM_ID ) &&
                ( this->m_assign_addr_program_id <  this->numAvailablePrograms() ) )
            {
                std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
                kernel_name += "AssignAddressItem_process_managed_buffer";
                kernel_name += "_opencl";

                st_kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_assign_addr_program_id );

                if( kernel_id != st::ARCH_ILLEGAL_KERNEL_ID )
                {
                    success = ( this->set_assign_addresses_kernel_id(
                        kernel_id ) == st::ARCH_STATUS_SUCCESS );
                }
            }
        }

        return success;
    }
}

#endif /* defined( __cplusplus ) */

/* ========================================================================= */

::NS(ClContext)* NS(ClContext_create)()
{
    return new SIXTRL_CXX_NAMESPACE::ClContext;
}

::NS(ClContext)* NS(ClContext_new)( const char* node_id_str )
{
    return new SIXTRL_CXX_NAMESPACE::ClContext( node_id_str, nullptr );
}

void NS(ClContext_delete)( ::NS(ClContext)* SIXTRL_RESTRICT ctx )
{
    delete ctx;
}

void NS(ClContext_clear)( ::NS(ClContext)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->clear();
}

/* ========================================================================= */

::NS(ctrl_status_t) NS(ClContext_assign_particles_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(ClArgument)* SIXTRL_RESTRICT particles_arg )
{
    return ( ( ctx != nullptr ) && ( particles_arg != nullptr ) )
        ? ctx->assign_particles_arg( *particles_arg )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(ClContext_assign_particle_set_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(buffer_size_t) const particle_set_index,
    ::NS(buffer_size_t) const num_particles_in_selected_set )
{
    return ( ctx != nullptr )
        ? ctx->assign_particle_set_arg(
            particle_set_index, num_particles_in_selected_set )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ctrl_status_t) NS(ClContext_assign_beam_elements_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(ClArgument)* SIXTRL_RESTRICT beam_elem_arg )
{
    return ( ( ctx != nullptr ) && ( beam_elem_arg != nullptr ) )
        ? ctx->assign_beam_elements_arg( *beam_elem_arg )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ctrl_status_t) NS(ClContext_assign_output_buffer_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(ClArgument)* SIXTRL_RESTRICT_REF out_buffer_arg )
{
    return ( ( ctx != nullptr ) && ( out_buffer_arg != nullptr ) )
        ? ctx->assign_output_buffer_arg( *out_buffer_arg )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ctrl_status_t) NS(ClContext_assign_elem_by_elem_config_buffer_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(ClArgument)* SIXTRL_RESTRICT elem_by_elem_config_buffer_arg )
{
    return ( ( ctx != nullptr ) &&
             ( elem_by_elem_config_buffer_arg != nullptr ) )
        ? ctx->assign_elem_by_elem_config_buffer_arg(
            *elem_by_elem_config_buffer_arg )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ctrl_status_t) NS(ClContext_assign_elem_by_elem_config_index_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(buffer_size_t) const elem_by_elem_config_index )
{
    return ( ctx != nullptr )
        ? ctx->assign_elem_by_elem_config_index_arg( elem_by_elem_config_index )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ctrl_status_t) NS(ClContext_assign_slot_size_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx, ::NS(buffer_size_t) const slot_size )
{
    return ( ctx != nullptr )
        ? ctx->assign_slot_size_arg( slot_size )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ctrl_status_t) NS(ClContext_assign_status_flags_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx, cl_mem status_flags_arg )
{
    cl::Buffer temp_buffer( status_flags_arg );

    return ( ctx != nullptr )
        ? ctx->assign_status_flags_arg( temp_buffer )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

/* ========================================================================= */

bool NS(ClContext_has_track_until_kernel)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) && ( ctx->has_track_until_kernel() ) );
}

::NS(arch_kernel_id_t) NS(ClContext_track_until_kernel_id)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->track_until_kernel_id() : ::NS(ARCH_ILLEGAL_KERNEL_ID);
}

::NS(arch_status_t) NS(ClContext_set_track_until_kernel_id)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->set_track_until_kernel_id( kernel_id )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(track_status_t) NS(ClContext_track_until)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(context_num_turns_t) const until_turn )
{
    return ( ctx != nullptr ) ? ctx->track_until( until_turn )
        : SIXTRL_CXX_NAMESPACE::TRACK_STATUS_GENERAL_FAILURE;
}

::NS(track_status_t) NS(ClContext_track_until_for_particle_set)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(context_num_turns_t) const until_turn,
    ::NS(buffer_size_t) const particle_set_index,
    ::NS(buffer_size_t) const num_particles_in_set,
    bool const restore_particle_set_index )
{
    return ( ctx != nullptr )
        ? ctx->track_until( until_turn, particle_set_index,
            num_particles_in_set, restore_particle_set_index )
        : SIXTRL_CXX_NAMESPACE::TRACK_STATUS_GENERAL_FAILURE;
}

/* ------------------------------------------------------------------------- */

bool NS(ClContext_has_track_line_kernel)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) && ( ctx->has_track_line_kernel() ) );
}

::NS(arch_kernel_id_t) NS(ClContext_track_line_kernel_id)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->track_line_kernel_id()
        : SIXTRL_CXX_NAMESPACE::ARCH_ILLEGAL_KERNEL_ID;
}

::NS(arch_status_t) NS(ClContext_set_track_line_kernel_id)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->set_track_line_kernel_id( kernel_id )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(track_status_t) NS(ClContext_track_line)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(buffer_size_t) const line_begin_idx,
    ::NS(buffer_size_t) const line_end_idx, bool const finish_turn )
{
    SIXTRL_ASSERT( ctx != nullptr );
    return ctx->track_line( line_begin_idx, line_end_idx, finish_turn );
}

::NS(track_status_t) NS(ClContext_track_line_for_particle_set)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(buffer_size_t) const line_begin_idx,
    ::NS(buffer_size_t) const line_end_idx, bool const finish_turn,
    ::NS(buffer_size_t) const particle_set_index,
    ::NS(buffer_size_t) const num_particles_in_set,
    bool const restore_particle_set_index )
{
    return ( ctx != nullptr )
        ? ctx->track_line( line_begin_idx, line_end_idx, finish_turn,
            particle_set_index, num_particles_in_set,
            restore_particle_set_index )
        : SIXTRL_CXX_NAMESPACE::TRACK_STATUS_GENERAL_FAILURE;
}

/* ------------------------------------------------------------------------- */

bool NS(ClContext_has_track_elem_by_elem_kernel)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) &&
             ( ctx->has_track_elem_by_elem_kernel() ) );
}

::NS(arch_kernel_id_t) NS(ClContext_track_elem_by_elem_kernel_id)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->track_elem_by_elem_kernel_id()
        : SIXTRL_CXX_NAMESPACE::ARCH_ILLEGAL_KERNEL_ID;
}

::NS(arch_status_t) NS(ClContext_set_track_elem_by_elem_kernel_id)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->set_track_elem_by_elem_kernel_id( kernel_id )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(track_status_t) NS(ClContext_track_elem_by_elem)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(buffer_size_t) const until_turn )
{
    return ( ctx != nullptr ) ? ctx->track_elem_by_elem( until_turn )
        : SIXTRL_CXX_NAMESPACE::TRACK_STATUS_GENERAL_FAILURE;
}

::NS(track_status_t) NS(ClContext_track_elem_by_elem_for_particle_set)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(context_num_turns_t) const until_turn,
    ::NS(buffer_size_t) const particle_set_index,
    ::NS(buffer_size_t) const num_particles_in_set,
    bool const restore_particle_set_index )
{
    return ( ctx != nullptr )
        ? ctx->track_elem_by_elem( until_turn, particle_set_index,
            num_particles_in_set, restore_particle_set_index )
        : SIXTRL_CXX_NAMESPACE::TRACK_STATUS_GENERAL_FAILURE;
}

/* ------------------------------------------------------------------------- */

bool NS(ClContext_has_assign_beam_monitor_output_kernel)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) &&
             ( ctx->has_assign_beam_monitor_output_kernel() ) );
}

::NS(arch_kernel_id_t) NS(ClContext_assign_beam_monitor_output_kernel_id)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->assign_beam_monitor_output_kernel_id()
        : SIXTRL_CXX_NAMESPACE::ARCH_ILLEGAL_KERNEL_ID;
}

::NS(arch_status_t) NS(ClContext_set_assign_beam_monitor_output_kernel_id)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->set_assign_beam_monitor_output_kernel_id( kernel_id )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(ClContext_assign_beam_monitor_output)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(particle_index_t) const min_turn_id,
    ::NS(buffer_size_t) const out_buffer_index_offset )
{
    return ( ctx != nullptr )
        ? ctx->assign_beam_monitor_output(
            min_turn_id, out_buffer_index_offset )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

/* ------------------------------------------------------------------------- */

bool NS(ClContext_has_assign_elem_by_elem_output_kernel)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) &&
             ( ctx->has_assign_elem_by_elem_output_kernel() ) );
}

::NS(arch_kernel_id_t) NS(ClContext_assign_elem_by_elem_output_kernel_id)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->assign_elem_by_elem_output_kernel_id()
        : SIXTRL_CXX_NAMESPACE::ARCH_ILLEGAL_KERNEL_ID;
}

::NS(arch_status_t) NS(ClContext_set_assign_elem_by_elem_output_kernel_id)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->set_assign_elem_by_elem_output_kernel_id( kernel_id )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(ClContext_assign_elem_by_elem_output)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(buffer_size_t) const out_buffer_index_offset )
{
    return ( ctx != nullptr )
        ? ctx->assign_elem_by_elem_output( out_buffer_index_offset )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

/* ------------------------------------------------------------------------- */

bool NS(ClContext_has_clear_beam_monitor_output_kernel)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) &&
             ( ctx->has_clear_beam_monitor_output_kernel() ) );
}

::NS(arch_kernel_id_t) NS(ClContext_clear_beam_monitor_output_kernel_id)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return (ctx != nullptr )
        ? ctx->clear_beam_monitor_output_kernel_id()
        : SIXTRL_CXX_NAMESPACE::ARCH_ILLEGAL_KERNEL_ID;
}

::NS(arch_status_t) NS(ClContext_set_clear_beam_monitor_output_kernel_id)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->set_clear_beam_monitor_output_kernel_id( kernel_id )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ctrl_status_t) NS(ClContext_clear_beam_monitor_output)(
    ::NS(ClContext)*  SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->clear_beam_monitor_output()
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

/* ========================================================================= */

bool NS(ClContext_has_assign_addresses_kernel)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) && ( ctx->has_assign_addresses_kernel() ) );
}

::NS(ctrl_kernel_id_t) NS(ClContext_assign_addresses_kernel_id)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->assign_addresses_kernel_id()
        : SIXTRL_CXX_NAMESPACE::ARCH_ILLEGAL_KERNEL_ID;
}

::NS(arch_status_t) NS(ClContext_set_assign_addresses_kernel_id)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(ctrl_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->set_assign_addresses_kernel_id( kernel_id )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

NS(arch_status_t) NS(ClContext_assign_addresses)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(ClArgument)* SIXTRL_RESTRICT assign_items_arg,
    ::NS(ClArgument)* SIXTRL_RESTRICT dest_buffer_arg,
    ::NS(buffer_size_t) const dest_buffer_id,
    ::NS(ClArgument)* SIXTRL_RESTRICT src_buffer_arg,
    ::NS(buffer_size_t) const src_buffer_id )
{
    return ( ( ctx != nullptr ) && ( assign_items_arg != nullptr ) &&
             ( dest_buffer_arg != nullptr ) && ( src_buffer_arg != nullptr ) )
        ? ctx->assign_addresses( *assign_items_arg,
            *dest_buffer_arg, dest_buffer_id, *src_buffer_arg, src_buffer_id )
        : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
}

/* ========================================================================= */

::NS(buffer_size_t) NS(ClContext_selected_particle_set)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->selected_particle_set() : ::NS(buffer_size_t){ 0 };
}

::NS(buffer_size_t) NS(ClContext_num_particles_in_selected_set)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->num_particles_in_selected_set() : ::NS(buffer_size_t){ 0 };
}

/* ------------------------------------------------------------------------- */

bool NS(ClContext_uses_optimized_tracking)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) && ( ctx->use_optimized_tracking() ) );
}

void NS(ClContext_enable_optimized_tracking)(
    NS(ClContext)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->enable_optimized_tracking();
}

void NS(ClContext_disable_optimized_tracking)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->disable_optimized_tracking();
}

/* ------------------------------------------------------------------------- */

bool NS(ClContext_is_beam_beam_tracking_enabled)(
    const ::NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) && ( ctx->is_beam_beam_tracking_enabled() ) );
}

void NS(ClContext_enable_beam_beam_tracking)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->enable_beam_beam_tracking();
}

void NS(ClContext_disable_beam_beam_tracking)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->disable_beam_beam_tracking();
}

void NS(ClContext_skip_beam_beam_tracking)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->skip_beam_beam_tracking();
}

#endif /* !defined( __CUDACC__ ) */
