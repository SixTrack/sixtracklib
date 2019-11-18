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
#include "sixtracklib/opencl/cl.h"

#if defined( __cplusplus )

namespace st = SIXTRL_CXX_NAMESPACE;
namespace SIXTRL_CXX_NAMESPACE
{
    using _base_t       = st::ClContextBase;
    using _this_t       = st::ClContext;
    using _size_t       = _this_t::size_type;
    using _kernel_id_t  = _this_t::kernel_id_t;
    using _program_id_t = _this_t::program_id_t;
    using _status_t     = _this_t::status_t;

    constexpr _size_t _this_t::MIN_NUM_TRACK_UNTIL_ARGS;
    constexpr _size_t _this_t::MIN_NUM_TRACK_LINE_ARGS;
    constexpr _size_t _this_t::MIN_NUM_TRACK_ELEM_ARGS;
    constexpr _size_t _this_t::MIN_NUM_ASSIGN_BE_MON_ARGS;
    constexpr _size_t _this_t::MIN_NUM_CLEAR_BE_MON_ARGS;
    constexpr _size_t _this_t::MIN_NUM_ASSIGN_ELEM_ARGS;

    ClContext::ClContext( const char *const SIXTRL_RESTRICT config_str ) :
        _base_t( config_str ),
        m_num_particles_in_pset( _size_t{ 0 } ), m_pset_index( _size_t{ 0 } ),
        m_track_until_turn_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_line_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_elem_by_elem_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_be_mon_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_clear_be_mon_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_track_line_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_elem_by_elem_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_be_mon_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_clear_be_mon_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_use_optimized_tracking( false ), m_enable_beam_beam( true )
    {
        this->doInitDefaultProgramsPrivImpl();
    }

    ClContext::ClContext( _size_t const node_index,
                          const char *const SIXTRL_RESTRICT config_str ) :
        _base_t( config_str ),
        m_num_particles_in_pset( _size_t{ 0 } ), m_pset_index( _size_t{ 0 } ),
        m_track_until_turn_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_line_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_elem_by_elem_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_be_mon_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_clear_be_mon_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_track_line_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_elem_by_elem_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_be_mon_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_clear_be_mon_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_use_optimized_tracking( false ), m_enable_beam_beam( true )
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

        this->doInitDefaultProgramsPrivImpl();

        if( ( node_index < this->numAvailableNodes() ) &&
            ( ClContextBase::doSelectNode( node_index ) ) )
        {
            ClContextBase::doInitDefaultKernels();
            this->doInitDefaultKernelsPrivImpl();
            this->doAssignSlotSizeArgPrivImpl( st::BUFFER_DEFAULT_SLOT_SIZE );
            this->doAssignStatusFlagsArgPrivImpl(
                this->internalStatusFlagsBuffer() );
        }
    }

    ClContext::ClContext( ClContext::node_id_t const node_id,
                          const char *const SIXTRL_RESTRICT config_str ) :
        _base_t( config_str ),
        m_num_particles_in_pset( _size_t{ 0 } ), m_pset_index( _size_t{ 0 } ),
        m_track_until_turn_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_line_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_elem_by_elem_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_be_mon_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_clear_be_mon_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_track_line_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_elem_by_elem_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_be_mon_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_clear_be_mon_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_use_optimized_tracking( false ), m_enable_beam_beam( true )
    {
        this->doInitDefaultProgramsPrivImpl();

        _size_t  const node_index = this->findAvailableNodesIndex(
            NS(ComputeNodeId_get_platform_id)( &node_id ),
            NS(ComputeNodeId_get_device_id)( &node_id ) );

        /* WARNING: Workaround for AMD Heisenbug */
        if( !this->isAvailableNodeAMDPlatform( node_index ) )
        {
            this->m_use_optimized_tracking = true;
        }

        if( ( node_index < this->numAvailableNodes() ) &&
            ( _base_t::doSelectNode( node_index ) ) )
        {
            _base_t::doInitDefaultKernels();
            this->doInitDefaultKernelsPrivImpl();
            this->doAssignSlotSizeArgPrivImpl( st::BUFFER_DEFAULT_SLOT_SIZE );
            this->doAssignStatusFlagsArgPrivImpl(
                this->internalStatusFlagsBuffer() );
        }
    }

    ClContext::ClContext( char const* node_id_str,
                          const char *const SIXTRL_RESTRICT config_str ) :
        _base_t( config_str ),
        m_num_particles_in_pset( _size_t{ 0 } ), m_pset_index( _size_t{ 0 } ),
        m_track_until_turn_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_line_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_elem_by_elem_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_be_mon_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_clear_be_mon_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_track_line_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_elem_by_elem_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_be_mon_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_clear_be_mon_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_use_optimized_tracking( false ), m_enable_beam_beam( true )
    {
        _size_t  node_index = this->findAvailableNodesIndex( node_id_str );

        /* WARNING: Workaround for AMD Heisenbug */
        if( !this->isAvailableNodeAMDPlatform( node_index ) )
        {
            this->m_use_optimized_tracking = true;
        }

        this->doInitDefaultProgramsPrivImpl();

        if( node_index >= this->numAvailableNodes() )
        {
            _this_t::node_id_t const default_node_id = this->defaultNodeId();

            _this_t::platform_id_t const platform_index =
                NS(ComputeNodeId_get_platform_id)( &default_node_id );

            _this_t::device_id_t const device_index =
                NS(ComputeNodeId_get_device_id)( &default_node_id );

            node_index = this->findAvailableNodesIndex(
                platform_index, device_index );
        }

        if( ( node_index < this->numAvailableNodes() ) &&
            ( _base_t::doSelectNode( node_index ) ) )
        {
            _base_t::doInitDefaultKernels();
            this->doInitDefaultKernelsPrivImpl();
            this->doAssignSlotSizeArgPrivImpl( st::BUFFER_DEFAULT_SLOT_SIZE );
            this->doAssignStatusFlagsArgPrivImpl(
                this->internalStatusFlagsBuffer() );
        }
    }

    ClContext::ClContext(
        ClContext::platform_id_t const platform_idx,
        ClContext::device_id_t const device_idx,
        const char *const SIXTRL_RESTRICT config_str ) :
        _base_t( config_str ),
        m_num_particles_in_pset( _size_t{ 0 } ), m_pset_index( _size_t{ 0 } ),
        m_track_until_turn_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_line_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_elem_by_elem_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_assign_be_mon_out_buffer_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_clear_be_mon_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_track_elem_by_elem_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_track_line_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_elem_by_elem_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_assign_be_mon_out_buffer_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_clear_be_mon_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_use_optimized_tracking( false ), m_enable_beam_beam( true )
    {
        _size_t const node_index =
            this->findAvailableNodesIndex( platform_idx, device_idx );

        /* WARNING: Workaround for AMD Heisenbug */
        if( !this->isAvailableNodeAMDPlatform( node_index ) )
        {
            this->m_use_optimized_tracking = true;
        }

        this->doInitDefaultProgramsPrivImpl();

        if( ( node_index < this->numAvailableNodes() ) &&
            ( _base_t::doSelectNode( node_index ) ) )
        {
            _base_t::doInitDefaultKernels();
            this->doInitDefaultKernelsPrivImpl();
            this->doAssignSlotSizeArgPrivImpl( st::BUFFER_DEFAULT_SLOT_SIZE );
            this->doAssignStatusFlagsArgPrivImpl(
                this->internalStatusFlagsBuffer() );
        }
    }

    ClContext::~ClContext() SIXTRL_NOEXCEPT {}

    _status_t ClContext::assign_particles_arg(
        ClArgument& SIXTRL_RESTRICT_REF particles_arg )
    {
        _status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( !particles_arg.usesCObjectBuffer() ) ||
            (  particles_arg.ptrCObjectBuffer() == nullptr ) )
        {
            return status;
        }

        constexpr _size_t NUM_KERNELS = _size_t{ 3 };
        status = st::ARCH_STATUS_SUCCESS;

        _kernel_id_t kernel_ids[ NUM_KERNELS ] =
        {
            st::ARCH_ILLEGAL_KERNEL_ID, st::ARCH_ILLEGAL_KERNEL_ID,
            st::ARCH_ILLEGAL_KERNEL_ID
        };

        kernel_ids[ 0 ] = this->track_until_kernel_id();
        kernel_ids[ 1 ] = this->track_line_kernel_id();
        kernel_ids[ 2 ] = this->track_elem_by_elem_kernel_id();

        size_t const min_num_kernel_args[ NUM_KERNELS ] =
        {
            _this_t::MIN_NUM_TRACK_UNTIL_ARGS, _this_t::MIN_NUM_TRACK_LINE_ARGS,
            _this_t::MIN_NUM_TRACK_ELEM_ARGS
        };

        _size_t const particles_arg_idx[ NUM_KERNELS ] =
        {
            _size_t{ 0 }, // track_until
            _size_t{ 0 }, // track_line
            _size_t{ 0 }, // track_elem_elem
        };

        for( _size_t ii = _size_t{ 0 } ; ii < NUM_KERNELS ; ++ii )
        {
            _kernel_id_t const kernel_id = kernel_ids[ ii ];
            _size_t const min_num_args = min_num_kernel_args[ ii ];

            if( kernel_id == st::ARCH_ILLEGAL_KERNEL_ID ) continue;
            if( min_num_args == _size_t{ 0 } ) continue;

            _size_t const num_args = this->kernelNumArgs( kernel_id );
            _size_t const arg_idx = particles_arg_idx[ ii ];

            if( ( num_args <= arg_idx ) || ( num_args < min_num_args ) )
            {
                status |= st::ARCH_STATUS_GENERAL_FAILURE;
                continue;
            }

            this->assignKernelArgument( kernel_id, arg_idx, particles_arg );
        }

        return status;
    }

    _status_t ClContext::assign_particle_set_arg(
        _size_t const particle_set_index,
        _size_t const num_particles_in_selected_set )
    {
        _status_t status = st::ARCH_STATUS_SUCCESS;
        constexpr size_t NUM_KERNELS = size_t{ 3 };

        _kernel_id_t kernel_ids[ NUM_KERNELS ] =
        {
            st::ARCH_ILLEGAL_KERNEL_ID, st::ARCH_ILLEGAL_KERNEL_ID,
            st::ARCH_ILLEGAL_KERNEL_ID
        };

        kernel_ids[ 0 ] = this->track_until_kernel_id();
        kernel_ids[ 1 ] = this->track_line_kernel_id();
        kernel_ids[ 2 ] = this->track_elem_by_elem_kernel_id();

        _size_t const min_num_kernel_args[ NUM_KERNELS ] =
        {
            _this_t::MIN_NUM_TRACK_UNTIL_ARGS, _this_t::MIN_NUM_TRACK_LINE_ARGS,
            _this_t::MIN_NUM_TRACK_ELEM_ARGS
        };

        _size_t const pset_arg_idx[ NUM_KERNELS ] =
        {
            _size_t{ 1 }, _size_t{ 1 }, _size_t{ 1 },
        };

        uint64_t const pset_idx_arg =
            static_cast< uint64_t >( particle_set_index );

        for( _size_t ii = _size_t{ 0 } ; ii < NUM_KERNELS ; ++ii )
        {
            _kernel_id_t const kernel_id = kernel_ids[ ii ];
            _size_t const min_num_args = min_num_kernel_args[ ii ];

            if( kernel_id == st::ARCH_ILLEGAL_KERNEL_ID ) continue;
            if( min_num_args == _size_t{ 0 } ) continue;

            _size_t const num_args = this->kernelNumArgs( kernel_id );
            _size_t const arg_idx  = pset_arg_idx[ ii ];

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

    _status_t ClContext::assign_beam_elements_arg(
        ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg )
    {
        using size_t       = _size_t;
        using kernel_id_t  = _kernel_id_t;

        _status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( !beam_elements_arg.usesCObjectBuffer() ) ||
            (  beam_elements_arg.ptrCObjectBuffer() == nullptr ) ||
            ( size_t{ 1 } > ::NS(Buffer_get_num_of_objects)(
                beam_elements_arg.ptrCObjectBuffer() ) ) )
        {
            return status;
        }

        constexpr size_t NUM_KERNELS = size_t{ 5 };

        status = st::ARCH_STATUS_SUCCESS;

        kernel_id_t kernel_ids[ NUM_KERNELS ];
        std::fill( &kernel_ids[ 0 ], &kernel_ids[ NUM_KERNELS ],
                   st::ARCH_ILLEGAL_KERNEL_ID );

        kernel_ids[ 0 ] = this->track_until_kernel_id();
        kernel_ids[ 1 ] = this->track_line_kernel_id();
        kernel_ids[ 2 ] = this->track_elem_by_elem_kernel_id();
        kernel_ids[ 3 ] = this->assign_beam_monitor_output_kernel_id();
        kernel_ids[ 4 ] = this->clear_beam_monitor_output_kernel_id();

        size_t const min_num_kernel_args[ NUM_KERNELS ] =
        {
            _this_t::MIN_NUM_TRACK_UNTIL_ARGS, _this_t::MIN_NUM_TRACK_LINE_ARGS,
            _this_t::MIN_NUM_TRACK_ELEM_ARGS,
            _this_t::MIN_NUM_ASSIGN_BE_MON_ARGS,
            _this_t::MIN_NUM_CLEAR_BE_MON_ARGS
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
            kernel_id_t const kernel_id = kernel_ids[ ii ];
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

    _status_t ClContext::assign_output_buffer_arg(
        ClArgument& SIXTRL_RESTRICT_REF out_buffer_arg )
    {
        using size_t       = _size_t;
        using kernel_id_t  = _kernel_id_t;

        _status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( !out_buffer_arg.usesCObjectBuffer() ) ||
            (  out_buffer_arg.ptrCObjectBuffer() == nullptr ) )
        {
            return status;
        }

        status = st::ARCH_STATUS_SUCCESS;
        constexpr size_t NUM_KERNELS = size_t{ 2 };

        kernel_id_t kernel_ids[ NUM_KERNELS ] =
        {
            st::ARCH_ILLEGAL_KERNEL_ID, st::ARCH_ILLEGAL_KERNEL_ID
        };

        kernel_ids[ 0 ] = this->assign_beam_monitor_output_kernel_id();
        kernel_ids[ 1 ] = this->assign_elem_by_elem_output_kernel_id();

        size_t const min_num_kernel_args[ NUM_KERNELS ] =
        {
            _this_t::MIN_NUM_ASSIGN_BE_MON_ARGS,
            _this_t::MIN_NUM_ASSIGN_ELEM_ARGS
        };

        size_t const out_buffer_arg_idx[ NUM_KERNELS ] =
        {
            size_t{ 1 }, // assign_be_mon
            size_t{ 1 }, // clear_be_mon
        };

        for( size_t ii = size_t{ 0 } ; ii < NUM_KERNELS ; ++ii )
        {
            kernel_id_t const kernel_id = kernel_ids[ ii ];
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

    _status_t ClContext::assign_elem_by_elem_config_arg(
        _this_t::cl_buffer_t& SIXTRL_RESTRICT_REF elem_by_elem_config_arg )
    {
        using size_t       = _size_t;
        using kernel_id_t  = _kernel_id_t;

        _status_t status = st::ARCH_STATUS_SUCCESS;
        constexpr size_t NUM_KERNELS = size_t{ 2 };

        kernel_id_t kernel_ids[ NUM_KERNELS ] =
        {
            st::ARCH_ILLEGAL_KERNEL_ID, st::ARCH_ILLEGAL_KERNEL_ID
        };

        kernel_ids[ 0 ] = this->track_elem_by_elem_kernel_id();
        kernel_ids[ 1 ] = this->assign_elem_by_elem_output_kernel_id();

        size_t const min_num_kernel_args[ NUM_KERNELS ] =
        {
            _this_t::MIN_NUM_TRACK_ELEM_ARGS,
            _this_t::MIN_NUM_ASSIGN_ELEM_ARGS
        };

        size_t const elem_by_elem_arg_idx[ NUM_KERNELS ] =
        {
            size_t{ 3 }, // assign_be_mon
            size_t{ 0 }, // clear_be_mon
        };

        for( size_t ii = size_t{ 0 } ; ii < NUM_KERNELS ; ++ii )
        {
            kernel_id_t const kernel_id = kernel_ids[ ii ];
            size_t const min_num_args = min_num_kernel_args[ ii ];

            if( kernel_id == st::ARCH_ILLEGAL_KERNEL_ID ) continue;
            if( min_num_args == size_t{ 0 } ) continue;

            size_t const num_args = this->kernelNumArgs( kernel_id );
            size_t const arg_idx  = elem_by_elem_arg_idx[ ii ];

            if( ( num_args <= arg_idx ) || ( num_args < min_num_args ) )
            {
                status |= st::ARCH_STATUS_GENERAL_FAILURE;
                continue;
            }

            this->assignKernelArgumentClBuffer(
                kernel_id, arg_idx, elem_by_elem_config_arg );
        }

        return status;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ClContext::has_track_until_kernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
                 ( this->m_track_until_turn_kernel_id >= _kernel_id_t{ 0 } ) &&
                 ( static_cast< _size_t >( this->m_track_until_turn_kernel_id )
                    < this->numAvailableKernels() ) );
    }

    _kernel_id_t ClContext::track_until_kernel_id() const SIXTRL_NOEXCEPT
    {
        return ( this->has_track_until_kernel() )
            ? this->m_track_until_turn_kernel_id : st::ARCH_ILLEGAL_KERNEL_ID;
    }

    _status_t ClContext::set_track_until_kernel_id(
        _kernel_id_t const kernel_id )
    {
        _status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->hasSelectedNode() ) && ( kernel_id >= _kernel_id_t{ 0 } ) &&
            ( static_cast< _size_t >( kernel_id ) <
                this->numAvailableKernels() ) )
        {
            _program_id_t const program_id =
                this->programIdByKernelId( kernel_id );

            if( ( program_id >= _program_id_t{ 0 } ) &&
                ( static_cast< _size_t >( program_id ) <
                    this->numAvailablePrograms() ) )
            {
                this->m_track_until_turn_kernel_id  = kernel_id;
                this->m_track_until_turn_program_id = program_id;
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    _this_t::track_status_t ClContext::track_until(
        _this_t::num_turns_t const until_turn )
    {
        _this_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;
        _kernel_id_t const kernel_id = this->track_until_kernel_id();
        _size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= _this_t::MIN_NUM_TRACK_UNTIL_ARGS ) )
        {
            SIXTRL_ASSERT( _this_t::MIN_NUM_TRACK_UNTIL_ARGS >= _size_t{ 4 } );
            SIXTRL_ASSERT( this->m_num_particles_in_pset > _size_t{ 0 } );
            SIXTRL_ASSERT( kernel_id >= _kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< _size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            int64_t const until_turn_arg = static_cast< int64_t >( until_turn );

            this->assignKernelArgumentValue(
                kernel_id, _size_t{ 3 }, until_turn_arg );

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
                    status = static_cast< _this_t::track_status_t >(
                        this->eval_status_flags_after_use() );
                }
            }
        }

        return status;
    }

    _this_t::track_status_t ClContext::track_until(
        _this_t::num_turns_t const until_turn, _size_t const pset_index,
        _size_t const num_particles_in_set, bool const restore_pset_index )
    {
        _this_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;
        _kernel_id_t const kernel_id = this->track_until_kernel_id();
        _size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= _this_t::MIN_NUM_TRACK_UNTIL_ARGS ) )
        {
            SIXTRL_ASSERT( _this_t::MIN_NUM_TRACK_UNTIL_ARGS >= _size_t{ 4 } );
            SIXTRL_ASSERT( num_particles_in_set != _size_t{ 0 } );
            SIXTRL_ASSERT( this->m_num_particles_in_pset > _size_t{ 0 } );
            SIXTRL_ASSERT( kernel_id >= _kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< _size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            if( this->m_pset_index != pset_index )
            {
                uint64_t const pset_index_arg =
                    static_cast< uint64_t >( pset_index );

                this->assignKernelArgumentValue(
                    kernel_id, _size_t{ 1 }, pset_index_arg );
            }

            int64_t const until_turn_arg = static_cast< int64_t >( until_turn );

            this->assignKernelArgumentValue(
                kernel_id, _size_t{ 3 }, until_turn_arg );

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
                    status = static_cast< _this_t::track_status_t >(
                        this->eval_status_flags_after_use() );
                }
            }

            if( ( this->m_pset_index != pset_index ) && ( restore_pset_index ) )
            {
                uint64_t const pset_index_arg =
                    static_cast< uint64_t >( this->m_pset_index );

                this->assignKernelArgumentValue(
                    kernel_id, _size_t{ 1 }, pset_index_arg );
            }
        }

        return status;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ClContext::has_track_line_kernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
            ( this->m_track_line_kernel_id >= _kernel_id_t{ 0 } ) &&
                ( static_cast< _size_t >( this->m_track_line_kernel_id ) <
                    this->numAvailableKernels() ) );
    }

    _kernel_id_t ClContext::track_line_kernel_id() const SIXTRL_NOEXCEPT
    {
        return ( this->has_track_until_kernel() )
            ? this->m_track_line_kernel_id : st::ARCH_ILLEGAL_KERNEL_ID;
    }

    _status_t ClContext::set_track_line_kernel_id(
        _kernel_id_t const kernel_id )
    {
        _status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->hasSelectedNode() ) && ( kernel_id >= _kernel_id_t{ 0 } ) &&
            ( static_cast< _size_t >( kernel_id ) < this->numAvailableKernels() ) )
        {
            _program_id_t const program_id =
                this->programIdByKernelId( kernel_id );

            if( ( program_id >= _program_id_t{ 0 } ) &&
                ( static_cast< _size_t >( program_id ) <
                    this->numAvailablePrograms() ) )
            {
                this->m_track_line_kernel_id  = kernel_id;
                this->m_track_line_program_id = program_id;
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    _this_t::track_status_t ClContext::track_line( _size_t const line_begin_idx,
        _size_t const line_end_idx, bool const finish_turn )
    {
        _this_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;
        _kernel_id_t const kernel_id = this->track_line_kernel_id();
        _size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= _this_t::MIN_NUM_TRACK_LINE_ARGS ) )
        {
            SIXTRL_ASSERT( _this_t::MIN_NUM_TRACK_LINE_ARGS >= _size_t{ 7 } );
            SIXTRL_ASSERT( this->m_num_particles_in_pset > _size_t{ 0 } );
            SIXTRL_ASSERT( kernel_id >= _kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< _size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            uint64_t const line_begin_idx_arg =
                static_cast< uint64_t >( line_begin_idx );

            uint64_t const line_end_idx_arg =
                static_cast< uint64_t >( line_end_idx );

            uint64_t const finish_turn_arg = ( finish_turn )
                ? uint64_t{ 1 } : uint64_t{ 0 };

            this->assignKernelArgumentValue(
                kernel_id, _size_t{ 3 }, line_begin_idx_arg );

            this->assignKernelArgumentValue(
                kernel_id, _size_t{ 4 }, line_end_idx_arg );

            this->assignKernelArgumentValue(
                kernel_id, _size_t{ 5 }, finish_turn_arg );

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
                    status = static_cast< _this_t::track_status_t >(
                        this->eval_status_flags_after_use() );
                }
            }
        }

        return status;
    }

    _this_t::track_status_t ClContext::track_line(
        _this_t::size_type const line_begin_idx,
        _this_t::size_type const line_end_idx,
        bool const finish_turn,
        _this_t::size_type const pset_index,
        _this_t::size_type const num_particles_in_set,
        bool const restore_pset_index )
    {
        _this_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;
        _kernel_id_t const kernel_id = this->track_line_kernel_id();
        _size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= _this_t::MIN_NUM_TRACK_LINE_ARGS ) )
        {
            SIXTRL_ASSERT( _this_t::MIN_NUM_TRACK_LINE_ARGS >= _size_t{ 7 } );
            SIXTRL_ASSERT( num_particles_in_set != _size_t{ 0 } );
            SIXTRL_ASSERT( kernel_id >= _kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< _size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            if( this->m_pset_index != pset_index )
            {
                uint64_t const pset_index_arg =
                    static_cast< uint64_t >( pset_index );

                this->assignKernelArgumentValue(
                    kernel_id, _size_t{ 1 }, pset_index_arg );
            }

            uint64_t const line_begin_idx_arg =
                static_cast< uint64_t >( line_begin_idx );

            uint64_t const line_end_idx_arg =
                static_cast< uint64_t >( line_end_idx );

            uint64_t const finish_turn_arg = ( finish_turn )
                ? uint64_t{ 1 } : uint64_t{ 0 };

            this->assignKernelArgumentValue(
                kernel_id, _size_t{ 3 }, line_begin_idx_arg );

            this->assignKernelArgumentValue(
                kernel_id, _size_t{ 4 }, line_end_idx_arg );

            this->assignKernelArgumentValue(
                kernel_id, _size_t{ 5 }, finish_turn_arg );

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
                    status = static_cast< _this_t::track_status_t >(
                        this->eval_status_flags_after_use() );
                }
            }

            if( ( this->m_pset_index != pset_index ) &&
                ( restore_pset_index ) )
            {
                uint64_t const pset_index_arg =
                    static_cast< uint64_t >( this->m_pset_index );

                this->assignKernelArgumentValue(
                    kernel_id, _size_t{ 1 }, pset_index_arg );
            }
        }

        return status;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ClContext::has_track_elem_by_elem_kernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
            ( this->m_track_elem_by_elem_kernel_id >= _kernel_id_t{ 0 } ) &&
            ( static_cast< _size_t >( this->m_track_elem_by_elem_kernel_id ) <
                         this->numAvailableKernels() ) );
    }

    _kernel_id_t ClContext::track_elem_by_elem_kernel_id() const SIXTRL_NOEXCEPT
    {
        return ( this->has_track_elem_by_elem_kernel() )
            ? this->m_track_elem_by_elem_kernel_id : st::ARCH_ILLEGAL_KERNEL_ID;
    }

    _status_t ClContext::set_track_elem_by_elem_kernel_id(
        _kernel_id_t const kernel_id )
    {
        _status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->hasSelectedNode() ) &&
            ( kernel_id >= _kernel_id_t{ 0 } ) &&
            ( static_cast< _size_t >( kernel_id ) <
                this->numAvailableKernels() ) )
        {
            _program_id_t const program_id =
                this->programIdByKernelId( kernel_id );

            if( ( program_id >= _program_id_t{ 0 } ) &&
                ( static_cast< _size_t >( program_id ) <
                  this->numAvailablePrograms() ) )
            {
                this->m_track_elem_by_elem_kernel_id  = kernel_id;
                this->m_track_elem_by_elem_program_id = program_id;
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    _this_t::track_status_t ClContext::track_elem_by_elem(
        _size_t const until_turn )
    {
        _this_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        _kernel_id_t const kernel_id =
            this->track_elem_by_elem_kernel_id();

        _size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= _this_t::MIN_NUM_TRACK_ELEM_ARGS ) )
        {
            SIXTRL_ASSERT( _this_t::MIN_NUM_TRACK_ELEM_ARGS >= _size_t{ 6 } );
            SIXTRL_ASSERT( this->m_num_particles_in_pset > _size_t{ 0 } );
            SIXTRL_ASSERT( kernel_id >= _kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< _size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            int64_t const until_turn_arg = static_cast< int64_t >( until_turn );

            this->assignKernelArgumentValue(
                kernel_id, _size_t{ 4 }, until_turn_arg );

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
                    status = static_cast< _this_t::track_status_t >(
                        this->eval_status_flags_after_use() );
                }
            }
        }

        return status;
    }

    _this_t::track_status_t ClContext::track_elem_by_elem(
        _size_t const until_turn, _size_t const pset_index,
        _size_t const num_particles_in_set, bool const restore_pset_index )
    {
        _this_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        _kernel_id_t const kernel_id =
            this->track_elem_by_elem_kernel_id();

        _size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= _this_t::MIN_NUM_TRACK_ELEM_ARGS ) )
        {
            SIXTRL_ASSERT( _this_t::MIN_NUM_TRACK_ELEM_ARGS >= _size_t{ 6 } );
            SIXTRL_ASSERT( num_particles_in_set != _size_t{ 0 } );
            SIXTRL_ASSERT( kernel_id >= _kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< _size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            if( this->m_pset_index != pset_index )
            {
                uint64_t const pset_index_arg =
                    static_cast< uint64_t >( pset_index );

                this->assignKernelArgumentValue(
                    kernel_id, _size_t{ 1 }, pset_index_arg );
            }

            int64_t const until_turn_arg = static_cast< int64_t >( until_turn );

            this->assignKernelArgumentValue(
                kernel_id, _size_t{ 4 }, until_turn_arg );

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
                    status = static_cast< _this_t::track_status_t >(
                        this->eval_status_flags_after_use() );
                }
            }

            if( ( this->m_pset_index != pset_index ) &&
                ( restore_pset_index ) )
            {
                uint64_t const pset_index_arg =
                    static_cast< uint64_t >( this->m_pset_index );

                this->assignKernelArgumentValue(
                    kernel_id, _size_t{ 1 }, pset_index_arg );
            }
        }

        return status;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ClContext::has_assign_beam_monitor_output_kernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
            ( this->m_assign_be_mon_out_buffer_kernel_id >=
                _kernel_id_t{ 0 } ) &&
            ( static_cast< _size_t >(
                this->m_assign_be_mon_out_buffer_kernel_id ) <
                this->numAvailableKernels() ) );
    }

    _kernel_id_t ClContext::assign_beam_monitor_output_kernel_id(
        ) const SIXTRL_NOEXCEPT
    {
        return ( this->has_assign_beam_monitor_output_kernel() )
            ? this->m_assign_be_mon_out_buffer_kernel_id
            : st::ARCH_ILLEGAL_KERNEL_ID;
    }

    _status_t ClContext::set_assign_beam_monitor_output_kernel_id(
        _kernel_id_t const kernel_id )
    {
        _status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->hasSelectedNode() ) && ( kernel_id >= _kernel_id_t{ 0 } ) &&
            ( this->numAvailableKernels() >
                static_cast< _size_t >( kernel_id ) ) )
        {
            _program_id_t const program_id =
                this->programIdByKernelId( kernel_id );

            if( ( program_id >= _program_id_t{ 0 } ) &&
                ( static_cast< _size_t >( program_id ) <
                    this->numAvailablePrograms() ) )
            {
                this->m_assign_be_mon_out_buffer_kernel_id  = kernel_id;
                this->m_assign_be_mon_out_buffer_program_id = program_id;
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    _status_t ClContext::assign_beam_monitor_output(
        _this_t::particle_index_t const min_turn_id,
        _size_t const out_buffer_index_offset  )
    {
        _status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        _kernel_id_t const kernel_id =
            this->assign_beam_monitor_output_kernel_id();

        _size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= _this_t::MIN_NUM_ASSIGN_BE_MON_ARGS ) )
        {
            SIXTRL_ASSERT( _this_t::MIN_NUM_ASSIGN_BE_MON_ARGS >= _size_t{ 5 } );
            SIXTRL_ASSERT( kernel_id >= _kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< _size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            int64_t const min_turn_id_arg =
                static_cast< int64_t >( min_turn_id );

            uint64_t const out_buffer_index_offset_arg =
                static_cast< uint64_t >( out_buffer_index_offset );

            this->assignKernelArgumentValue(
                kernel_id, _size_t{ 2 }, min_turn_id_arg );

            this->assignKernelArgumentValue(
                kernel_id, _size_t{ 3 }, out_buffer_index_offset_arg );

            if( !this->debugMode() )
            {
                if( this->runKernel( kernel_id, _size_t{ 1 },
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = st::TRACK_SUCCESS;
                }
            }
            else if( this->prepare_status_flags_for_use() ==
                     st::ARCH_STATUS_SUCCESS )
            {
                if( this->runKernel( kernel_id, _size_t{ 1 },
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = this->eval_status_flags_after_use();
                }
            }
        }

        return status;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ClContext::has_assign_elem_by_elem_output_kernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
            ( this->m_assign_elem_by_elem_out_buffer_kernel_id >=
                _kernel_id_t{ 0 } ) &&
            ( static_cast< _size_t >(
                this->m_assign_elem_by_elem_out_buffer_kernel_id ) <
                this->numAvailableKernels() ) );
    }

    _kernel_id_t ClContext::assign_elem_by_elem_output_kernel_id() const SIXTRL_NOEXCEPT
    {
        return ( this->has_assign_elem_by_elem_output_kernel() )
            ? this->m_assign_elem_by_elem_out_buffer_kernel_id
            : st::ARCH_ILLEGAL_KERNEL_ID;
    }

    _status_t ClContext::set_assign_elem_by_elem_output_kernel_id(
        _kernel_id_t const kernel_id )
    {
        _status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->hasSelectedNode() ) && ( kernel_id >= _kernel_id_t{ 0 } ) &&
            ( this->numAvailableKernels() >
                static_cast< _size_t >( kernel_id ) ) )
        {
            _program_id_t const program_id = this->programIdByKernelId(
                kernel_id );

            if( ( program_id >= _program_id_t{ 0 } ) &&
                ( static_cast< _size_t >( program_id ) <
                    this->numAvailablePrograms() ) )
            {
                this->m_assign_elem_by_elem_out_buffer_kernel_id  = kernel_id;
                this->m_assign_elem_by_elem_out_buffer_program_id = program_id;
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    _status_t ClContext::assign_elem_by_elem_output(
        ClContext::size_type const out_buffer_index_offset )
    {
        _status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        _kernel_id_t const kernel_id =
            this->assign_elem_by_elem_output_kernel_id();

        _size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= _this_t::MIN_NUM_ASSIGN_ELEM_ARGS ) )
        {
            SIXTRL_ASSERT( _this_t::MIN_NUM_ASSIGN_ELEM_ARGS >= _size_t{ 4 } );
            SIXTRL_ASSERT( kernel_id >= _kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< _size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            uint64_t const out_buffer_index_offset_arg =
                static_cast< uint64_t >( out_buffer_index_offset );

            this->assignKernelArgumentValue(
                kernel_id, _size_t{ 2 }, out_buffer_index_offset_arg );

            if( !this->debugMode() )
            {
                if( this->runKernel( kernel_id, _size_t{ 1 },
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = st::TRACK_SUCCESS;
                }
            }
            else if( this->prepare_status_flags_for_use() ==
                     st::ARCH_STATUS_SUCCESS )
            {
                if( this->runKernel( kernel_id, _size_t{ 1 },
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = this->eval_status_flags_after_use();
                }
            }
        }

        return status;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ClContext::has_clear_beam_monitor_output_kernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
            ( this->m_clear_be_mon_kernel_id >= _kernel_id_t{ 0 } ) &&
            ( static_cast< _size_t >( this->m_clear_be_mon_kernel_id ) <
                this->numAvailableKernels() ) );
    }

    _kernel_id_t ClContext::clear_beam_monitor_output_kernel_id() const SIXTRL_NOEXCEPT
    {
        return ( this->has_clear_beam_monitor_output_kernel() )
            ? this->m_clear_be_mon_kernel_id : st::ARCH_ILLEGAL_KERNEL_ID;
    }

    _status_t ClContext::set_clear_beam_monitor_output_kernel_id(
        _kernel_id_t const kernel_id )
    {
        using size_t = _size_t;
        using kernel_id_t = _kernel_id_t;

        _status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->hasSelectedNode() ) && ( kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_t >( kernel_id ) < this->numAvailableKernels() ) )
        {
            _program_id_t const program_id =
                this->programIdByKernelId( kernel_id );

            if( ( program_id >= _program_id_t{ 0 } ) &&
                ( static_cast< _size_t >( program_id ) <
                  this->numAvailablePrograms() ) )
            {
                this->m_clear_be_mon_kernel_id  = kernel_id;
                this->m_clear_be_mon_program_id = program_id;
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    _status_t ClContext::clear_beam_monitor_output()
    {
        _status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        _kernel_id_t const kernel_id =
            this->clear_beam_monitor_output_kernel_id();

        _size_t const num_kernel_args = this->kernelNumArgs( kernel_id );

        if( ( this->hasSelectedNode() ) &&
            ( num_kernel_args >= _this_t::MIN_NUM_CLEAR_BE_MON_ARGS ) )
        {
            SIXTRL_ASSERT( _this_t::MIN_NUM_CLEAR_BE_MON_ARGS >= _size_t{ 2 } );
            SIXTRL_ASSERT( kernel_id >= _kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< _size_t >( kernel_id ) <
                           this->numAvailableKernels() );

            if( !this->debugMode() )
            {
                if( this->runKernel( kernel_id, _size_t{ 1 },
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = st::TRACK_SUCCESS;
                }
            }
            else if( this->prepare_status_flags_for_use() ==
                     st::ARCH_STATUS_SUCCESS )
            {
                if( this->runKernel( kernel_id, _size_t{ 1 },
                        this->lastExecWorkGroupSize( kernel_id ) ) )
                {
                    status = this->eval_status_flags_after_use();
                }
            }
        }

        return status;
    }

    /* --------------------------------------------------------------------- */

    _size_t ClContext::selected_particle_set() const SIXTRL_NOEXCEPT
    {
        return this->m_pset_index;
    }

    _size_t ClContext::num_particles_in_selected_set() const SIXTRL_NOEXCEPT
    {
        return this->m_num_particles_in_pset;
    }

    /* --------------------------------------------------------------------- */

    bool ClContext::use_optimized_tracking() const SIXTRL_NOEXCEPT
    {
        return this->m_use_optimized_tracking;
    }

    void ClContext::enable_optimized_tracking()
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

    void ClContext::disable_optimized_tracking()
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

    bool ClContext::is_beam_beam_tracking_enabled() const SIXTRL_NOEXCEPT
    {
        return this->m_enable_beam_beam;
    }

    void ClContext::enable_beam_beam_tracking()
    {
        if( ( !this->is_beam_beam_tracking_enabled() ) &&
            ( !this->hasSelectedNode() ) )
        {
            this->clear();

            this->m_enable_beam_beam = true;
            this->doInitDefaultPrograms();
            this->doInitDefaultKernels();
        }
    }

    void ClContext::disable_beam_beam_tracking()
    {
        if( (  this->is_beam_beam_tracking_enabled() ) &&
            ( !this->hasSelectedNode() ) )
        {
            this->clear();

            this->m_enable_beam_beam = false;
            this->doInitDefaultPrograms();
            this->doInitDefaultKernels();
        }
    }

    /* --------------------------------------------------------------------- */

    std::unique_ptr< _this_t::cl_buffer_t >
    ClContext::create_elem_by_elem_config_arg()
    {
        if( ( this->hasSelectedNode() ) &&
            ( this->openClContext() != nullptr ) &&
            ( this->openClQueue() != nullptr ) )
        {
            _size_t const type_size = sizeof( _this_t::elem_by_elem_config_t );

            return std::unique_ptr< _this_t::cl_buffer_t >(
                new _this_t::cl_buffer_t( *this->openClContext(),
                    CL_MEM_READ_WRITE, type_size , nullptr ) );
        }

        return std::unique_ptr< _this_t::cl_buffer_t >( nullptr );
    }

    void ClContext::delete_elem_by_elem_config_arg(
        std::unique_ptr< SIXTRL_CXX_NAMESPACE::ClContext::cl_buffer_t >&& ptr )
    {
        std::unique_ptr< SIXTRL_CXX_NAMESPACE::ClContext::cl_buffer_t >
            _local_ptr = std::move( ptr );

        _local_ptr.reset( nullptr );
    }

    _this_t::status_t ClContext::init_elem_by_elem_config_arg(
        _this_t::cl_buffer_t& SIXTRL_RESTRICT_REF elem_by_elem_config_arg,
        _this_t::elem_by_elem_config_t& SIXTRL_RESTRICT_REF elem_by_elem_config,
        const ::NS(Buffer) *const SIXTRL_RESTRICT pbuffer,
        _this_t::size_type const num_psets,
        _this_t::size_type const* SIXTRL_RESTRICT pset_indices_begin,
        const ::NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
        _this_t::size_type const until_turn_elem_by_elem,
        _this_t::particle_index_t const start_elem_id  )
    {
        _this_t::status_t status = ::st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( !this->hasSelectedNode() ) ||
            (  this->openClQueue() == nullptr ) ||
            (  this->openClContext() == nullptr ) )
        {
            return status;
        }

        status = ::NS(ElemByElemConfig_init_on_particle_sets)(
            &elem_by_elem_config, pbuffer, num_psets, pset_indices_begin,
                beam_elements_buffer, start_elem_id,
                    until_turn_elem_by_elem );

        if( status == ::NS(ARCH_STATUS_SUCCESS) )
        {
            ::NS(ElemByElemConfig_set_output_store_address)(
                &elem_by_elem_config, uintptr_t{ 0 } );

            _size_t const type_size = sizeof( _this_t::elem_by_elem_config_t );

            cl_int const cl_ret = this->openClQueue()->enqueueWriteBuffer(
                elem_by_elem_config_arg, CL_TRUE, _size_t{ 0 }, type_size,
                    &elem_by_elem_config );

            if( cl_ret != CL_SUCCESS )
            {
                status = ::NS(ARCH_STATUS_GENERAL_FAILURE);
            }
        }

        return status;
    }

    _this_t::status_t ClContext::collect_elem_by_elem_config_arg(
        _this_t::cl_buffer_t& SIXTRL_RESTRICT_REF elem_by_elem_config_arg,
        _this_t::elem_by_elem_config_t& SIXTRL_RESTRICT_REF
            elem_by_elem_config )
    {
        _this_t::status_t status = ::st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( !this->hasSelectedNode() ) ||
            (  this->openClQueue() == nullptr ) ||
            (  this->openClContext() == nullptr ) )
        {
            return status;
        }

        _size_t const type_size = sizeof( _this_t::elem_by_elem_config_t );
        cl_int const cl_ret = this->openClQueue()->enqueueReadBuffer(
            elem_by_elem_config_arg, CL_TRUE, _size_t{ 0 }, type_size,
                &elem_by_elem_config );

        if( cl_ret == CL_SUCCESS ) status = st::ARCH_STATUS_SUCCESS;

        return status;
    }

    _this_t::status_t ClContext::push_elem_by_elem_config_arg(
        _this_t::cl_buffer_t& SIXTRL_RESTRICT_REF elem_by_elem_config_arg,
        _this_t::elem_by_elem_config_t const& SIXTRL_RESTRICT_REF
            elem_by_elem_config )
    {
        _this_t::status_t status = ::st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( !this->hasSelectedNode() ) ||
            (  this->openClQueue() == nullptr ) ||
            (  this->openClContext() == nullptr ) )
        {
            return status;
        }

        _size_t const type_size = sizeof( _this_t::elem_by_elem_config_t );
        cl_int const cl_ret = this->openClQueue()->enqueueWriteBuffer(
            elem_by_elem_config_arg, CL_TRUE, _size_t{ 0 }, type_size,
                &elem_by_elem_config );

        if( cl_ret == CL_SUCCESS ) status = st::ARCH_STATUS_SUCCESS;

        return status;
    }

    /* --------------------------------------------------------------------- */

    bool ClContext::doSelectNode( _this_t::size_type node_index )
    {
        /* WARNING: Workaround for AMD Heisenbug */
        if( ( this->use_optimized_tracking() ) &&
            ( this->isAvailableNodeAMDPlatform( node_index ) ) )
        {
            this->disable_optimized_tracking();
        }

        return _base_t::doSelectNode( node_index );
    }

    bool ClContext::doInitDefaultPrograms()
    {
        return ( ( ClContextBase::doInitDefaultPrograms() ) &&
                 ( this->doInitDefaultProgramsPrivImpl() ) );
    }

    bool ClContext::doInitDefaultKernels()
    {
        return ( ( ClContextBase::doInitDefaultKernels() ) &&
                 ( this->doInitDefaultKernelsPrivImpl() ) );
    }

    _status_t ClContext::doAssignStatusFlagsArgPrivImpl(
        _this_t::cl_buffer_t& SIXTRL_RESTRICT_REF status_flags_arg )
    {
        using size_t = _size_t;
        using kernel_id_t = _kernel_id_t;

        _status_t status = st::ARCH_STATUS_SUCCESS;
        if( !this->debugMode() ) return status;

        constexpr size_t NUM_KERNELS = size_t{ 6 };

        kernel_id_t kernel_ids[ NUM_KERNELS ];
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
            _this_t::MIN_NUM_TRACK_UNTIL_ARGS,
            _this_t::MIN_NUM_TRACK_LINE_ARGS,
            _this_t::MIN_NUM_TRACK_ELEM_ARGS,
            _this_t::MIN_NUM_ASSIGN_BE_MON_ARGS,
            _this_t::MIN_NUM_CLEAR_BE_MON_ARGS,
            _this_t::MIN_NUM_ASSIGN_ELEM_ARGS
        };

        size_t const status_flags_arg_idx[ NUM_KERNELS ] =
        {
            size_t{ 5 }, // track_until
            size_t{ 7 }, // track_line
            size_t{ 6 }, // track_elem_elem
            size_t{ 5 }, // assign_be_mon
            size_t{ 2 }, // clear_be_mon
            size_t{ 4 }  // assign_elem_by_elem
        };

        for( size_t ii = size_t{ 0 } ; ii < NUM_KERNELS ; ++ii )
        {
            kernel_id_t const kernel_id = kernel_ids[ ii ];
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

    _status_t ClContext::doAssignStatusFlagsArg(
        _this_t::cl_buffer_t& SIXTRL_RESTRICT_REF status_flags_arg )
    {
        _status_t status =
            _base_t::doAssignStatusFlagsArg( status_flags_arg );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doAssignStatusFlagsArgPrivImpl( status_flags_arg );
        }

        return status;
    }

    _status_t ClContext::doAssignSlotSizeArgPrivImpl( _size_t const slot_size )
    {
        using size_t = _size_t;
        using kernel_id_t = _kernel_id_t;

        _status_t status = st::ARCH_STATUS_SUCCESS;
        constexpr size_t NUM_KERNELS = size_t{ 6 };

        kernel_id_t kernel_ids[ NUM_KERNELS ];
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
            _this_t::MIN_NUM_TRACK_UNTIL_ARGS,
            _this_t::MIN_NUM_TRACK_LINE_ARGS,
            _this_t::MIN_NUM_TRACK_ELEM_ARGS,
            _this_t::MIN_NUM_ASSIGN_BE_MON_ARGS,
            _this_t::MIN_NUM_CLEAR_BE_MON_ARGS,
            _this_t::MIN_NUM_ASSIGN_ELEM_ARGS
        };

        size_t const slot_size_arg_idx[ NUM_KERNELS ] =
        {
            size_t{ 4 }, // track_until
            size_t{ 6 }, // track_line
            size_t{ 5 }, // track_elem_elem
            size_t{ 4 }, // assign_be_mon
            size_t{ 1 }, // clear_be_mon
            size_t{ 3 }  // assign_elem_by_elem
        };

        uint64_t const slot_size_arg = static_cast< uint64_t >( slot_size );

        for( size_t ii = size_t{ 0 } ; ii < NUM_KERNELS ; ++ii )
        {
            kernel_id_t const kernel_id = kernel_ids[ ii ];
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

    _status_t ClContext::doAssignSlotSizeArg( _size_t const slot_size )
    {
        _status_t status = _base_t::doAssignSlotSizeArg( slot_size );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doAssignSlotSizeArgPrivImpl( slot_size );
        }

        return status;
    }

    bool ClContext::doInitDefaultProgramsPrivImpl()
    {
        bool success = false;

        std::string path_to_kernel_dir( NS(PATH_TO_BASE_DIR) );
        path_to_kernel_dir += "sixtracklib/opencl/kernels/";

        std::string path_to_particles_track_prog          = path_to_kernel_dir;
        std::string path_to_particles_track_opt_prog      = path_to_kernel_dir;
        std::string path_to_assign_be_mon_out_buffer_prog = path_to_kernel_dir;
        std::string path_to_assign_elem_out_buffer_prog   = path_to_kernel_dir;

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

        std::string track_compile_options = "-D_GPUCODE=1";
        track_compile_options += " -DSIXTRL_BUFFER_ARGPTR_DEC=__private";
        track_compile_options += " -DSIXTRL_BUFFER_DATAPTR_DEC=__global";
        track_compile_options += " -DSIXTRL_PARTICLE_ARGPTR_DEC=__global";
        track_compile_options += " -DSIXTRL_PARTICLE_DATAPTR_DEC=__global";

        if( !this->is_beam_beam_tracking_enabled() )
        {
            track_compile_options += " -DSIXTRL_DISABLE_BEAM_BEAM=1";
        }

        track_compile_options += " -I";
        track_compile_options += NS(PATH_TO_SIXTRL_INCLUDE_DIR);

        std::string track_optimized_compile_options = "-D_GPUCODE=1";
        track_optimized_compile_options += " -DSIXTRL_BUFFER_ARGPTR_DEC=__private";
        track_optimized_compile_options += " -DSIXTRL_BUFFER_DATAPTR_DEC=__global";
        track_optimized_compile_options += " -DSIXTRL_PARTICLE_ARGPTR_DEC=__private";
        track_optimized_compile_options += " -DSIXTRL_PARTICLE_DATAPTR_DEC=__private";

        if( !this->is_beam_beam_tracking_enabled() )
        {
            track_optimized_compile_options += " -DSIXTRL_DISABLE_BEAM_BEAM=1";
        }

        track_optimized_compile_options += " -I";
        track_optimized_compile_options += NS(PATH_TO_SIXTRL_INCLUDE_DIR);

        std::string assign_out_buffer_compile_options = " -D_GPUCODE=1";
        assign_out_buffer_compile_options += " -DSIXTRL_BUFFER_ARGPTR_DEC=__private";
        assign_out_buffer_compile_options += " -DSIXTRL_BUFFER_DATAPTR_DEC=__global";
        assign_out_buffer_compile_options += " -DSIXTRL_PARTICLE_ARGPTR_DEC=__global";
        assign_out_buffer_compile_options += " -DSIXTRL_PARTICLE_DATAPTR_DEC=__global";
        assign_out_buffer_compile_options += " -I";
        assign_out_buffer_compile_options += NS(PATH_TO_SIXTRL_INCLUDE_DIR);

        program_id_t const track_program_id = this->addProgramFile(
            path_to_particles_track_prog, track_compile_options );

        program_id_t const track_optimized_program_id = this->addProgramFile(
            path_to_particles_track_opt_prog, track_optimized_compile_options );

        program_id_t const assign_be_mon_out_buffer_program_id =
            this->addProgramFile( path_to_assign_be_mon_out_buffer_prog,
                                  assign_out_buffer_compile_options );

        program_id_t const assign_elem_by_elem_out_buffer_program_id =
            this->addProgramFile( path_to_assign_elem_out_buffer_prog,
                                  assign_out_buffer_compile_options );

        if( ( track_program_id            >= program_id_t{ 0 } ) &&
            ( track_optimized_program_id  >= program_id_t{ 0 } ) &&
            ( assign_be_mon_out_buffer_program_id >= program_id_t{ 0 } ) &&
            ( assign_elem_by_elem_out_buffer_program_id >= program_id_t{ 0 } ) )
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

            success = true;
        }

        return success;
    }

    bool ClContext::doInitDefaultKernelsPrivImpl()
    {
        bool success = false;

        if( this->hasSelectedNode() )
        {
            program_id_t const max_program_id = static_cast< program_id_t >(
                this->numAvailablePrograms() );

            if( ( this->m_track_until_turn_program_id >= program_id_t{ 0 } ) &&
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

                kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_track_until_turn_program_id );

                if( kernel_id >= kernel_id_t{ 0 } )
                {
                    success = ( this->set_track_until_kernel_id( kernel_id ) ==
                        st::ARCH_STATUS_SUCCESS );
                }
            }

            if( ( success ) &&
                ( this->m_track_elem_by_elem_program_id >= program_id_t{ 0 } ) &&
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

                kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_track_elem_by_elem_program_id );

                if( kernel_id >= kernel_id_t{ 0 } )
                {
                    success = ( this->set_track_elem_by_elem_kernel_id(
                        kernel_id ) == st::ARCH_STATUS_SUCCESS );
                }
            }

            if( ( success ) &&
                ( this->m_track_line_program_id >= program_id_t{ 0 } ) &&
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

                kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_track_line_program_id );

                if( kernel_id >= kernel_id_t{ 0 } )
                {
                    success = ( this->set_track_line_kernel_id( kernel_id ) ==
                        st::ARCH_STATUS_SUCCESS );
                }
            }

            if( ( success ) &&
                ( this->m_assign_be_mon_out_buffer_program_id >= program_id_t{ 0 } ) &&
                ( this->m_assign_be_mon_out_buffer_program_id <  max_program_id ) )
            {
                std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
                kernel_name += "BeamMonitor_assign_out_buffer_from_offset";

                if( this->debugMode() )
                {
                    kernel_name += "_debug";
                }

                kernel_name += "_opencl";

                kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_assign_be_mon_out_buffer_program_id );

                if( kernel_id >= kernel_id_t{ 0 } )
                {
                    success = ( this->set_assign_beam_monitor_output_kernel_id(
                        kernel_id ) == st::ARCH_STATUS_SUCCESS );
                }
            }

            if( ( success ) &&
                ( this->m_clear_be_mon_program_id >= program_id_t{ 0 } ) &&
                ( this->m_clear_be_mon_program_id <  max_program_id ) )
            {
                std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
                kernel_name += "BeamMonitor_clear_all_line_obj";

                if( this->debugMode() )
                {
                    kernel_name += "_debug";
                }

                kernel_name += "_opencl";

                kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_clear_be_mon_program_id );

                if( kernel_id >= kernel_id_t{ 0 } )
                {
                    success = ( this->set_clear_beam_monitor_output_kernel_id(
                        kernel_id ) == st::ARCH_STATUS_SUCCESS );
                }
            }

            if( ( success ) &&
                ( this->m_assign_elem_by_elem_out_buffer_program_id >=
                    program_id_t{ 0 } ) &&
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

                kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(),
                    this->m_assign_elem_by_elem_out_buffer_program_id );

                if( kernel_id >= kernel_id_t{ 0 } )
                {
                    success = ( this->set_assign_elem_by_elem_output_kernel_id(
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
    return new st::ClContext;
}

::NS(ClContext)* NS(ClContext_new)( const char* node_id_str )
{
    return new st::ClContext( node_id_str, nullptr );
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
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(ClContext_assign_particle_set_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(buffer_size_t) const particle_set_index,
    ::NS(buffer_size_t) const num_particles_in_selected_set )
{
    return ( ctx != nullptr )
        ? ctx->assign_particle_set_arg(
            particle_set_index, num_particles_in_selected_set )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ctrl_status_t) NS(ClContext_assign_beam_elements_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(ClArgument)* SIXTRL_RESTRICT beam_elem_arg )
{
    return ( ( ctx != nullptr ) && ( beam_elem_arg != nullptr ) )
        ? ctx->assign_beam_elements_arg( *beam_elem_arg )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ctrl_status_t) NS(ClContext_assign_output_buffer_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(ClArgument)* SIXTRL_RESTRICT_REF out_buffer_arg )
{
    return ( ( ctx != nullptr ) && ( out_buffer_arg != nullptr ) )
        ? ctx->assign_output_buffer_arg( *out_buffer_arg )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ctrl_status_t) NS(ClContext_assign_elem_by_elem_config_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx, cl_mem elem_by_elem_config_arg  )
{
    cl::Buffer temp_buffer( elem_by_elem_config_arg );

    return ( ctx != nullptr )
        ? ctx->assign_elem_by_elem_config_arg( temp_buffer )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ctrl_status_t) NS(ClContext_assign_slot_size_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx, ::NS(buffer_size_t) const slot_size )
{
    return ( ctx != nullptr )
        ? ctx->assign_slot_size_arg( slot_size )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ctrl_status_t) NS(ClContext_assign_status_flags_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx, cl_mem status_flags_arg )
{
    cl::Buffer temp_buffer( status_flags_arg );

    return ( ctx != nullptr )
        ? ctx->assign_status_flags_arg( temp_buffer )
        : st::ARCH_STATUS_GENERAL_FAILURE;
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
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(track_status_t) NS(ClContext_track_until)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(context_num_turns_t) const until_turn )
{
    return ( ctx != nullptr ) ? ctx->track_until( until_turn )
        : st::TRACK_STATUS_GENERAL_FAILURE;
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
        : st::TRACK_STATUS_GENERAL_FAILURE;
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
        : st::ARCH_ILLEGAL_KERNEL_ID;
}

::NS(arch_status_t) NS(ClContext_set_track_line_kernel_id)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->set_track_line_kernel_id( kernel_id )
        : st::ARCH_STATUS_GENERAL_FAILURE;
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
        : st::TRACK_STATUS_GENERAL_FAILURE;
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
        ? ctx->track_elem_by_elem_kernel_id() : st::ARCH_ILLEGAL_KERNEL_ID;
}

::NS(arch_status_t) NS(ClContext_set_track_elem_by_elem_kernel_id)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->set_track_elem_by_elem_kernel_id( kernel_id )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(track_status_t) NS(ClContext_track_elem_by_elem)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(buffer_size_t) const until_turn )
{
    return ( ctx != nullptr ) ? ctx->track_elem_by_elem( until_turn )
        : st::TRACK_STATUS_GENERAL_FAILURE;
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
        : st::TRACK_STATUS_GENERAL_FAILURE;
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
        : st::ARCH_ILLEGAL_KERNEL_ID;
}

::NS(arch_status_t) NS(ClContext_set_assign_beam_monitor_output_kernel_id)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->set_assign_beam_monitor_output_kernel_id( kernel_id )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(ClContext_assign_beam_monitor_output)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(particle_index_t) const min_turn_id,
    ::NS(buffer_size_t) const out_buffer_index_offset )
{
    return ( ctx != nullptr )
        ? ctx->assign_beam_monitor_output(
            min_turn_id, out_buffer_index_offset )
        : st::ARCH_STATUS_GENERAL_FAILURE;
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
        : st::ARCH_ILLEGAL_KERNEL_ID;
}

::NS(arch_status_t) NS(ClContext_set_assign_elem_by_elem_output_kernel_id)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->set_assign_elem_by_elem_output_kernel_id( kernel_id )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(ClContext_assign_elem_by_elem_output)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(buffer_size_t) const out_buffer_index_offset )
{
    return ( ctx != nullptr )
        ? ctx->assign_elem_by_elem_output( out_buffer_index_offset )
        : st::ARCH_STATUS_GENERAL_FAILURE;
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
        : st::ARCH_ILLEGAL_KERNEL_ID;
}

::NS(arch_status_t) NS(ClContext_set_clear_beam_monitor_output_kernel_id)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->set_clear_beam_monitor_output_kernel_id( kernel_id )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ctrl_status_t) NS(ClContext_clear_beam_monitor_output)(
    ::NS(ClContext)*  SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->clear_beam_monitor_output() : st::ARCH_STATUS_GENERAL_FAILURE;
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

/* ------------------------------------------------------------------------- */

cl_mem NS(ClContext_create_elem_by_elem_config_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx )
{
    if( ( ctx != nullptr ) && ( ctx->hasSelectedNode() ) &&
        ( ctx->openClQueue()   != nullptr ) &&
        ( ctx->openClContext() != nullptr ) )
    {
        cl_context& ocl_ctx = ( *ctx->openClContext() )();
        cl_int cl_ret = CL_SUCCESS;

        cl_mem arg_buffer = ::clCreateBuffer( ocl_ctx, CL_MEM_READ_WRITE,
            sizeof( ::NS(ElemByElemConfig) ), nullptr, &cl_ret );

        if( cl_ret == CL_SUCCESS )
        {
            cl_uint ref_cnt = cl_uint{ 10000 };
            cl_ret = ::clGetMemObjectInfo( arg_buffer, CL_MEM_REFERENCE_COUNT,
                sizeof( cl_uint ), &ref_cnt, nullptr );

            if( cl_ret == CL_SUCCESS )
            {
                ::clRetainMemObject( arg_buffer );
                return arg_buffer;
            }
        }
    }

    return cl_mem{};
}

void NS(ClContext_delete_elem_by_elem_config_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx, cl_mem elem_by_elem_config_arg )
{
    cl_uint ref_cnt = cl_uint{ 10000 };
    cl_int cl_ret = ::clGetMemObjectInfo( elem_by_elem_config_arg,
        CL_MEM_REFERENCE_COUNT, sizeof( cl_uint ), &ref_cnt, nullptr );

    if( ( cl_ret == CL_SUCCESS ) && ( ref_cnt > cl_uint{ 0 } ) )
    {
        ::clReleaseMemObject( elem_by_elem_config_arg );
    }

    ( void )ctx;
}

::NS(arch_status_t) NS(ClContext_init_elem_by_elem_config_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx,
    cl_mem elem_by_elem_config_arg,
    ::NS(ElemByElemConfig)* SIXTRL_RESTRICT elem_by_elem_config,
    const ::NS(Buffer) *const SIXTRL_RESTRICT pbuffer,
    ::NS(buffer_size_t) const num_psets,
    ::NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    const ::NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
    ::NS(buffer_size_t) const until_turn_elem_by_elem,
    ::NS(particle_index_t) const start_elem_id )
{
    ::NS(arch_status_t) status = ::NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( elem_by_elem_config == nullptr ) || ( ctx == nullptr ) ||
        ( !ctx->hasSelectedNode() ) ||
        ( ctx->openClQueue() == nullptr ) ||
        ( ctx->openClContext() == nullptr ) )
    {
        return status;
    }

    cl_int cl_ret = ::clRetainMemObject( elem_by_elem_config_arg );

    if( cl_ret == CL_SUCCESS )
    {
        ::cl_command_queue& ocl_queue = ( *ctx->openClQueue() )();

        status = ::NS(ElemByElemConfig_init_on_particle_sets)(
            elem_by_elem_config, pbuffer, num_psets, pset_indices_begin,
                beam_elements_buffer, start_elem_id,
                    until_turn_elem_by_elem );

        if( status == ::NS(ARCH_STATUS_SUCCESS) )
        {
            ::NS(ElemByElemConfig_set_output_store_address)(
                elem_by_elem_config, uintptr_t{ 0 } );

            size_t const type_size = sizeof( ::NS(ElemByElemConfig) );

            cl_ret = ::clEnqueueWriteBuffer( ocl_queue, elem_by_elem_config_arg,
                CL_TRUE, size_t { 0 }, type_size, elem_by_elem_config,
                    cl_uint{ 0 }, nullptr, nullptr );

            if( cl_ret != CL_SUCCESS )
            {
                status = ::NS(ARCH_STATUS_GENERAL_FAILURE);
            }
        }

        ::clReleaseMemObject( elem_by_elem_config_arg );
    }

    return status;
}

::NS(arch_status_t) NS(ClContext_collect_elem_by_elem_config_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx, cl_mem elem_by_elem_config_arg,
    ::NS(ElemByElemConfig)* SIXTRL_RESTRICT config )
{
    ::NS(arch_status_t) status = ::NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( config == nullptr ) || ( ctx == nullptr ) ||
        ( !ctx->hasSelectedNode() ) ||
        ( ctx->openClQueue() == nullptr ) ||
        ( ctx->openClContext() == nullptr ) )
    {
        return status;
    }

    cl_int cl_ret = ::clRetainMemObject( elem_by_elem_config_arg );

    if( cl_ret == CL_SUCCESS )
    {
        ::cl_command_queue& ocl_queue = ( *ctx->openClQueue() )();
        size_t const type_size = sizeof( ::NS(ElemByElemConfig) );

        cl_ret = ::clEnqueueReadBuffer( ocl_queue,
            elem_by_elem_config_arg, CL_TRUE, size_t{ 0 }, type_size,
                config, cl_uint{ 0 }, nullptr, nullptr );


        if( cl_ret != CL_SUCCESS ) status = ::NS(ARCH_STATUS_SUCCESS);
        ::clReleaseMemObject( elem_by_elem_config_arg );
    }

    return status;
}

::NS(arch_status_t) NS(ClContext_push_elem_by_elem_config_arg)(
    ::NS(ClContext)* SIXTRL_RESTRICT ctx, cl_mem elem_by_elem_config_arg,
    const ::NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    ::NS(arch_status_t) status = ::NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( config == nullptr ) || ( ctx == nullptr ) ||
        ( !ctx->hasSelectedNode() ) ||
        ( ctx->openClQueue() == nullptr ) ||
        ( ctx->openClContext() == nullptr ) )
    {
        return status;
    }

    cl_int cl_ret = ::clRetainMemObject( elem_by_elem_config_arg );

    if( cl_ret == CL_SUCCESS )
    {
        ::cl_command_queue& ocl_queue = ( *ctx->openClQueue() )();
        size_t const type_size = sizeof( ::NS(ElemByElemConfig) );

        cl_ret = ::clEnqueueWriteBuffer( ocl_queue,
            elem_by_elem_config_arg, CL_TRUE, size_t { 0 }, type_size,
                config, cl_uint{ 0 }, nullptr, nullptr );

        if( cl_ret != CL_SUCCESS ) status = ::NS(ARCH_STATUS_SUCCESS);
        ::clReleaseMemObject( elem_by_elem_config_arg );
    }

    return status;
}

#endif /* !defined( __CUDACC__ ) */
/* end: sixtracklib/opencl/internal/context.cpp */
