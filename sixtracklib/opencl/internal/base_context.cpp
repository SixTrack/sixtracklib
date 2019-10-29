#include "sixtracklib/opencl/internal/base_context.h"

#if !defined( __CUDACC__ )

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cctype>
#include <fstream>
#include <iterator>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/debug_register.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/context/compute_arch.h"

#include "sixtracklib/opencl/cl.h"
#include "sixtracklib/opencl/argument.h"

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    using  _this_t = st::ClContextBase;

    constexpr _this_t::kernel_arg_type_t _this_t::ARG_TYPE_NONE;
    constexpr _this_t::kernel_arg_type_t _this_t::ARG_TYPE_VALUE;
    constexpr _this_t::kernel_arg_type_t _this_t::ARG_TYPE_RAW_PTR;
    constexpr _this_t::kernel_arg_type_t _this_t::ARG_TYPE_CL_ARGUMENT;
    constexpr _this_t::kernel_arg_type_t _this_t::ARG_TYPE_CL_BUFFER;
    constexpr _this_t::kernel_arg_type_t _this_t::ARG_TYPE_INVALID;
    constexpr _this_t::size_type _this_t::MIN_NUM_REMAP_BUFFER_ARGS;

    ClContextBase::ClContextBase(
        const char *const SIXTRL_RESTRICT config_str ) :
        m_cl_programs(),
        m_cl_kernels(),
        m_cl_buffers(),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_available_devices(),
        m_program_data(),
        m_kernel_data(),
        m_default_compile_options(),
        m_config_str(),
        m_cl_context(),
        m_cl_queue(),
        m_cl_success_flag(),
        m_remap_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_remap_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_selected_node_index( int64_t{ -1 } ),
        m_default_kernel_arg( uint64_t{ 0 } ),
        m_debug_mode( false )
    {
        using _this_t = ClContextBase;

        _this_t::UpdateAvailableNodes(
            this->m_available_nodes_id, this->m_available_nodes_info,
            this->m_available_devices );

        SIXTRL_ASSERT( this->m_available_nodes_id.size() ==
                       this->m_available_nodes_info.size() );

        this->doInitDefaultProgramsBaseImpl();

        this->doSetConfigStr( config_str );

        if( this->configStr() != nullptr )
        {
            this->doParseConfigStringBaseImpl( this->configStr() );
        }
    }

    ClContextBase::ClContextBase(
        _this_t::size_type const node_index,
        const char *const SIXTRL_RESTRICT config_str ) :
        m_cl_programs(),
        m_cl_kernels(),
        m_cl_buffers(),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_available_devices(),
        m_program_data(),
        m_kernel_data(),
        m_default_compile_options(),
        m_config_str(),
        m_cl_context(),
        m_cl_queue(),
        m_cl_success_flag(),
        m_remap_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_remap_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_selected_node_index( int64_t{ -1 } ),
        m_default_kernel_arg( uint64_t{ 0 } ),
        m_debug_mode( false )
    {
        using _this_t = ClContextBase;

        _this_t::UpdateAvailableNodes(
            this->m_available_nodes_id, this->m_available_nodes_info,
            this->m_available_devices, nullptr, this->debugMode() );

        this->doSetConfigStr( config_str );

        if( this->configStr() != nullptr )
        {
            this->doParseConfigStringBaseImpl( this->configStr() );
        }

        SIXTRL_ASSERT( this->m_available_nodes_id.size() ==
                       this->m_available_nodes_info.size() );

        SIXTRL_ASSERT( this->m_available_devices.size() ==
                       this->m_available_nodes_id.size() );

        this->doInitDefaultProgramsBaseImpl();

        if( ( node_index < this->numAvailableNodes() ) &&
            ( this->doSelectNodeBaseImpl( node_index ) ) )
        {
            this->doInitDefaultKernelsBaseImpl();
            this->doAssignSlotSizeArgBaseImpl( st::BUFFER_DEFAULT_SLOT_SIZE );
            this->doAssignStatusFlagsArgBaseImpl(
                this->internalStatusFlagsBuffer() );
        }
    }

    ClContextBase::ClContextBase(
        _this_t::node_id_t const node_id,
        const char *const SIXTRL_RESTRICT config_str ) :
        m_cl_programs(),
        m_cl_kernels(),
        m_cl_buffers(),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_available_devices(),
        m_program_data(),
        m_kernel_data(),
        m_default_compile_options(),
        m_config_str(),
        m_cl_context(),
        m_cl_queue(),
        m_cl_success_flag(),
        m_remap_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_remap_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_selected_node_index( int64_t{ -1 } ),
        m_default_kernel_arg( uint64_t{ 0 } ),
        m_debug_mode( false )
    {
        using _this_t = ClContextBase;

        _this_t::UpdateAvailableNodes(
            this->m_available_nodes_id, this->m_available_nodes_info,
            this->m_available_devices, nullptr, this->debugMode() );

        SIXTRL_ASSERT( this->m_available_nodes_id.size() ==
                       this->m_available_nodes_info.size() );

        SIXTRL_ASSERT( this->m_available_devices.size() ==
                       this->m_available_nodes_id.size() );

        this->doSetConfigStr( config_str );

        if( this->configStr() != nullptr )
        {
            this->doParseConfigStringBaseImpl( this->configStr() );
        }

        this->doInitDefaultProgramsBaseImpl();

        size_type const node_index = this->findAvailableNodesIndex(
            NS(ComputeNodeId_get_platform_id)( &node_id ),
            NS(ComputeNodeId_get_device_id)( &node_id ) );

        if( ( node_index < this->numAvailableNodes() ) &&
            ( this->doSelectNodeBaseImpl( node_index ) ) )
        {
            this->doInitDefaultKernelsBaseImpl();
            this->doAssignSlotSizeArgBaseImpl( st::BUFFER_DEFAULT_SLOT_SIZE );
            this->doAssignStatusFlagsArgBaseImpl(
                this->internalStatusFlagsBuffer() );
        }
    }

    ClContextBase::ClContextBase(
        char const* node_id_str,
        const char *const SIXTRL_RESTRICT config_str ) :
        m_cl_programs(),
        m_cl_kernels(),
        m_cl_buffers(),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_available_devices(),
        m_program_data(),
        m_kernel_data(),
        m_default_compile_options(),
        m_config_str(),
        m_cl_context(),
        m_cl_queue(),
        m_cl_success_flag(),
        m_remap_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_remap_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_selected_node_index( int64_t{ -1 } ),
        m_default_kernel_arg( uint64_t{ 0 } ),
        m_debug_mode( false )
    {
        using _this_t = ClContextBase;

        _this_t::UpdateAvailableNodes(
            this->m_available_nodes_id, this->m_available_nodes_info,
            this->m_available_devices, nullptr, this->debugMode() );

        SIXTRL_ASSERT( this->m_available_nodes_id.size() ==
                       this->m_available_nodes_info.size() );

        SIXTRL_ASSERT( this->m_available_devices.size() ==
                       this->m_available_nodes_id.size() );

        this->doSetConfigStr( config_str );

        if( this->configStr() != nullptr )
        {
            this->doParseConfigStringBaseImpl( this->configStr() );
        }

        this->doInitDefaultProgramsBaseImpl();

        size_type const node_index =
            this->findAvailableNodesIndex( node_id_str );

        if( ( node_index < this->numAvailableNodes() ) &&
            ( this->doSelectNodeBaseImpl( node_index ) ) )
        {
            this->doInitDefaultKernelsBaseImpl();
            this->doAssignSlotSizeArgBaseImpl( st::BUFFER_DEFAULT_SLOT_SIZE );
            this->doAssignStatusFlagsArgBaseImpl(
                this->internalStatusFlagsBuffer() );
        }
    }

    ClContextBase::ClContextBase(
        _this_t::platform_id_t const platform_idx,
        _this_t::device_id_t const device_idx,
        const char *const SIXTRL_RESTRICT config_str ) :
        m_cl_programs(),
        m_cl_kernels(),
        m_cl_buffers(),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_available_devices(),
        m_program_data(),
        m_kernel_data(),
        m_default_compile_options(),
        m_config_str(),
        m_cl_context(),
        m_cl_queue(),
        m_cl_success_flag(),
        m_remap_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_remap_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_selected_node_index( int64_t{ -1 } ),
        m_default_kernel_arg( uint64_t{ 0 } ),
        m_debug_mode( false )
    {
        using _this_t = ClContextBase;

        _this_t::UpdateAvailableNodes(
            this->m_available_nodes_id, this->m_available_nodes_info,
            this->m_available_devices, nullptr, this->debugMode() );

        SIXTRL_ASSERT( this->m_available_nodes_id.size() ==
                       this->m_available_nodes_info.size() );

        SIXTRL_ASSERT( this->m_available_devices.size() ==
                       this->m_available_nodes_id.size() );

        this->doSetConfigStr( config_str );

        if( this->configStr() != nullptr )
        {
            this->doParseConfigStringBaseImpl( this->configStr() );
        }

        this->doInitDefaultProgramsBaseImpl();

        size_type const node_index =
            this->findAvailableNodesIndex( platform_idx, device_idx );

        if( ( node_index < this->numAvailableNodes() ) &&
            ( this->doSelectNodeBaseImpl( node_index ) ) )
        {
            this->doInitDefaultKernelsBaseImpl();
            this->doAssignSlotSizeArgBaseImpl( st::BUFFER_DEFAULT_SLOT_SIZE );
            this->doAssignStatusFlagsArgBaseImpl(
                this->internalStatusFlagsBuffer() );
        }
    }

    ClContextBase::~ClContextBase() SIXTRL_NOEXCEPT
    {
        if( !this->m_available_nodes_info.empty() )
        {
            for( auto& nodes_info : this->m_available_nodes_info )
            {
                NS(ComputeNodeInfo_free)( &nodes_info );
            }

            this->m_available_nodes_info.clear();
        }
    }

    cl::Buffer const& ClContextBase::internalStatusFlagsBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_cl_success_flag;
    }

    cl::Buffer& ClContextBase::internalStatusFlagsBuffer() SIXTRL_NOEXCEPT
    {
        return this->m_cl_success_flag;
    }

    _this_t::size_type ClContextBase::numAvailableNodes() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_available_nodes_id.size() ==
                       this->m_available_nodes_info.size() );

        return this->m_available_nodes_id.size();
    }

    _this_t::node_info_t const*
    ClContextBase::availableNodesInfoBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_available_nodes_info.data();
    }

    _this_t::node_info_t const*
    ClContextBase::availableNodesInfoEnd()   const SIXTRL_NOEXCEPT
    {
        _this_t::node_info_t const* ptr_end =
            this->availableNodesInfoBegin();

        if( ptr_end != nullptr )
        {
            std::advance( ptr_end, this->numAvailableNodes() );
        }

        return ptr_end;
    }

    _this_t::node_info_t const*
    ClContextBase::defaultNodeInfo() const SIXTRL_NOEXCEPT
    {
        return this->availableNodesInfoBegin();
    }

    _this_t::node_id_t ClContextBase::defaultNodeId() const SIXTRL_NOEXCEPT
    {
        NS(ComputeNodeId) default_node_id;

        _this_t::node_info_t const* default_node_info = this->defaultNodeInfo();
        NS(ComputeNodeId_preset)( &default_node_id );

        if( default_node_info != nullptr )
        {
            default_node_id = default_node_info->id;
        }

        return default_node_id;
    }

    bool ClContextBase::isNodeIndexAvailable(
         _this_t::size_type const node_index ) const SIXTRL_RESTRICT
    {
        return ( node_index < this->numAvailableNodes() );
    }

    bool ClContextBase::isNodeIdAvailable(
        _this_t::node_id_t const node_id ) const SIXTRL_NOEXCEPT
    {
        platform_id_t const platform_index =
            NS(ComputeNodeId_get_platform_id)( &node_id );

        device_id_t const device_index =
            NS(ComputeNodeId_get_device_id)( &node_id );

        return ( this->numAvailableNodes() >
                 this->findAvailableNodesIndex( platform_index, device_index ) );
    }

    bool ClContextBase::isNodeIdAvailable(
        _this_t::platform_id_t const platform_index,
        _this_t::device_id_t  const device_index ) const SIXTRL_NOEXCEPT
    {
        return ( this->numAvailableNodes() >
                 this->findAvailableNodesIndex( platform_index, device_index ) );
    }

    bool ClContextBase::isNodeIdAvailable( char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        return ( this->numAvailableNodes() >
                 this->findAvailableNodesIndex( node_id_str ) );
    }

    bool ClContextBase::isDefaultNode(
        char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        node_id_t const default_node_id = this->defaultNodeId();

        return ( NS(ComputeNodeId_are_equal)(
            this->ptrAvailableNodesId( node_id_str ), &default_node_id ) );
    }

    bool ClContextBase::isDefaultNode(
        _this_t::node_id_t const node_id ) const SIXTRL_NOEXCEPT
    {
        node_id_t const default_node_id = this->defaultNodeId();
        return ( NS(ComputeNodeId_are_equal)( &node_id, &default_node_id ) );
    }

    bool ClContextBase::isDefaultNode(
        _this_t::size_type const node_index ) const SIXTRL_NOEXCEPT
    {
        node_id_t const default_node_id = this->defaultNodeId();

        return ( NS(ComputeNodeId_are_equal)(
            this->ptrAvailableNodesId( node_index ), &default_node_id ) );
    }

    bool ClContextBase::isDefaultNode(
        _this_t::platform_id_t const platform_index,
        _this_t::device_id_t const device_index ) const SIXTRL_NOEXCEPT
    {
        node_id_t const default_node_id = this->defaultNodeId();

        return ( NS(ComputeNodeId_are_equal)(
            this->ptrAvailableNodesId( platform_index, device_index ),
                                       &default_node_id ) );
    }

    _this_t::node_id_t const* ClContextBase::ptrAvailableNodesId(
        _this_t::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_id[ index ] : nullptr;
    }

    _this_t::node_id_t const* ClContextBase::ptrAvailableNodesId(
        _this_t::platform_id_t const platform_index,
        _this_t::device_id_t   const device_index ) const SIXTRL_NOEXCEPT
    {
        size_type const index =
            this->findAvailableNodesIndex( platform_index, device_index );

        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_id[ index ] : nullptr;
    }

    _this_t::node_id_t const* ClContextBase::ptrAvailableNodesId(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        size_type const index = this->findAvailableNodesIndex( node_id_str );

        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_id[ index ] : nullptr;
    }

    _this_t::node_info_t const* ClContextBase::ptrAvailableNodesInfo(
        size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_info[ index ] : nullptr;
    }

    _this_t::node_info_t const* ClContextBase::ptrAvailableNodesInfo(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        size_type const index = this->findAvailableNodesIndex( node_id_str );

        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_info[ index ] : nullptr;
    }

    _this_t::node_info_t const* ClContextBase::ptrAvailableNodesInfo(
        _this_t::node_id_t const node_id ) const SIXTRL_NOEXCEPT
    {
        size_type const index = this->findAvailableNodesIndex(
                                    NS(ComputeNodeId_get_platform_id)( &node_id ),
                                    NS(ComputeNodeId_get_device_id)( &node_id ) );

        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_info[ index ] : nullptr;
    }

    bool ClContextBase::isAvailableNodeAMDPlatform(
        _this_t::size_type const index ) const SIXTRL_NOEXCEPT
    {
        bool is_amd_platform = false;

        _this_t::node_info_t const* node_info =
            this->ptrAvailableNodesInfo( index );

        if( node_info != nullptr )
        {
            char _temp[ 5 ] = { '\0', '\0', '\0', '\0', '\0' };

            std::strncpy( &_temp[ 0 ], ::NS(ComputeNodeInfo_get_platform)(
                node_info ), 5u );

            std::transform( &_temp[ 0 ], &_temp[ 5 ], &_temp[ 0 ],
                [](unsigned char c){ return std::tolower(c); } );

            is_amd_platform = ( 0 == std::strncmp( &_temp[ 0 ], "amd ", 4u ) );
        }

        return is_amd_platform;
    }

    bool ClContextBase::hasSelectedNode() const SIXTRL_NOEXCEPT
    {
        return ( ( this->m_selected_node_index >= 0 ) &&
                 ( this->numAvailableNodes() > static_cast< size_type >(
                       this->m_selected_node_index ) ) );
    }

    cl::Device const* ClContextBase::selectedNodeDevice() const SIXTRL_NOEXCEPT
    {
        cl::Device const* ptr_selected_device = nullptr;

        if( this->hasSelectedNode() )
        {
            SIXTRL_ASSERT( this->numAvailableNodes() > size_type{ 0 } );
            SIXTRL_ASSERT( this->m_available_devices.size() ==
                           this->numAvailableNodes() );

            ptr_selected_device =
                &this->m_available_devices[ this->m_selected_node_index ];

        }

        return ptr_selected_device;
    }

    cl::Device* ClContextBase::selectedNodeDevice() SIXTRL_NOEXCEPT
    {
        return const_cast< cl::Device* >(
            static_cast< ClContextBase const& >( *this ).selectedNodeDevice() );
    }

    _this_t::node_id_t const*
    ClContextBase::ptrSelectedNodeId() const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesId( static_cast< size_type >(
            this->m_selected_node_index ) );
    }

    _this_t::node_info_t const*
    ClContextBase::ptrSelectedNodeInfo() const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesInfo( static_cast< size_type >(
            this->m_selected_node_index ) );
    }

    _this_t::size_type ClContextBase::selectedNodeIndex() const SIXTRL_NOEXCEPT
    {
        using size_t = _this_t::size_type;

        return ( this->hasSelectedNode() )
            ? static_cast< size_t >( this->m_selected_node_index )
            : this->numAvailableNodes();
    }

    std::string ClContextBase::selectedNodeIdStr() const SIXTRL_NOEXCEPT
    {
        char node_id_str[ 32 ];
        std::memset( &node_id_str[ 0 ], ( int )'\0', 32 );

        if( this->selectedNodeIdStr( &node_id_str[ 0 ], 32 ) )
        {
            return std::string( node_id_str );
        }

        return std::string{ "" };
    }

    bool ClContextBase::selectedNodeIdStr( char* SIXTRL_RESTRICT node_id_str,
        _this_t::size_type const max_str_length ) const SIXTRL_NOEXCEPT
    {
        return ( 0 == NS(ComputeNodeId_to_string)(
            this->ptrSelectedNodeId(), node_id_str, max_str_length ) );
    }

    bool ClContextBase::selectNode( size_type const node_index )
    {
        bool success = false;

        if( this->doSelectNode( node_index ) )
        {
            success = this->doInitDefaultKernels();

            if( success )
            {
                success = (
                ( this->doAssignSlotSizeArg(
                    st::BUFFER_DEFAULT_SLOT_SIZE ) ==
                        st::ARCH_STATUS_SUCCESS ) &&
                ( this->doAssignStatusFlagsArg(
                    this->internalStatusFlagsBuffer() ) ==
                        st::ARCH_STATUS_SUCCESS ) );
            }
        }

        return success;
    }

    bool ClContextBase::selectNode( node_id_t const node_id )
    {
        bool success = false;

        platform_id_t const platform_idx =
            NS(ComputeNodeId_get_platform_id)( &node_id );

        device_id_t const device_idx =
            NS(ComputeNodeId_get_device_id)( &node_id );

        if( this->doSelectNode( this->findAvailableNodesIndex(
                platform_idx, device_idx ) ) )
        {
            success = this->doInitDefaultKernels();
        }

        return success;
    }

    bool ClContextBase::selectNode(
         _this_t::platform_id_t const platform_idx,
         _this_t::device_id_t   const device_idx )
    {
        bool success = false;

        if( this->doSelectNode( this->findAvailableNodesIndex(
                platform_idx, device_idx ) ) )
        {
            success = this->doInitDefaultKernels();
        }

        return success;
    }

    bool ClContextBase::selectNode( char const* node_id_str )
    {
        bool success = false;

        if( this->doSelectNode( this->findAvailableNodesIndex( node_id_str ) ) )
        {
            success = this->doInitDefaultKernels();
        }

        return success;
    }

    bool ClContextBase::doSelectNode( size_type const node_index )
    {
        return this->doSelectNodeBaseImpl( node_index );
    }

    _this_t::status_t ClContextBase::doSetStatusFlags(
        _this_t::status_flag_t const value )
    {
        _this_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        cl::CommandQueue* queue = this->openClQueue();

        if( ( queue != nullptr ) &&
            ( this->doGetPtrLocalStatusFlags() != nullptr ) )
        {
            *this->doGetPtrLocalStatusFlags() = value;

            cl_int const ret = queue->enqueueWriteBuffer(
                this->internalStatusFlagsBuffer(), CL_TRUE, 0,
                    sizeof( _this_t::status_flag_t ),
                        this->doGetPtrLocalStatusFlags() );

            if( ret == CL_SUCCESS )
            {
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    _this_t::status_t ClContextBase::doFetchStatusFlags(
        _this_t::status_flag_t* SIXTRL_RESTRICT ptr_status_flags )
    {
        _this_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        cl::CommandQueue* queue = this->openClQueue();

        if( ( queue != nullptr ) &&
            ( this->doGetPtrLocalStatusFlags() != nullptr ) )
        {
            *this->doGetPtrLocalStatusFlags() =
                st::ARCH_DEBUGGING_GENERAL_FAILURE;

            cl_int const ret = queue->enqueueReadBuffer(
                this->internalStatusFlagsBuffer(), CL_TRUE, 0,
                    sizeof( _this_t::status_flag_t ),
                        this->doGetPtrLocalStatusFlags() );

            if( ret == CL_SUCCESS )
            {
                status = ::NS(DebugReg_get_stored_arch_status)(
                    *this->doGetPtrLocalStatusFlags() );
            }

            queue->finish();
        }

        return status;
    }

    _this_t::status_t ClContextBase::doAssignStatusFlagsArg(
        cl::Buffer& SIXTRL_RESTRICT_REF status_flags_arg )
    {
        return this->doAssignStatusFlagsArgBaseImpl( status_flags_arg );
    }


    _this_t::status_t ClContextBase::doAssignStatusFlagsArgBaseImpl(
            cl::Buffer& SIXTRL_RESTRICT_REF status_flags_arg )
    {
        using kernel_id_t = _this_t::kernel_id_t;
        using size_t = _this_t::size_type;

        _this_t::status_t status = st::ARCH_STATUS_SUCCESS;

        if( this->has_remapping_kernel() )
        {
             kernel_id_t const kernel_id = this->remapping_kernel_id();

            if( ( kernel_id != st::ARCH_ILLEGAL_KERNEL_ID ) &&
                ( kernel_id >= kernel_id_t{ 0 } ) && ( kernel_id <
                    static_cast< size_t >( this->numAvailableKernels() ) ) )
            {
                size_t const num_args = this->kernelNumArgs( kernel_id );

                if( num_args >= _this_t::MIN_NUM_REMAP_BUFFER_ARGS )
                {
                    if( ( this->debugMode() ) &&
                        ( num_args > _this_t::MIN_NUM_REMAP_BUFFER_ARGS ) )
                    {
                        SIXTRL_ASSERT( num_args > size_t{ 1 } );

                        this->assignKernelArgumentClBuffer( kernel_id,
                            num_args - size_t{ 1 }, status_flags_arg );
                    }
                    else if( this->debugMode() )
                    {
                        status = st::ARCH_STATUS_GENERAL_FAILURE;
                    }
                }
                else if( num_args < _this_t::MIN_NUM_REMAP_BUFFER_ARGS )
                {
                    status = st::ARCH_STATUS_GENERAL_FAILURE;
                }
            }
        }

        return status;
    }

    _this_t::status_t ClContextBase::doAssignSlotSizeArg(
        _this_t::size_type const slot_size )
    {
         return this->doAssignSlotSizeArgBaseImpl( slot_size );
    }

    _this_t::status_t ClContextBase::doAssignSlotSizeArgBaseImpl(
        _this_t::size_type const slot_size )
    {
        using size_t = _this_t::size_type;
        _this_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( slot_size > size_t{ 0 } )
        {
            status = st::ARCH_STATUS_SUCCESS;

            if( this->has_remapping_kernel() )
            {
                _this_t::kernel_id_t const kernel_id =
                    this->remapping_kernel_id();

                if( ( kernel_id != st::ARCH_ILLEGAL_KERNEL_ID ) &&
                    ( kernel_id >= _this_t::kernel_id_t{ 0 } ) &&
                    ( kernel_id <  static_cast< size_t >(
                            this->numAvailableKernels() ) ) )
                {
                    if( this->kernelNumArgs( kernel_id ) >=
                        _this_t::MIN_NUM_REMAP_BUFFER_ARGS )
                    {
                        uint64_t const slot_size_arg =
                            static_cast< uint64_t >( slot_size );

                        this->assignKernelArgumentValue(
                            kernel_id, size_t{ 1 }, slot_size_arg );
                    }
                    else
                    {
                        status = st::ARCH_STATUS_GENERAL_FAILURE;
                    }
                }
            }
        }

        return status;
    }

    void ClContextBase::doSetConfigStr(
        const char *const SIXTRL_RESTRICT config_str )
    {
        if( ( config_str != nullptr ) &&
            ( std::strlen( config_str ) > _this_t::size_type{ 0 } ) )
        {
            this->m_config_str = std::string{ config_str };
        }
        else
        {
            this->m_config_str.clear();
        }

        return;
    }

    bool ClContextBase::doSelectNodeBaseImpl( size_type const index )
    {
        bool success = false;

        if( ( !this->hasSelectedNode() ) &&
            ( index < this->numAvailableNodes() ) )
        {
            typedef NS(arch_debugging_t) status_flag_t;

            SIXTRL_ASSERT( this->m_cl_programs.empty() );
            SIXTRL_ASSERT( this->m_selected_node_index < int64_t{ 0 } );
            SIXTRL_ASSERT( this->m_available_devices.size() ==
                           this->numAvailableNodes() );

            cl::Device  device  = this->m_available_devices[ index ];
            cl::Context context( device );
            cl::CommandQueue queue( context, device,
                                    CL_QUEUE_PROFILING_ENABLE );

            this->m_cl_success_flag = cl::Buffer(
                context, CL_MEM_READ_WRITE, sizeof( status_flag_t ), nullptr );

            status_flag_t init_success_flag = status_flag_t{ 0 };

            cl_int cl_ret = queue.enqueueWriteBuffer(
                this->m_cl_success_flag, true, size_type{ 0 },
                sizeof( init_success_flag ), &init_success_flag );

            success = ( cl_ret == CL_SUCCESS );

            if( success )
            {
                this->m_cl_context = context;
                this->m_cl_queue   = queue;

                this->m_cl_programs.clear();
                this->m_cl_kernels.clear();
                this->m_kernel_data.clear();

                this->m_selected_node_index = index;
            }

            if( ( success ) && ( !this->m_program_data.empty() ) )
            {
                for( auto& program_data : this->m_program_data )
                {
                    program_data.m_kernels.clear();
                    program_data.m_compiled = false;
                    program_data.m_compile_report.clear();

                    this->m_cl_programs.emplace_back(
                        this->m_cl_context, program_data.m_source_code );

                    SIXTRL_ASSERT( this->m_cl_programs.size() <=
                                   this->m_program_data.size() );

                    success &= this->doCompileProgramBaseImpl(
                        this->m_cl_programs.back(), program_data );
                }
            }
        }

        return success;
    }

    void ClContextBase::printNodesInfo() const SIXTRL_NOEXCEPT
    {
        if( this->numAvailableNodes() > size_type{ 0 } )
        {
            node_id_t const default_node_id = this->defaultNodeId();

            auto node_it  = this->availableNodesInfoBegin();
            auto node_end = this->availableNodesInfoEnd();

            for( ; node_it != node_end ; ++node_it )
            {
                ::NS(ComputeNodeInfo_print_out)( node_it, &default_node_id );
            }
        }
        else
        {
            printf( "No OpenCL Devices found\r\n" );
        }

        return;
    }

    char const* ClContextBase::configStr() const SIXTRL_NOEXCEPT
    {
        return this->m_config_str.c_str();
    }

    void ClContextBase::clear()
    {
        this->doClear();
        this->doInitDefaultPrograms();
        return;
    }

    void ClContextBase::setDefaultCompileOptions(
        std::string const& compile_options_str )
    {
        this->setDefaultCompileOptions( compile_options_str.c_str() );
        return;
    }


    void ClContextBase::setDefaultCompileOptions( char const* compile_options_str )
    {
        SIXTRL_ASSERT( compile_options_str != nullptr );
        this->m_default_compile_options = compile_options_str;
        return;
    }

    char const* ClContextBase::defaultCompileOptions() const SIXTRL_NOEXCEPT
    {
        return this->m_default_compile_options.c_str();
    }

    _this_t::program_id_t ClContextBase::addProgramCode(
        std::string const& source_code )
    {
        return this->addProgramCode(
            source_code.c_str(), this->defaultCompileOptions() );
    }

    _this_t::program_id_t ClContextBase::addProgramCode( char const* source_code )
    {
        return this->addProgramCode(
            source_code, this->defaultCompileOptions() );
    }

    _this_t::program_id_t ClContextBase::addProgramCode(
        std::string const& source_code, std::string const& compile_options )
    {
        _this_t::program_id_t program_id = st::ARCH_ILLEGAL_PROGRAM_ID;

        if( !source_code.empty() )
        {
            program_id = this->m_program_data.size();

            this->m_program_data.emplace_back();
            this->m_program_data.back().m_source_code = source_code;
            this->m_program_data.back().m_compile_options = compile_options;

            if( this->hasSelectedNode() )
            {
                this->m_cl_programs.emplace_back(
                    this->m_cl_context, source_code );

                SIXTRL_ASSERT( this->m_cl_programs.size() ==
                               this->m_program_data.size() );

                this->doCompileProgram(
                    this->m_cl_programs.back(), this->m_program_data.back() );
            }
            else
            {
                this->m_program_data.back().m_compiled = false;
                SIXTRL_ASSERT( this->m_cl_programs.empty() );
            }
        }

        return program_id;
    }

    _this_t::program_id_t
    ClContextBase::addProgramCode(
        char const* source_code, char const* compile_options )
    {
        std::string const str_source_code( ( source_code != nullptr )
                                           ? std::string( source_code ) : std::string() );

        std::string const str_compile_options( ( compile_options != nullptr )
                                               ? std::string( compile_options )
                                               : this->m_default_compile_options );

        return this->addProgramCode( str_source_code, str_compile_options );
    }

    _this_t::program_id_t
    ClContextBase::addProgramFile( std::string const& path_to_program )
    {
        return this->addProgramFile(
                   path_to_program, this->m_default_compile_options );
    }

    _this_t::program_id_t
    ClContextBase::addProgramFile( char const* path_to_program )
    {
        return ( path_to_program != nullptr )
               ? this->addProgramFile( std::string( path_to_program ) )
               : st::ARCH_ILLEGAL_PROGRAM_ID;
    }

    _this_t::program_id_t
    ClContextBase::addProgramFile(
        std::string const& path_to_program, std::string const& compile_options )
    {
        std::fstream kernel_file( path_to_program, std::ios::in );

        if( kernel_file.is_open() )
        {
            std::string const source_code(
                ( std::istreambuf_iterator< char >( kernel_file ) ),
                std::istreambuf_iterator< char >() );

            if( !source_code.empty() )
            {
                program_id_t program_id = this->addProgramCode(
                                              source_code, compile_options );

                if( program_id >= program_id_t { 0 } )
                {
                    this->m_program_data[ program_id ].m_file_path =
                        path_to_program;
                }

                return program_id;
            }
        }

        return st::ARCH_ILLEGAL_PROGRAM_ID;
    }

    _this_t::program_id_t
    ClContextBase::addProgramFile(
        char const* path_to_program, char const* compile_options )
    {
        std::string options_str = ( compile_options != nullptr )
                                  ? std::string( compile_options )
                                  : this->m_default_compile_options;

        return ( path_to_program != nullptr )
               ? this->addProgramFile( std::string( path_to_program ), options_str )
               : st::ARCH_ILLEGAL_PROGRAM_ID;
    }

    bool ClContextBase::compileProgram(
        _this_t::program_id_t const program_id )
    {
        bool success = false;

        if( ( this->hasSelectedNode() ) &&
            ( program_id >= program_id_t{ 0 } ) &&
            ( static_cast< size_type >( program_id ) <
              this->m_program_data.size() ) &&
            ( this->m_cl_programs.size() ==
              this->m_program_data.size() ) )
        {
            success = this->doCompileProgram(
                this->m_cl_programs[ program_id ],
                this->m_program_data[ program_id ] );
        }

        return success;
    }

    char const* ClContextBase::programSourceCode(
        _this_t::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) )
            ? this->m_program_data[ program_id ].m_source_code.c_str()
            : nullptr;
    }

    bool ClContextBase::programHasFilePath(
         _this_t::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) &&
                 ( !this->m_program_data[ program_id ].m_file_path.empty() ) );
    }

    char const* ClContextBase::programPathToFile(
        _this_t::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) )
            ? this->m_program_data[ program_id ].m_file_path.c_str()
            : nullptr;
    }

    char const* ClContextBase::programCompileOptions(
        _this_t::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) )
            ? this->m_program_data[ program_id ].m_compile_options.c_str()
            : nullptr;
    }

    char const* ClContextBase::programCompileReport(
        _this_t::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) )
            ? this->m_program_data[ program_id ].m_compile_report.c_str()
            : nullptr;
    }

    bool ClContextBase::isProgramCompiled(
        _this_t::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) )
            ? this->m_program_data[ program_id ].m_compiled : false;
    }

    _this_t::size_type
    ClContextBase::numAvailablePrograms() const SIXTRL_NOEXCEPT
    {
        return this->m_program_data.size();
    }

    _this_t::kernel_id_t ClContextBase::enableKernel(
        std::string const& kernel_name,
        _this_t::program_id_t const program_id )
    {
        return this->enableKernel( kernel_name.c_str(), program_id );
    }

    _this_t::kernel_id_t
    ClContextBase::enableKernel(
        char const* kernel_name, program_id_t const program_id )
    {
        kernel_id_t kernel_id = st::ARCH_ILLEGAL_KERNEL_ID;

        if( ( this->hasSelectedNode() ) && ( kernel_name != nullptr  ) &&
            ( std::strlen( kernel_name ) > 0u ) &&
            ( program_id >= program_id_t { 0 } ) &&
            ( static_cast< size_type >( program_id ) <
                this->m_cl_programs.size() ) )
        {
            SIXTRL_ASSERT( this->m_cl_programs.size() ==
                           this->m_program_data.size() );

            program_data_t& program_data = this->m_program_data[ program_id ];

            bool add_kernel = false;
            bool program_compiled = this->isProgramCompiled( program_id );

            if( !program_compiled )
            {
                add_kernel = program_compiled =
                    this->compileProgram( program_id );
            }

            if( ( !add_kernel ) && ( program_compiled ) )
            {
                add_kernel = true;

                auto it  = program_data.m_kernels.begin();
                auto end = program_data.m_kernels.end();

                kernel_id_t const num_kernels = this->m_cl_kernels.size();

                SIXTRL_ASSERT( this->m_cl_kernels.size() ==
                               this->m_kernel_data.size() );

                for( ; it != end ; ++it )
                {
                    kernel_id_t const test_kernel_id = *it;

                    if( (  test_kernel_id >= kernel_id_t{ 0 } ) &&
                        (  test_kernel_id <  num_kernels ) &&
                        ( !this->m_kernel_data[ test_kernel_id
                            ].m_kernel_name.empty() ) &&
                        (  this->m_kernel_data[ test_kernel_id
                            ].m_kernel_name.compare( kernel_name ) == 0 ) )
                    {
                        add_kernel = false;
                        break;
                    }
                }
            }

            if( ( add_kernel ) && ( program_compiled ) )
            {
                size_type const num_kernels_so_far = this->m_cl_kernels.size();

                kernel_id = static_cast< kernel_id_t >( num_kernels_so_far );
                SIXTRL_ASSERT( this->m_kernel_data.size() == num_kernels_so_far );
                SIXTRL_ASSERT( kernel_id >= kernel_id_t{ 0 } );

                cl::Kernel kernel(
                    this->m_cl_programs[ program_id ], kernel_name );

                this->m_cl_kernels.push_back( kernel );
                this->m_kernel_data.emplace_back();

                this->m_kernel_data.back().m_kernel_name = kernel_name;
                this->m_kernel_data.back().m_program_id  = program_id;

                cl::Device& selected_device = this->m_available_devices.at(
                    this->m_selected_node_index );

                size_type const num_kernel_args =
                    kernel.getInfo< CL_KERNEL_NUM_ARGS >();

                size_type const max_work_group_size = kernel.getWorkGroupInfo<
                    CL_KERNEL_WORK_GROUP_SIZE >( selected_device );

                size_type const pref_work_group_size = kernel.getWorkGroupInfo<
                    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE >(
                        selected_device );

                size_type const loc_mem_size = kernel.getWorkGroupInfo<
                    CL_KERNEL_LOCAL_MEM_SIZE >( selected_device );

                SIXTRL_ASSERT( num_kernel_args  >= size_type{  0 } );
                SIXTRL_ASSERT( num_kernel_args  <  size_type{ 32 } );
                SIXTRL_ASSERT( pref_work_group_size > size_type{ 0 } );
                SIXTRL_ASSERT( max_work_group_size  >= pref_work_group_size );
                SIXTRL_ASSERT( max_work_group_size  <= size_type{ 65535 } );

                this->m_kernel_data.back().resetArguments( num_kernel_args );

                this->m_kernel_data.back().m_work_group_size = size_type{ 0 };

                this->m_kernel_data.back().m_max_work_group_size =
                    max_work_group_size;

                this->m_kernel_data.back().m_preferred_work_group_multiple =
                    pref_work_group_size;

                this->m_kernel_data.back().m_local_mem_size = loc_mem_size;
                program_data.m_kernels.push_back( kernel_id );
            }
        }

        return kernel_id;
    }

    _this_t::kernel_id_t ClContextBase::findKernelByName(
        char const* SIXTRL_RESTRICT kernel_name ) const SIXTRL_NOEXCEPT
    {
        kernel_id_t kernel_id = st::ARCH_ILLEGAL_KERNEL_ID;

        SIXTRL_ASSERT( this->m_kernel_data.size() ==
                       this->m_cl_kernels.size() );

        if( ( !this->m_kernel_data.empty() ) && ( kernel_name != nullptr ) &&
            ( std::strlen( kernel_name ) > 0u ) )
        {
            kernel_id_t const num_kernels = this->m_kernel_data.size();
            kernel_id_t cmp_kernel_id = kernel_id_t{ 0 };

            for( ; cmp_kernel_id < num_kernels ; ++cmp_kernel_id )
            {
                kernel_data_t const& kernel_data =
                    this->m_kernel_data[ cmp_kernel_id ];

                program_id_t const kernel_program_id =
                    kernel_data.m_program_id;

                if( ( kernel_program_id < program_id_t{ 0 } ) ||
                    ( static_cast< size_type >( kernel_program_id ) >=
                      this->numAvailablePrograms() ) )
                {
                    continue;
                }

                if( 0 == kernel_data.m_kernel_name.compare( kernel_name ) )
                {
                    kernel_id = cmp_kernel_id;
                    break;
                }
            }
        }

        return kernel_id;
    }

    bool ClContextBase::has_remapping_program() const SIXTRL_NOEXCEPT
    {
        return ( ( this->m_remap_program_id >= program_id_t{ 0 } ) &&
                 ( static_cast< size_type >( this->m_remap_program_id ) <
                   this->numAvailablePrograms() ) );
    }

    _this_t::program_id_t
    ClContextBase::remapping_program_id() const SIXTRL_NOEXCEPT
    {
        return ( this->has_remapping_program() )
            ? this->m_remap_program_id : st::ARCH_ILLEGAL_PROGRAM_ID;
    }

    bool ClContextBase::has_remapping_kernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
                 ( this->m_remap_kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( this->m_remap_kernel_id ) <
                   this->m_cl_kernels.size() ) );
    }

    char const* ClContextBase::kernelFunctionName(
        _this_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_cl_kernels.size() ==
                       this->m_kernel_data.size() );

        return ( ( this->hasSelectedNode() ) &&
                 ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->m_kernel_data.size() ) )
            ? this->m_kernel_data[ kernel_id ].m_kernel_name.c_str()
            : nullptr;
    }

    _this_t::size_type ClContextBase::kernelLocalMemSize(
        _this_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_cl_kernels.size() ==
                       this->m_kernel_data.size() );

        return ( ( this->hasSelectedNode() ) &&
                 ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->m_kernel_data.size() ) )
            ? this->m_kernel_data[ kernel_id ].m_local_mem_size
            : size_type{ 0 };
    }

    _this_t::size_type ClContextBase::kernelNumArgs(
        _this_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_cl_kernels.size() ==
                       this->m_kernel_data.size() );

        return ( ( this->hasSelectedNode() ) &&
                 ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->m_kernel_data.size() ) )
            ? this->m_kernel_data[ kernel_id ].m_num_args
            : size_type{ 0 };
    }

    _this_t::size_type ClContextBase::kernelWorkGroupSize(
        _this_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_cl_kernels.size() ==
                       this->m_kernel_data.size() );

        size_type work_group_size = size_type{ 0 };

        if( ( this->hasSelectedNode() ) && ( kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( kernel_id ) < this->m_kernel_data.size() ) )
        {
            kernel_data_t const& kernel_data = this->m_kernel_data[ kernel_id ];

            work_group_size = std::min( kernel_data.m_work_group_size,
                                        kernel_data.m_max_work_group_size );
        }

        return work_group_size;
    }

    _this_t::size_type ClContextBase::kernelMaxWorkGroupSize(
        _this_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_cl_kernels.size() ==
                       this->m_kernel_data.size() );

        return ( ( this->hasSelectedNode() ) &&
                 ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->m_kernel_data.size() ) )
            ? this->m_kernel_data[ kernel_id ].m_max_work_group_size
            : size_type{ 0 };
    }

    bool ClContextBase::setKernelWorkGroupSize(
        _this_t::kernel_id_t const kernel_id,
        _this_t::size_type work_group_size ) SIXTRL_NOEXCEPT
    {
        bool success = false;

        if( ( this->hasSelectedNode() ) &&
            ( kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( kernel_id ) < this->m_kernel_data.size() ) )
        {
            if( this->m_kernel_data[ kernel_id ].m_max_work_group_size >=
                work_group_size )
            {
                this->m_kernel_data[ kernel_id ].m_work_group_size =
                    work_group_size;

                success = true;
            }
        }

        return success;
    }

    _this_t::size_type ClContextBase::kernelPreferredWorkGroupSizeMultiple(
        _this_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_cl_kernels.size() ==
                       this->m_kernel_data.size() );

        return ( ( this->hasSelectedNode() ) &&
                 ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->m_kernel_data.size() ) )
            ? this->m_kernel_data[ kernel_id ].m_preferred_work_group_multiple
            : size_type{ 0 };
    }

    _this_t::program_id_t ClContextBase::programIdByKernelId(
        _this_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_cl_kernels.size() ==
                       this->m_kernel_data.size() );

        return ( ( this->hasSelectedNode() ) &&
                 ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->m_kernel_data.size() ) )
            ? this->m_kernel_data[ kernel_id ].m_program_id
            : st::ARCH_ILLEGAL_PROGRAM_ID;
    }

    _this_t::size_type ClContextBase::kernelExecCounter(
        _this_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->numAvailableKernels() ) )
            ? this->m_kernel_data[ kernel_id ].m_exec_count
            : size_type{ 0 };
    }

    ClArgument* ClContextBase::ptrKernelArgument(
        _this_t::kernel_id_t const kernel_id,
        _this_t::size_type const arg_index ) SIXTRL_NOEXCEPT
    {
        using _this_t = ClContextBase;
        using   ptr_t = ClArgument*;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).ptrKernelArgument( kernel_id, arg_index ) );
    }

    ClArgument const* ClContextBase::ptrKernelArgument(
        _this_t::kernel_id_t const kernel_id,
        _this_t::size_type const arg_index ) const SIXTRL_NOEXCEPT
    {
        ClArgument const* ptr_arg = nullptr;

        if( ( kernel_id >= kernel_id_t{ 0 } ) && ( static_cast< size_type >(
              kernel_id ) < this->numAvailableKernels() ) )
        {
            kernel_data_t const& kernel_data = this->m_kernel_data[ kernel_id ];

            if( kernel_data.m_num_args > arg_index )
            {
                SIXTRL_ASSERT( kernel_data.m_arguments.size() > arg_index );
                ptr_arg = kernel_data.m_arguments[ arg_index ];
            }
        }

        return ptr_arg;
    }

    ClContextBase::kernel_arg_type_t ClContextBase::kernelArgumentType(
        _this_t::kernel_id_t const kernel_id,
        _this_t::size_type const arg_index) const SIXTRL_NOEXCEPT
    {
        return ( ( static_cast< size_t >( kernel_id ) <
                   this->m_kernel_data.size() ) &&
                 ( arg_index < this->m_kernel_data[ kernel_id ].m_num_args ) )
            ? this->m_kernel_data[ kernel_id ].m_arg_types[ arg_index ]
            : SIXTRL_CXX_NAMESPACE::ClContextBase::ARG_TYPE_INVALID;
    }

    void ClContextBase::assignKernelArgument(
        _this_t::kernel_id_t const kernel_id,
        _this_t::size_type const index,
        ClArgument& SIXTRL_RESTRICT_REF arg )
    {
        SIXTRL_ASSERT( arg.context() == this );
        SIXTRL_ASSERT( arg.size() > _this_t::size_type{ 0 } );

        this->m_kernel_data.at( kernel_id ).setKernelArg(
            _this_t::ARG_TYPE_CL_ARGUMENT, index, &arg );

        cl::Kernel* kernel = this->openClKernel( kernel_id );
        if( kernel != nullptr ) kernel->setArg( index, arg.openClBuffer() );

    }

    void ClContextBase::assignKernelArgumentClBuffer(
            _this_t::kernel_id_t const kernel_id,
            _this_t::size_type const arg_index,
            cl::Buffer& SIXTRL_RESTRICT_REF cl_buffer_arg )
    {
        this->m_kernel_data.at( kernel_id ).setKernelArg(
            _this_t::ARG_TYPE_CL_BUFFER, arg_index, &cl_buffer_arg );

        cl::Kernel* kernel = this->openClKernel( kernel_id );
        if( kernel != nullptr ) kernel->setArg( arg_index, cl_buffer_arg );
    }

    void ClContextBase::resetKernelArguments(
        _this_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        if( ( kernel_id >= kernel_id_t{ 0 } ) && ( static_cast< size_type >(
              kernel_id ) < this->numAvailableKernels() ) )
        {
            cl::Kernel* kernel = this->openClKernel( kernel_id );

            kernel_data_t& kernel_data = this->m_kernel_data[ kernel_id ];
            size_type const nn = kernel_data.m_num_args;

            SIXTRL_ASSERT( ( kernel_data.m_arguments.empty() ) ||
                           ( kernel_data.m_arguments.size() >= nn ) );

            if( ( kernel != nullptr ) &&
                ( !kernel_data.m_arguments.empty() ) && ( nn > size_type{ 0 } ) )
            {
                for( size_type ii = 0 ; ii < nn ; ++ii )
                {
                    if( kernel_data.m_arguments[ ii ] != nullptr )
                    {
                        kernel->setArg( ii, this->m_default_kernel_arg );
                    }
                }
            }

            kernel_data.resetArguments( nn );
        }

        return;
    }

    void ClContextBase::resetSingleKernelArgument(
        _this_t::kernel_id_t const kernel_id,
        _this_t::size_type const arg_index ) SIXTRL_NOEXCEPT
    {
        using _this_t = ClContextBase;

        cl::Kernel* kernel = this->openClKernel( kernel_id );
        if( kernel != nullptr ) kernel->setArg(
                arg_index, this->m_default_kernel_arg );

        this->m_kernel_data.at( kernel_id ).setKernelArg(
            _this_t::ARG_TYPE_NONE, arg_index, nullptr );
    }

    _this_t::size_type ClContextBase::calculateKernelNumWorkItems(
        _this_t::kernel_id_t const kernel_id,
        _this_t::size_type const min_num_work_items ) const SIXTRL_NOEXCEPT
    {
        size_type num_threads = min_num_work_items;
        size_type work_group_size = this->kernelWorkGroupSize( kernel_id );

        if( work_group_size == size_type{ 0 } )
        {
            work_group_size =
                this->kernelPreferredWorkGroupSizeMultiple( kernel_id );
        }

        if( work_group_size > size_type{ 0 } )
        {
            size_type const num_blocks = min_num_work_items / work_group_size;
            num_threads = num_blocks * work_group_size;

            if( num_threads < min_num_work_items )
            {
                num_threads += work_group_size;
            }
        }

        return num_threads;
    }

    bool ClContextBase::runKernel(
        _this_t::kernel_id_t const kernel_id,
        _this_t::size_type min_num_work_items )
    {
        return this->runKernel( kernel_id, min_num_work_items,
                         this->kernelWorkGroupSize( kernel_id ) );
    }

    bool ClContextBase::runKernel( _this_t::kernel_id_t const kernel_id,
        _this_t::size_type const min_num_work_items,
        _this_t::size_type work_group_size )
    {
        bool success = false;

        cl::Kernel* kernel = this->openClKernel( kernel_id );
        cl::CommandQueue* ptr_queue = this->openClQueue();

        if( ( kernel != nullptr ) && ( ptr_queue != nullptr ) )
        {
            size_type num_work_items = min_num_work_items;
            cl::NDRange local_size = cl::NullRange;

            if( work_group_size != size_type{ 0 } )
            {
                size_type const num_blocks =
                    ( min_num_work_items + work_group_size - 1 ) / work_group_size;

                num_work_items = num_blocks * work_group_size;
                SIXTRL_ASSERT( num_work_items >= min_num_work_items );
                local_size  = cl::NDRange( work_group_size );
            }

            cl::Event run_event;

            cl_ulong run_when_queued    = cl_ulong{ 0 };
            cl_ulong run_when_submitted = cl_ulong{ 0 };
            cl_ulong run_when_started   = cl_ulong{ 0 };
            cl_ulong run_when_ended     = cl_ulong{ 0 };

            cl_int cl_ret = ptr_queue->enqueueNDRangeKernel( *kernel,
                cl::NullRange, cl::NDRange( num_work_items ),
                    local_size, nullptr, &run_event );

            cl_ret |= ptr_queue->flush();
            run_event.wait();

            success = ( cl_ret == CL_SUCCESS );

            cl_ret  = run_event.getProfilingInfo< cl_ulong >(
                CL_PROFILING_COMMAND_QUEUED, &run_when_queued );

            cl_ret |= run_event.getProfilingInfo< cl_ulong >(
                CL_PROFILING_COMMAND_SUBMIT, &run_when_submitted );

            cl_ret |= run_event.getProfilingInfo< cl_ulong >(
                CL_PROFILING_COMMAND_START, &run_when_started );

            cl_ret |= run_event.getProfilingInfo< cl_ulong >(
                CL_PROFILING_COMMAND_END, &run_when_ended );

            if( cl_ret == CL_SUCCESS )
            {
                double const last_event_time = ( run_when_ended >= run_when_started )
                    ? ( double{ 1e-9 } * static_cast< double >(
                        run_when_ended - run_when_started ) )
                    : double{ 0 };

                this->addKernelExecTime(    last_event_time, kernel_id );
                this->setLastWorkGroupSize( work_group_size, kernel_id );
                this->setLastNumWorkItems(  num_work_items,  kernel_id );
            }
        }

        return success;
    }

    double ClContextBase::lastExecTime(
        _this_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->numAvailableKernels() ) )
            ? this->m_kernel_data[ kernel_id ].m_last_exec_time : double{ 0 };
    }

    double ClContextBase::minExecTime(
        _this_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->numAvailableKernels() ) )
            ? this->m_kernel_data[ kernel_id ].m_min_exec_time : double{ 0 };
    }

    double ClContextBase::maxExecTime(
        _this_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->numAvailableKernels() ) )
            ? this->m_kernel_data[ kernel_id ].m_max_exec_time : double{ 0 };
    }

    double ClContextBase::avgExecTime(
        _this_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->numAvailableKernels() ) )
            ? this->m_kernel_data[ kernel_id ].avgExecTime() : double{ 0 };
    }

    _this_t::size_type ClContextBase::lastExecWorkGroupSize(
        _this_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->numAvailableKernels() ) )
            ? this->m_kernel_data[ kernel_id ].m_last_work_group_size
            : size_type{ 0 };
    }

    _this_t::size_type ClContextBase::lastExecNumWorkItems(
        _this_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->numAvailableKernels() ) )
            ? this->m_kernel_data[ kernel_id ].m_last_num_of_threads
            : size_type{ 0 };
    }

    void ClContextBase::resetKernelExecTiming(
         _this_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        if( ( kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( kernel_id ) <
              this->numAvailableKernels() ) )
        {
            this->m_kernel_data[ kernel_id ].resetTiming();
        }

        return;
    }

    void ClContextBase::addKernelExecTime( double const time,
        _this_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        if( ( kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( kernel_id ) <
              this->numAvailableKernels() ) )
        {
            this->m_kernel_data[ kernel_id ].addExecTime( time );
        }

        return;
    }

    _this_t::kernel_id_t
    ClContextBase::remapping_kernel_id() const SIXTRL_NOEXCEPT
    {
        return ( this->has_remapping_kernel() )
            ? this->m_remap_kernel_id : st::ARCH_ILLEGAL_KERNEL_ID;
    }

    bool ClContextBase::set_remapping_kernel_id(
        _this_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        bool success = false;

        if( ( this->hasSelectedNode() ) &&
            ( kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( kernel_id ) <
              this->m_cl_kernels.size() ) &&
            ( this->m_kernel_data[ kernel_id ].m_program_id >=
              program_id_t{ 0 } ) &&
            ( static_cast< size_type >(
                this->m_kernel_data[ kernel_id ].m_program_id ) <
                this->numAvailablePrograms() ) )
        {
            this->m_remap_kernel_id = kernel_id;
            this->m_remap_program_id =
                this->m_kernel_data[ kernel_id ].m_program_id;

            success = true;
        }

        return success;
    }

    _this_t::size_type
    ClContextBase::numAvailableKernels() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_cl_kernels.size() ==
                       this->m_kernel_data.size() );

        return this->m_cl_kernels.size();
    }

    cl::Program* ClContextBase::openClProgram(
        _this_t::program_id_t const program_id ) SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
        ( this->m_cl_programs.size() >
            static_cast< size_type >( program_id ) ) )
        ? &this->m_cl_programs[ program_id ] : nullptr;
    }

    cl::Kernel* ClContextBase::openClKernel(
        _this_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
            ( this->m_cl_kernels.size() > static_cast< size_type >( kernel_id ) ) )
            ? &this->m_cl_kernels[ kernel_id ] : nullptr;
    }

    cl::CommandQueue* ClContextBase::openClQueue() SIXTRL_NOEXCEPT
    {
        return &this->m_cl_queue;
    }

    cl::Context* ClContextBase::openClContext() SIXTRL_NOEXCEPT
    {
        return &this->m_cl_context;
    }

    bool ClContextBase::debugMode() const  SIXTRL_NOEXCEPT
    {
        return this->m_debug_mode;
    }

    void ClContextBase::enableDebugMode()  SIXTRL_NOEXCEPT
    {
        if( ( !this->debugMode() ) && ( !this->hasSelectedNode() ) )
        {
            this->m_debug_mode = true;
            this->clear();
            this->doInitDefaultPrograms();
        }

        return;
    }

    void ClContextBase::disableDebugMode() SIXTRL_NOEXCEPT
    {
        if( ( this->debugMode() ) && ( !this->hasSelectedNode() ) )
        {
            this->m_debug_mode = false;
            this->clear();
            this->doInitDefaultPrograms();
        }

        return;
    }

    _this_t::status_flag_t ClContextBase::status_flags()
    {
        _this_t::status_t const status = this->doFetchStatusFlags(
            this->doGetPtrLocalStatusFlags() );

        return ( status == st::ARCH_STATUS_SUCCESS )
            ? *this->doGetPtrLocalStatusFlags()
            : st::ARCH_DEBUGGING_GENERAL_FAILURE;
    }

    _this_t::status_flag_t ClContextBase::set_status_flags(
        _this_t::status_flag_t const status_flags )
    {
        return this->doSetStatusFlags( status_flags );
    }

    _this_t::status_t ClContextBase::prepare_status_flags_for_use()
    {
        _this_t::status_t const status = this->doSetStatusFlags(
            st::ARCH_DEBUGGING_REGISTER_EMPTY );

        SIXTRL_ASSERT( ( status != st::ARCH_STATUS_SUCCESS ) ||
            ( ( this->doGetPtrLocalStatusFlags() != nullptr ) &&
              ( *this->doGetPtrLocalStatusFlags() ==
                st::ARCH_DEBUGGING_REGISTER_EMPTY ) ) );

        return status;
    }

    _this_t::status_t ClContextBase::eval_status_flags_after_use()
    {
        _this_t::status_t status = this->doFetchStatusFlags(
            this->doGetPtrLocalStatusFlags() );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            if( ::NS(DebugReg_has_status_flags_set)(
                    *this->doGetPtrLocalStatusFlags() ) )
            {
                status = ::NS(DebugReg_get_stored_arch_status)(
                    *this->doGetPtrLocalStatusFlags() );
            }
            else if( ::NS(DebugReg_has_any_flags_set)(
                *this->doGetPtrLocalStatusFlags() ) )
            {
                status = st::ARCH_STATUS_GENERAL_FAILURE;
            }
        }

        return status;
    }

    _this_t::status_t ClContextBase::assign_slot_size_arg(
        _this_t::size_type const slot_size )
    {
        return this->doAssignSlotSizeArg( slot_size );
    }

    _this_t::status_t ClContextBase::assign_status_flags_arg(
        _this_t::cl_buffer_t& SIXTRL_RESTRICT_REF success_flag_arg )
    {
        return this->doAssignStatusFlagsArg( success_flag_arg );
    }

    /* --------------------------------------------------------------------- */

    _this_t::kernel_data_list_t const&
    ClContextBase::kernelData() const SIXTRL_NOEXCEPT
    {
        return this->m_kernel_data;
    }

    _this_t::program_data_list_t const&
    ClContextBase::programData() const SIXTRL_NOEXCEPT
    {
        return this->m_program_data;
    }

    void ClContextBase::setLastWorkGroupSize(
         _this_t::size_type const work_group_size,
         _this_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        if( ( kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( kernel_id ) <
              this->numAvailableKernels() ) )
        {
            this->m_kernel_data[ kernel_id ].m_last_work_group_size =
                work_group_size;
        }

        return;
    }

    void ClContextBase::setLastNumWorkItems(
        _this_t::size_type const num_work_items,
        _this_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        if( ( kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( kernel_id ) <
              this->numAvailableKernels() ) )
        {
            this->m_kernel_data[ kernel_id ].m_last_num_of_threads =
                num_work_items;
        }

        return;
    }

    _this_t::size_type ClContextBase::findAvailableNodesIndex(
        _this_t::platform_id_t const platform_index,
        _this_t::device_id_t const device_index ) const SIXTRL_NOEXCEPT
    {
        size_type index = this->numAvailableNodes();

        if( ( platform_index >= platform_id_t { 0 } ) &&
                ( device_index   >= device_id_t { 0 } ) )
        {
            index = size_type { 0 };

            for( auto const& cmp_node_id : this->m_available_nodes_id )
            {
                if( ( platform_index ==
                        NS(ComputeNodeId_get_platform_id)( &cmp_node_id ) ) &&
                        ( device_index ==
                          NS(ComputeNodeId_get_device_id)( &cmp_node_id ) ) )
                {
                    break;
                }

                ++index;
            }
        }

        SIXTRL_ASSERT( index <= this->numAvailableNodes() );

        return index;
    }

    _this_t::size_type ClContextBase::findAvailableNodesIndex(
        char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        if( ( node_id_str != nullptr ) && ( std::strlen( node_id_str ) >= 3u ) )
        {
            int temp_platform_index = -1;
            int temp_device_index   = -1;

            int const cnt = std::sscanf( node_id_str, "%d.%d",
                                         &temp_platform_index, &temp_device_index );

            if( cnt == 2 )
            {
                return this->findAvailableNodesIndex(
                           static_cast< platform_id_t >( temp_platform_index ),
                           static_cast< device_id_t   >( temp_device_index ) );
            }
        }

        return this->numAvailableNodes();
    }

    _this_t::status_flag_t*
    ClContextBase::doGetPtrLocalStatusFlags() SIXTRL_NOEXCEPT
    {
        return &this->m_local_status_flags;
    }

    _this_t::status_flag_t const*
    ClContextBase::doGetPtrLocalStatusFlags() const SIXTRL_NOEXCEPT
    {
        return &this->m_local_status_flags;
    }

    void ClContextBase::doParseConfigString(
        const char *const SIXTRL_RESTRICT config_str )
    {
        this->doParseConfigStringBaseImpl( config_str );
        return;
    }

    void ClContextBase::doParseConfigStringBaseImpl(
        const char *const SIXTRL_RESTRICT config_str )
    {
        ( void )config_str;
        return;
    }

    void ClContextBase::doClear()
    {
        this->doClearBaseImpl();
        return;
    }

    void ClContextBase::doClearBaseImpl() SIXTRL_NOEXCEPT
    {
        cl::CommandQueue dummy_queue;
        cl::Context dummy_context;
        cl::Buffer  dummy_success_flag;

        this->m_cl_programs.clear();
        this->m_program_data.clear();

        this->m_cl_kernels.clear();
        this->m_kernel_data.clear();

        this->m_cl_queue            = dummy_queue;
        this->m_cl_context          = dummy_context;
        this->m_cl_success_flag     = dummy_success_flag;
        this->m_selected_node_index = int64_t{ -1 };
        this->m_remap_kernel_id     = st::ARCH_ILLEGAL_KERNEL_ID;

        return;
    }

    bool ClContextBase::doInitDefaultPrograms()
    {
        return this->doInitDefaultProgramsBaseImpl();
    }

    bool ClContextBase::doInitDefaultProgramsBaseImpl()
    {
        bool success = false;

        std::string path_to_remap_kernel_program( NS(PATH_TO_BASE_DIR) );
        path_to_remap_kernel_program += "sixtracklib/opencl/kernels/";

        if( !this->debugMode() )
        {
            path_to_remap_kernel_program += "managed_buffer_remap.cl";
        }
        else
        {
            path_to_remap_kernel_program += "managed_buffer_remap_debug.cl";
        }

        std::string remap_program_compile_options = "-D_GPUCODE=1";
        remap_program_compile_options += " -DSIXTRL_BUFFER_ARGPTR_DEC=__private";
        remap_program_compile_options += " -DSIXTRL_BUFFER_DATAPTR_DEC=__global";
        #if !defined( SIXTRL_DISABLE_BEAM_BEAM )
        remap_program_compile_options += " -DSIXTRL_DISABLE_BEAM_BEAM=1";
        #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */
        remap_program_compile_options += " -I";
        remap_program_compile_options += NS(PATH_TO_SIXTRL_INCLUDE_DIR);

        program_id_t const remap_program_id = this->addProgramFile(
            path_to_remap_kernel_program, remap_program_compile_options );

        if( remap_program_id >= program_id_t{ 0 } )
        {
            this->m_remap_program_id = remap_program_id;
            success = true;
        }

        return success;
    }

    bool ClContextBase::doInitDefaultKernels()
    {
        return this->doInitDefaultKernelsBaseImpl();
    }

    bool ClContextBase::doInitDefaultKernelsBaseImpl()
    {
        bool success = false;

        if( this->hasSelectedNode() )
        {
            if( ( this->m_remap_program_id >= program_id_t{ 0 } ) &&
                ( static_cast< size_type >( this->m_remap_program_id ) <
                  this->numAvailablePrograms() ) )
            {
                std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );

                if( !this->debugMode() )
                {
                    kernel_name += "ManagedBuffer_remap_opencl";
                }
                else
                {
                    kernel_name += "ManagedBuffer_remap_debug_opencl";
                }

                kernel_id_t const remap_kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_remap_program_id );

                if( remap_kernel_id >= kernel_id_t{ 0 } )
                {
                    success = this->set_remapping_kernel_id( remap_kernel_id );
                }
            }
        }

        return success;
    }

    bool ClContextBase::doCompileProgram(
         cl::Program& cl_program, ClContextBase::program_data_t& program_data )
    {
        return this->doCompileProgramBaseImpl( cl_program, program_data );
    }

    bool ClContextBase::doCompileProgramBaseImpl(
         cl::Program& cl_program, ClContextBase::program_data_t& program_data )
    {
        bool success = false;

        if( (  this->hasSelectedNode() ) &&
            ( !program_data.m_compiled ) &&
            (  program_data.m_kernels.empty() ) &&
            ( !program_data.m_compile_options.empty() ) )
        {
            auto& build_device =
                this->m_available_devices.at( this->m_selected_node_index );

            cl_int ret = CL_SUCCESS;
            cl_build_status build_status = CL_BUILD_NONE;

            #if defined( SIXTRL_OPENCL_CXX_ENABLES_HOST_EXCEPTIONS ) && \
                         SIXTRL_OPENCL_CXX_ENABLES_HOST_EXCEPTIONS == 1
            try
            {
            #endif /* OpenCL 1.x C++ Host Exceptions enabled */

                ret = cl_program.build( program_data.m_compile_options.c_str() );
                build_status = cl_program.getBuildInfo<
                        CL_PROGRAM_BUILD_STATUS >( build_device );

            #if defined( SIXTRL_OPENCL_CXX_ENABLES_HOST_EXCEPTIONS ) && \
                         SIXTRL_OPENCL_CXX_ENABLES_HOST_EXCEPTIONS == 1
            }
            catch( cl::Error& e )
            {
                if( ( this->debugMode() ) &&
                    ( e.err() == CL_BUILD_PROGRAM_FAILURE ) )
                {
                    std::string name = build_device.getInfo< CL_DEVICE_NAME >();
                    std::string buildlog = cl_program.getBuildInfo<
                        CL_PROGRAM_BUILD_LOG >( build_device );

                    std::cerr << "Build log for " << name << ":" << std::endl
                              << buildlog << std::endl;
                }

                throw e;
            }
            #endif /* OpenCL 1.x C++ Host Exceptions enabled */

            if( ( build_status != CL_BUILD_NONE ) || ( ret == CL_SUCCESS ) )
            {
                program_data.m_compile_report =
                    cl_program.getBuildInfo< CL_PROGRAM_BUILD_LOG >( build_device );

                if( !program_data.m_compile_report.empty() )
                {
                    program_data.m_compile_report.erase(
                        std::find_if(
                            program_data.m_compile_report.rbegin(),
                            program_data.m_compile_report.rend(),
                            []( int ch ){ return !std::isspace( ch ); } ).base(),
                            program_data.m_compile_report.end() );
                }

                if( !program_data.m_compile_report.empty() )
                {
                    program_data.m_compile_report.erase(
                        program_data.m_compile_report.begin(),
                        std::find_if(
                            program_data.m_compile_report.begin(),
                            program_data.m_compile_report.end(),
                            []( int ch ){ return !std::isspace( ch ); } ) );
                }

                if( ( !program_data.m_compile_report.empty() ) &&
                    (  program_data.m_compile_report.size() == size_type{ 1 } ) &&
                    (  program_data.m_compile_report[ 0 ] == '\0' ) )
                {
                    program_data.m_compile_report.clear();
                }
            }

            if( build_status == CL_BUILD_SUCCESS )
            {
                success = program_data.m_compiled = true;
            }
            else if( build_status == CL_BUILD_ERROR )
            {
                SIXTRL_ASSERT( this->m_selected_node_index >= 0 );
                SIXTRL_ASSERT( this->m_available_devices.size() ==
                               this->numAvailableNodes() );

                program_data.m_compiled = false;
            }
            else
            {
                SIXTRL_ASSERT( build_status == CL_BUILD_NONE );
                program_data.m_compiled = false;
                program_data.m_compile_report.clear();
            }

            if( ( this->debugMode() ) &&
                ( ( !program_data.m_compile_report.empty() ) ||
                  ( !program_data.m_compiled ) ) )
            {
                std::cerr << "program_name    : "
                          << program_data.m_file_path << "\r\n"
                          << "compiled        : "
                          << std::boolalpha   << program_data.m_compiled
                          << std::noboolalpha << "\r\n"
                          << "compile options : "
                          << program_data.m_compile_options << "\r\n"
                          << "compile report  : " << "\r\n"
                          << program_data.m_compile_report
                          << std::endl;
            }
        }

        return success;
    }

    void ClContextBase::UpdateAvailableNodes(
        std::vector< _this_t::node_id_t>& available_nodes_id,
        std::vector< _this_t::node_info_t >& available_nodes_info,
        std::vector< cl::Device >& available_devices,
        const char *const filter_str, bool const debug_mode )
    {
        platform_id_t platform_index = 0;
        device_id_t   device_index   = 0;

        std::vector< cl::Device > devices;
        std::vector< cl::Platform > platforms;

        #if !defined( SIXTRL_OPENCL_CXX_ENABLES_HOST_EXCEPTIONS ) || \
                      SIXTRL_OPENCL_CXX_ENABLES_HOST_EXCEPTIONS == 0
        ( void )debug_mode;
        #endif /* OpenCL 1.x C++ Host Exceptions enabled */

        ( void )filter_str;

        available_nodes_id.clear();
        available_nodes_info.clear();
        available_devices.clear();

        platforms.clear();
        platforms.reserve( 10 );

        devices.clear();
        devices.reserve( 100 );

        cl::Platform::get( &platforms );

        for( auto const& platform : platforms )
        {
            devices.clear();
            device_index = 0;

            std::string platform_name = platform.getInfo< CL_PLATFORM_NAME >();

            #if defined( SIXTRL_OPENCL_CXX_ENABLES_HOST_EXCEPTIONS ) && \
                         SIXTRL_OPENCL_CXX_ENABLES_HOST_EXCEPTIONS == 1
            try
            {
            #endif /* OpenCL 1.x C++ Host Exceptions enabled */

            platform.getDevices( CL_DEVICE_TYPE_ALL, &devices );

            #if defined( SIXTRL_OPENCL_CXX_ENABLES_HOST_EXCEPTIONS ) && \
                         SIXTRL_OPENCL_CXX_ENABLES_HOST_EXCEPTIONS == 1
            }
            catch( cl::Error const& e )
            {
                if( debug_mode )
                {
                    std::cerr << "Error while probing devices for platform "
                              << platform_name << " --> skipping"
                              << std::endl;
                }
            }
            #endif /* OpenCL 1.x C++ Host Exceptions enabled */

            bool added_at_least_one_device = false;

            for( auto const& device : devices )
            {
                std::string name;
                std::string description;

                cl_int ret = device.getInfo( CL_DEVICE_NAME, &name );
                ret |= device.getInfo( CL_DEVICE_EXTENSIONS, &description );

                available_nodes_id.push_back( node_id_t {} );
                node_id_t* ptr_node_id = &available_nodes_id.back();

                NS(ComputeNodeId_set_platform_id)( ptr_node_id, platform_index );
                NS(ComputeNodeId_set_device_id)( ptr_node_id, device_index++ );

                available_nodes_info.push_back( node_info_t {} );
                node_info_t* ptr_node_info = &available_nodes_info.back();
                NS(ComputeNodeInfo_preset)( ptr_node_info );

                std::string arch( "opencl" );

                if( nullptr != NS(ComputeNodeInfo_reserve)(
                            ptr_node_info, arch.size(), platform_name.size(),
                            name.size(), description.size() ) )
                {
                    ptr_node_info->id = *ptr_node_id;

                    std::strncpy( ptr_node_info->arch, arch.c_str(), arch.size() );
                    std::strncpy( ptr_node_info->name, name.c_str(), name.size() );

                    if( !platform_name.empty() )
                    {
                        std::strncpy( ptr_node_info->platform,
                                      platform_name.c_str(), platform_name.size() );
                    }

                    if( !description.empty() )
                    {
                        std::strncpy( ptr_node_info->description,
                                      description.c_str(), description.size() );
                    }
                }

                added_at_least_one_device = true;
                available_devices.push_back( device );
            }

            if( added_at_least_one_device )
            {
                ++platform_index;
            }
        }

        return;
    }
}

/* ------------------------------------------------------------------------- */
/* -----             Implementation of C Wrapper functions              ---- */
/* ------------------------------------------------------------------------- */

::NS(ClContextBase)* NS(ClContextBase_create)()
{
    ::NS(ClContextBase)* ptr_base_ctx = new st::ClContextBase;
    return ptr_base_ctx;
}

::NS(context_size_t) NS(ClContextBase_get_num_available_nodes)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->numAvailableNodes() : ::NS(context_size_t){ 0 };
}

::NS(context_node_info_t) const*
NS(ClContextBase_get_available_nodes_info_begin)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->availableNodesInfoBegin() : nullptr;

}

::NS(context_node_info_t) const*
NS(ClContextBase_get_available_nodes_info_end)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->availableNodesInfoEnd() : nullptr;
}

::NS(context_node_info_t) const*
NS(ClContextBase_get_default_node_info)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT context )
{
    return ( context != nullptr ) ? context->defaultNodeInfo() : nullptr;
}

::NS(context_node_id_t) NS(ClContextBase_get_default_node_id)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT context )
{
    st::ClContextBase::node_id_t default_node_id;
    ::NS(ComputeNodeId_preset)( &default_node_id );
    ::NS(context_node_info_t) const* default_node_info =
        ::NS(ClContextBase_get_default_node_info)( context );

    if( default_node_info != nullptr )
    {
        default_node_id = default_node_info->id;
    }

    return default_node_id;
}

::NS(context_node_info_t) const*
NS(ClContextBase_get_available_node_info_by_index)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_size_t) const node_index )
{
    return ( ctx != nullptr )
        ? ctx->ptrAvailableNodesInfo( node_index ) : nullptr;
}

bool NS(ClContextBase_is_available_node_amd_platform)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_size_t) const node_index )
{
    return ( ( ctx != nullptr ) &&
             ( ctx->isAvailableNodeAMDPlatform( node_index ) ) );
}

::NS(context_node_info_t) const*
NS(ClContextBase_get_available_node_info_by_node_id)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    const ::NS(context_node_id_t) *const SIXTRL_RESTRICT node_id )
{
    return ( ( ctx != nullptr ) && ( node_id != nullptr ) )
        ? ctx->ptrAvailableNodesInfo( *node_id ) : nullptr;
}

bool NS(ClContextBase_is_node_id_available)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    const ::NS(context_node_id_t) *const SIXTRL_RESTRICT node_id )
{
    return ( ( ctx != nullptr ) && ( node_id != nullptr ) &&
             ( ctx->isNodeIdAvailable( *node_id ) ) );
}

bool NS(ClContextBase_is_node_id_str_available)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT node_id_str )
{
    return ( ( ctx != nullptr ) && ( node_id_str != nullptr ) &&
             ( ctx->isNodeIdAvailable( node_id_str ) ) );
}

bool NS(ClContextBase_is_node_index_available)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_size_t) const node_index )
{
    return ( ctx != nullptr )
        ? ctx->isNodeIndexAvailable( node_index ) : false;
}

bool NS(ClContextBase_is_platform_device_tuple_available)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(comp_node_id_num_t) const platform_idx,
    ::NS(comp_node_id_num_t) const device_idx )
{
    return ( ctx != nullptr )
        ? ctx->isNodeIdAvailable( platform_idx, device_idx ) : false;
}

bool NS(ClContextBase_is_node_id_default_node)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    const ::NS(context_node_id_t) *const SIXTRL_RESTRICT node_id )
{
    return ( ( ctx != nullptr ) && ( node_id != nullptr ) )
        ? ctx->isDefaultNode( *node_id ) : false;
}

bool NS(ClContextBase_is_node_id_str_default_node)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT node_id_str )
{
    return ( ctx != nullptr ) ? ctx->isDefaultNode( node_id_str ) : false;
}

bool NS(ClContextBase_is_platform_device_tuple_default_node)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(comp_node_id_num_t) const platform_idx,
    ::NS(comp_node_id_num_t) const device_idx )
{
    return ( ctx != nullptr )
        ? ctx->isDefaultNode( platform_idx, device_idx ) : false;
}

bool NS(ClContextBase_is_node_index_default_node)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_size_t) const node_index )
{
    return ( ctx != nullptr ) ? ctx->isDefaultNode( node_index ) : false;
}

bool NS(ClContextBase_has_selected_node)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->hasSelectedNode() : false;
}

cl_device_id NS(ClContextBase_get_selected_node_device)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    cl::Device const* ptr_device = ( ctx != nullptr )
        ? ctx->selectedNodeDevice() : nullptr;

    if( ptr_device != nullptr )
    {
        return ptr_device->operator()();
    }

    return cl_device_id{};
}

::NS(context_node_info_t) const* NS(ClContextBase_get_selected_node_info)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->ptrSelectedNodeInfo() : nullptr;
}

::NS(context_node_id_t) const* NS(ClContextBase_get_selected_node_id)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->ptrSelectedNodeId() : nullptr;
}

::NS(context_size_t) NS(ClContextBase_get_selected_node_index)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    using size_type = ::NS(context_size_t);
    using limits_t  = std::numeric_limits< size_type >;

    return ( ctx != nullptr ) ? ctx->selectedNodeIndex() : limits_t::max();
}

bool NS(ClContextBase_get_selected_node_id_str)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    char* SIXTRL_RESTRICT node_id_str, ::NS(context_size_t) const max_length )
{
    return ( ctx != nullptr )
        ? ctx->selectedNodeIdStr( node_id_str, max_length ) : false;
}

void NS(ClContextBase_print_nodes_info)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->printNodesInfo();
}

void NS(ClContextBase_clear)( ::NS(ClContextBase)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->clear();
}

bool NS(ClContextBase_select_node)( ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT node_id_str )
{
    return ( ( ctx != nullptr ) && ( ctx->selectNode( node_id_str ) ) );
}

bool NS(ClContextBase_select_node_by_node_id)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    const ::NS(context_node_id_t) *const SIXTRL_RESTRICT node_id )
{
    return ( ctx != nullptr ) ? ctx->selectNode( *node_id ) : false;
}

bool NS(ClContextBase_select_node_by_index)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(context_size_t) const index )
{
    return ( ctx != nullptr ) ? ctx->selectNode( index ) : false;
}

::NS(ClContextBase)* NS(ClContextBase_new_on_selected_node_id_str)(
    char const* SIXTRL_RESTRICT node_id_str )
{
    ::NS(ClContextBase)* ctx = ::NS(ClContextBase_create)();

    if( ctx != nullptr )
    {
        using node_id_t = NS(ClContextBase)::node_id_t;
        node_id_t const* ptr_node_id = ctx->ptrAvailableNodesId( node_id_str );

        if( ( ptr_node_id == nullptr ) || ( !ctx->selectNode( *ptr_node_id ) ) )
        {
            delete ctx;
            ctx = nullptr;
        }
    }

    return ctx;
}

::NS(ClContextBase)* NS(ClContextBase_new_on_selected_node_id)(
    const ::NS(context_node_id_t) *const node_id )
{
    ::NS(ClContextBase)* ctx = nullptr;

    if( ( node_id != nullptr ) &&
        ( ::NS(ComputeNodeId_is_valid)( node_id ) ) )
    {
        ctx = new st::ClContextBase( *node_id );

        if( ( ctx != nullptr ) && ( !ctx->hasSelectedNode() ) )
        {
           delete ctx;
           ctx = nullptr;
        }
    }
    else
    {
        ctx = ::NS(ClContextBase_create)();

        if( ctx != nullptr )
        {
            ::NS(context_node_id_t) const default_node =
                ::NS(ClContextBase_get_default_node_id)( ctx );

            if( 0 != ::NS(ClContextBase_select_node_by_node_id)(
                    ctx, &default_node ) )
            {
                delete ctx;
                ctx = nullptr;
            }
        }
    }

    return ctx;
}

::NS(ClContextBase)* NS(ClContextBase_new)()
{
    return ::NS(ClContextBase_new_on_selected_node_id)( nullptr );
}

::NS(context_size_t) NS(ClContextBase_get_num_available_programs)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->numAvailablePrograms() : ( ::NS(context_size_t) )0u;
}

::NS(arch_program_id_t) NS(ClContextBase_add_program_file)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT path_to_program_file,
    char const* SIXTRL_RESTRICT compile_options )
{
    return ( ctx != nullptr )
        ? ctx->addProgramFile( path_to_program_file, compile_options )
        : st::ARCH_ILLEGAL_PROGRAM_ID;
}

bool NS(ClContextBase_compile_program)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_program_id_t) const program_id )
{
    return ( ctx != nullptr ) ? ctx->compileProgram( program_id ) : false;
}

char const* NS(ClContextBase_get_program_source_code)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_program_id_t) const program_id )
{
    return ( ctx != nullptr ) ? ctx->programSourceCode( program_id ) : nullptr;
}

bool NS(ClContextBase_has_program_file_path)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_program_id_t) const program_id )
{
    return ( ctx != nullptr ) ? ctx->programHasFilePath( program_id ) : false;
}

char const* NS(ClContextBase_get_program_path_to_file)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_program_id_t) const program_id )
{
    return ( ctx != nullptr ) ? ctx->programPathToFile( program_id ) : nullptr;
}

char const* NS(ClContextBase_get_program_compile_options)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_program_id_t) const program_id )
{
    return ( ctx != nullptr )
        ? ctx->programCompileOptions( program_id ) : nullptr;
}

char const* NS(ClContextBase_get_program_compile_report)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_program_id_t) const program_id )
{
    return ( ctx != nullptr )
        ? ctx->programCompileReport( program_id ) : nullptr;
}

bool NS(ClContextBase_is_program_compiled)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_program_id_t) const program_id )
{
    return ( ctx != nullptr ) ? ctx->isProgramCompiled( program_id ) : false;
}

::NS(arch_kernel_id_t) NS(ClContextBase_enable_kernel)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx, char const* kernel_name,
    ::NS(arch_program_id_t) const program_id )
{
    return ( ctx != nullptr )
        ? ctx->enableKernel( kernel_name, program_id )
        : st::ARCH_ILLEGAL_KERNEL_ID;
}

::NS(context_size_t) NS(ClContextBase_get_num_available_kernels)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->numAvailableKernels() : ( ::NS(context_size_t) )0u;
}


::NS(arch_kernel_id_t) NS(ClContextBase_find_kernel_id_by_name)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT kernel_name )
{
    return ( ctx != nullptr ) ? ctx->findKernelByName( kernel_name )
        : ::NS(ARCH_ILLEGAL_KERNEL_ID);
}


char const* NS(ClContextBase_get_kernel_function_name)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->kernelFunctionName( kernel_id ) : nullptr;
}

::NS(context_size_t) NS(ClContextBase_get_kernel_local_mem_size)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->kernelLocalMemSize( kernel_id )
        : NS(context_size_t){ 0 };
}

::NS(context_size_t) NS(ClContextBase_get_kernel_num_args)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->kernelNumArgs( kernel_id ) : ::NS(context_size_t){ 0 };
}

::NS(context_size_t) NS(ClContextBase_get_kernel_work_group_size)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->kernelWorkGroupSize( kernel_id ) : ::NS(context_size_t){ 0 };
}

::NS(context_size_t) NS(ClContextBase_get_kernel_max_work_group_size)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->kernelMaxWorkGroupSize( kernel_id ) : ::NS(context_size_t){ 0 };
}

bool NS(ClContextBase_set_kernel_work_group_size)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(context_size_t) const work_group_size )
{
    return ( ctx != nullptr )
        ? ctx->setKernelWorkGroupSize( kernel_id, work_group_size ) : false;
}

::NS(context_size_t)
NS(ClContextBase_get_kernel_preferred_work_group_size_multiple)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->kernelPreferredWorkGroupSizeMultiple( kernel_id )
        : ::NS(context_size_t){ 0 };
}

::NS(context_size_t) NS(ClContextBase_get_kernel_exec_counter)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->kernelExecCounter( kernel_id ) : ::NS(context_size_t){ 0 };
}

::NS(ClArgument)* NS(ClContextBase_get_ptr_kernel_argument)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(context_size_t) const arg_index ) SIXTRL_NOEXCEPT
{
    return ( ctx != nullptr )
        ? ctx->ptrKernelArgument( kernel_id, arg_index ) : nullptr;
}

::NS(kernel_arg_type_t) NS(ClContextBase_get_kernel_argument_type)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(context_size_t) const arg_index ) SIXTRL_NOEXCEPT
{
    return ( ctx != nullptr ) ? ctx->kernelArgumentType( kernel_id, arg_index )
        : st::ClContextBase::ARG_TYPE_INVALID;
}

::NS(ClArgument) const* NS(ClContextBase_get_const_ptr_kernel_argument)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(context_size_t) const arg_index ) SIXTRL_NOEXCEPT
{
    return ( ctx != nullptr )
        ? ctx->ptrKernelArgument( kernel_id, arg_index ) : nullptr;
}

void NS(ClContextBase_assign_kernel_argument)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx, ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(context_size_t) const arg_index,
    ::NS(ClArgument)* SIXTRL_RESTRICT ptr_arg )
{
    if( ( ctx != nullptr ) && ( ptr_arg != nullptr ) )
    {
        ctx->assignKernelArgument( kernel_id, arg_index, *ptr_arg );
    }
}

void NS(ClContextBase_reset_kernel_arguments)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id ) SIXTRL_NOEXCEPT
{
    if( ctx != nullptr ) ctx->resetKernelArguments( kernel_id );
}

void NS(ClContextBase_reset_single_kernel_argument)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(context_size_t) const arg_index ) SIXTRL_NOEXCEPT
{
    if( ctx != nullptr ) ctx->resetSingleKernelArgument( kernel_id, arg_index );
}

void NS(ClContextBase_assign_kernel_argument_ptr)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id, ::NS(context_size_t) const arg_idx,
    void* SIXTRL_RESTRICT ptr ) SIXTRL_NOEXCEPT
{
    if( ctx != nullptr ) ctx->assignKernelArgumentPtr( kernel_id, arg_idx, ptr );
}

void NS(ClContextBase_assign_kernel_argument_value)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(context_size_t) const arg_index, void* SIXTRL_RESTRICT arg_data,
    ::NS(context_size_t) const arg_data_size ) SIXTRL_NOEXCEPT
{
    if( arg_index < NS(ClContextBase_get_kernel_num_args)( ctx, kernel_id ) )
    {
        cl_kernel kernel = NS(ClContextBase_get_kernel)( ctx, kernel_id );
        cl_int ret = CL_SUCCESS;

        NS(ClContextBase_reset_single_kernel_argument)(
            ctx, kernel_id, arg_index );

        ret = clSetKernelArg( kernel, arg_index, arg_data_size, arg_data );
        SIXTRL_ASSERT( ret == CL_SUCCESS );
        ( void )ret;
    }
}

::NS(context_size_t) NS(ClContextBase_calculate_kernel_num_work_items)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(context_size_t) const min_num_work_items )
{
    return ( ctx != nullptr )
        ? ctx->calculateKernelNumWorkItems( kernel_id, min_num_work_items )
        : st::ClContextBase::size_type{ 0 };
}

bool NS(ClContextBase_run_kernel)( ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(context_size_t) num_work_items )
{
    return ( ctx != nullptr )
        ? ctx->runKernel( kernel_id, num_work_items ) : false;
}

bool NS(ClContextBase_run_kernel_wgsize)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(context_size_t) const num_work_items,
    ::NS(context_size_t) const work_group_size )
{
    return ( ctx != nullptr )
        ? ctx->runKernel( kernel_id, num_work_items, work_group_size )
        : false;
}

double NS(ClContextBase_get_last_exec_time)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->lastExecTime( kernel_id ) : double{ 0 };
}

double NS(ClContextBase_get_min_exec_time)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->minExecTime( kernel_id ) : double{ 0 };
}

double NS(ClContextBase_get_max_exec_time)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->maxExecTime( kernel_id ) : double{ 0 };
}

double NS(ClContextBase_get_avg_exec_time)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->avgExecTime( kernel_id ) : double{ 0 };
}

NS(context_size_t) NS(ClContextBase_get_last_exec_work_group_size)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->lastExecWorkGroupSize( kernel_id )
        : ::NS(context_size_t){ 0 };
}

NS(context_size_t) NS(ClContextBase_get_last_exec_num_work_items)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->lastExecNumWorkItems( kernel_id )
        : ::NS(context_size_t){ 0 };
}

void NS(ClContextBase_reset_kernel_exec_timing)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    if( ctx != nullptr ) ctx->resetKernelExecTiming( kernel_id );
}

::NS(arch_program_id_t) NS(ClContextBase_get_program_id_by_kernel_id)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->programIdByKernelId( kernel_id ) : ::NS(ARCH_ILLEGAL_PROGRAM_ID);
}

/* ------------------------------------------------------------------------- */

bool NS(ClContextBase_has_remapping_kernel)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) && ( ctx->has_remapping_kernel() ) );
}

::NS(arch_kernel_id_t) NS(ClContextBase_remapping_kernel_id)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->remapping_kernel_id()
        : ::NS(ARCH_ILLEGAL_KERNEL_ID);
}

::NS(arch_status_t) NS(ClContextBase_set_remapping_kernel_id)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->set_remapping_kernel_id( kernel_id )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

/* ------------------------------------------------------------------------- */

bool NS(ClContextBase_has_remapping_program)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->has_remapping_program() : false;
}

::NS(arch_program_id_t) NS(ClContextBase_remapping_program_id)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->remapping_program_id()
        : st::ARCH_ILLEGAL_PROGRAM_ID;
}

cl_program NS(ClContextBase_get_program)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_program_id_t) const program_id )
{
    cl::Program* ptr_program = ( ctx != nullptr )
        ? ctx->openClProgram( program_id ) : nullptr;

    if( ptr_program != nullptr )
    {
        return ptr_program->operator()();
    }

    return cl_program{};
}

cl_kernel NS(ClContextBase_get_kernel)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    cl::Kernel* ptr_kernel = ( ctx != nullptr )
        ? ctx->openClKernel( kernel_id ) : nullptr;

    if( ptr_kernel != nullptr )
    {
        return ptr_kernel->operator()();
    }

    return cl_kernel{};
}

cl_command_queue NS(ClContextBase_get_queue)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx )
{
    cl::CommandQueue* ptr_queue = ( ctx != nullptr )
        ? ctx->openClQueue() : nullptr;

    if( ptr_queue != nullptr )
    {
        return ptr_queue->operator()();
    }

    return cl_command_queue{};
}

cl_context NS(ClContextBase_get_opencl_context)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx )
{
    cl::Context* ptr_context = ( ctx != nullptr )
        ? ctx->openClContext() : nullptr;

    if( ptr_context != nullptr )
    {
        ptr_context->operator()();
    }

    return cl_context{};
}


void NS(ClContextBase_delete)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx )
{
    delete ctx;
    return;
}

bool NS(ClContextBase_is_debug_mode_enabled)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->debugMode() : false;
}

void NS(ClContextBase_enable_debug_mode)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->enableDebugMode();
    return;
}

void NS(ClContextBase_disable_debug_mode)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->disableDebugMode();
    return;
}

#endif /* !defined( __CUDACC__ )  */

/* end: sixtracklib/opencl/internal/cl_context_base.cpp */


