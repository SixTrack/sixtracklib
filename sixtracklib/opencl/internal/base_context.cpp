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
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/debug_register.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/context/compute_arch.h"

#include "sixtracklib/opencl/cl.h"
#include "sixtracklib/opencl/argument.h"

namespace SIXTRL_CXX_NAMESPACE
{
    namespace
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        using  ctx_t = st::ClContextBase;
        using  st_size_t = st_size_t;
        using  st_kernel_id_t = ctx_t::kernel_id_t;
        using  st_kernel_arg_type_t = ctx_t::kernel_arg_type_t;
        using  st_status_t = ctx_t::status_t;
    }

    constexpr st_kernel_arg_type_t ctx_t::ARG_TYPE_NONE;
    constexpr st_kernel_arg_type_t ctx_t::ARG_TYPE_VALUE;
    constexpr st_kernel_arg_type_t ctx_t::ARG_TYPE_RAW_PTR;
    constexpr st_kernel_arg_type_t ctx_t::ARG_TYPE_CL_ARGUMENT;
    constexpr st_kernel_arg_type_t ctx_t::ARG_TYPE_CL_BUFFER;
    constexpr st_kernel_arg_type_t ctx_t::ARG_TYPE_INVALID;
    constexpr st_size_t ctx_t::MIN_NUM_REMAP_BUFFER_ARGS;

    constexpr ctx_t::program_path_type_t ctx_t::PROGRAM_PATH_ABSOLUTE;
    constexpr ctx_t::program_path_type_t ctx_t::PROGRAM_PATH_RELATIVE;

    ctx_t::ClContextBase(
        const char *const SIXTRL_RESTRICT config_str ) :
        m_feature_flags(),
        m_cl_programs(),
        m_cl_kernels(),
        m_cl_buffers(),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_available_devices(),
        m_program_data(),
        m_kernel_data(),
        m_default_path_to_kernel_dir(),
        m_default_sixtrlib_inc_dir(),
        m_default_compile_options( SIXTRL_DEFAULT_OPENCL_COMPILER_FLAGS ),
        m_config_str(),
        m_cl_context(),
        m_cl_queue(),
        m_cl_success_flag(),
        m_remap_program_id( st::ARCH_ILLEGAL_PROGRAM_ID ),
        m_remap_kernel_id( st::ARCH_ILLEGAL_KERNEL_ID ),
        m_selected_node_index( int64_t{ -1 } ),
        m_default_kernel_arg( uint64_t{ 0 } ), m_debug_mode( false )
    {
        this->doSetConfigStr( config_str );

        if( this->configStr() != nullptr )
        {
            this->doParseConfigStringBaseImpl( this->configStr() );
        }

        if( this->m_default_path_to_kernel_dir.empty() )
        {
            this->m_default_path_to_kernel_dir =
                ctx_t::GetDefaultPathKernelDir();
        }

        if( this->m_default_sixtrlib_inc_dir.empty() )
        {
            this->m_default_sixtrlib_inc_dir =
                ctx_t::GetDefaultSixTrlLibIncludeDir();
        }

        st_status_t status = this->doInitDefaultFeatureFlagsBaseImpl();
        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

        status = ctx_t::GetAvailableNodes( this->m_available_nodes_id,
            &this->m_available_nodes_info, &this->m_available_devices );

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;

        this->doInitDefaultProgramsBaseImpl();
    }

    ctx_t::ClContextBase(
        st_size_t const node_index,
        const char *const SIXTRL_RESTRICT config_str ) :
        m_feature_flags(),
        m_cl_programs(),
        m_cl_kernels(),
        m_cl_buffers(),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_available_devices(),
        m_program_data(),
        m_kernel_data(),
        m_default_path_to_kernel_dir(),
        m_default_sixtrlib_inc_dir(),
        m_default_compile_options( SIXTRL_DEFAULT_OPENCL_COMPILER_FLAGS ),
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
        this->doSetConfigStr( config_str );

        if( this->configStr() != nullptr )
        {
            this->doParseConfigStringBaseImpl( this->configStr() );
        }

        if( this->m_default_path_to_kernel_dir.empty() )
        {
            this->m_default_path_to_kernel_dir =
                ctx_t::GetDefaultPathKernelDir();
        }

        if( this->m_default_sixtrlib_inc_dir.empty() )
        {
            this->m_default_sixtrlib_inc_dir =
                ctx_t::GetDefaultSixTrlLibIncludeDir();
        }

        st_status_t status = this->doInitDefaultFeatureFlagsBaseImpl();
        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

        status = ctx_t::GetAvailableNodes( this->m_available_nodes_id,
            &this->m_available_nodes_info, &this->m_available_devices );

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;

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

    ctx_t::ClContextBase(
        ctx_t::node_id_t const node_id,
        const char *const SIXTRL_RESTRICT config_str ) :
        m_feature_flags(),
        m_cl_programs(),
        m_cl_kernels(),
        m_cl_buffers(),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_available_devices(),
        m_program_data(),
        m_kernel_data(),
        m_default_path_to_kernel_dir(),
        m_default_sixtrlib_inc_dir(),
        m_default_compile_options( SIXTRL_DEFAULT_OPENCL_COMPILER_FLAGS ),
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
        this->doSetConfigStr( config_str );

        if( this->configStr() != nullptr )
        {
            this->doParseConfigStringBaseImpl( this->configStr() );
        }

        if( this->m_default_path_to_kernel_dir.empty() )
        {
            this->m_default_path_to_kernel_dir =
                ctx_t::GetDefaultPathKernelDir();
        }

        if( this->m_default_sixtrlib_inc_dir.empty() )
        {
            this->m_default_sixtrlib_inc_dir =
                ctx_t::GetDefaultSixTrlLibIncludeDir();
        }

        st_status_t status = this->doInitDefaultFeatureFlagsBaseImpl();
        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

        status = ctx_t::GetAvailableNodes( this->m_available_nodes_id,
            &this->m_available_nodes_info, &this->m_available_devices );

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;

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

    ctx_t::ClContextBase(
        char const* node_id_str,
        const char *const SIXTRL_RESTRICT config_str ) :
        m_feature_flags(),
        m_cl_programs(),
        m_cl_kernels(),
        m_cl_buffers(),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_available_devices(),
        m_program_data(),
        m_kernel_data(),
        m_default_path_to_kernel_dir(),
        m_default_sixtrlib_inc_dir(),
        m_default_compile_options( SIXTRL_DEFAULT_OPENCL_COMPILER_FLAGS ),
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
        this->doSetConfigStr( config_str );

        if( this->configStr() != nullptr )
        {
            this->doParseConfigStringBaseImpl( this->configStr() );
        }

        if( this->m_default_path_to_kernel_dir.empty() )
        {
            this->m_default_path_to_kernel_dir =
                ctx_t::GetDefaultPathKernelDir();
        }

        if( this->m_default_sixtrlib_inc_dir.empty() )
        {
            this->m_default_sixtrlib_inc_dir =
                ctx_t::GetDefaultSixTrlLibIncludeDir();
        }

        st_status_t status = this->doInitDefaultFeatureFlagsBaseImpl();
        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

        status = ctx_t::GetAvailableNodes( this->m_available_nodes_id,
            &this->m_available_nodes_info, &this->m_available_devices );

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;

        this->doInitDefaultProgramsBaseImpl();

        st_size_t const node_index =
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

    ctx_t::ClContextBase(
        ctx_t::platform_id_t const platform_idx,
        ctx_t::device_id_t const device_idx,
        const char *const SIXTRL_RESTRICT config_str ) :
        m_feature_flags(),
        m_cl_programs(),
        m_cl_kernels(),
        m_cl_buffers(),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_available_devices(),
        m_program_data(),
        m_kernel_data(),
        m_default_path_to_kernel_dir(),
        m_default_sixtrlib_inc_dir(),
        m_default_compile_options( SIXTRL_DEFAULT_OPENCL_COMPILER_FLAGS ),
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
        this->doSetConfigStr( config_str );

        if( this->configStr() != nullptr )
        {
            this->doParseConfigStringBaseImpl( this->configStr() );
        }

        if( this->m_default_path_to_kernel_dir.empty() )
        {
            this->m_default_path_to_kernel_dir =
                ctx_t::GetDefaultPathKernelDir();
        }

        if( this->m_default_sixtrlib_inc_dir.empty() )
        {
            this->m_default_sixtrlib_inc_dir =
                ctx_t::GetDefaultSixTrlLibIncludeDir();
        }

        st_status_t status = this->doInitDefaultFeatureFlagsBaseImpl();
        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

        status = ctx_t::GetAvailableNodes( this->m_available_nodes_id,
            &this->m_available_nodes_info, &this->m_available_devices );

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;

        this->doInitDefaultProgramsBaseImpl();

        st_size_t const node_index =
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

    ctx_t::~ClContextBase() SIXTRL_NOEXCEPT
    {
        if( !this->m_available_nodes_info.empty() )
        {
            for( auto& nodes_info : this->m_available_nodes_info )
            {
                ::NS(ComputeNodeInfo_free)( &nodes_info );
            }

            this->m_available_nodes_info.clear();
        }
    }

    /* --------------------------------------------------------------------- */

    std::string const&
    ctx_t::default_path_to_kernel_dir_str() const SIXTRL_NOEXCEPT
    {
        return this->m_default_path_to_kernel_dir;
    }

    char const* ctx_t::default_to_kernel_dir() const SIXTRL_NOEXCEPT
    {
        return this->m_default_path_to_kernel_dir.c_str();
    }

    void ctx_t::set_default_path_to_kernel_dir(
        std::string const& SIXTRL_RESTRICT_REF default_path )
    {
        this->m_default_path_to_kernel_dir = default_path;
    }

    void ctx_t::set_default_path_to_kernel_dir(
        char const* SIXTRL_RESTRICT default_path )
    {
        if( ( default_path != nullptr ) &&
            ( std::strlen( default_path ) > st_size_t{ 0 } ) )
        {
            this->m_default_path_to_kernel_dir = default_path;
        }
    }

    /* --------------------------------------------------------------------- */

    std::string const&
    ctx_t::default_sixtrlib_inc_dir_str() const SIXTRL_NOEXCEPT
    {
        return this->m_default_sixtrlib_inc_dir;
    }

    char const* ctx_t::default_sixtrlib_inc_dir() const SIXTRL_NOEXCEPT
    {
        return this->m_default_sixtrlib_inc_dir.c_str();
    }

    void ctx_t::set_default_sixtrlib_inc_dir(
        std::string const& SIXTRL_RESTRICT_REF include_path )
    {
        this->m_default_sixtrlib_inc_dir = include_path;
    }

    void ctx_t::set_default_sixtrlib_inc_dir(
        char const* SIXTRL_RESTRICT include_path )
    {
        if( ( include_path != nullptr ) &&
            ( std::strlen( include_path ) > st_size_t{ 0 } ) )
        {
            this->m_default_path_to_kernel_dir = include_path;
        }
    }

    /* --------------------------------------------------------------------- */

    cl::Buffer const& ctx_t::internalStatusFlagsBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_cl_success_flag;
    }

    cl::Buffer& ctx_t::internalStatusFlagsBuffer() SIXTRL_NOEXCEPT
    {
        return this->m_cl_success_flag;
    }

    st_size_t ctx_t::numAvailableNodes() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_available_nodes_id.size() ==
                       this->m_available_nodes_info.size() );

        return this->m_available_nodes_id.size();
    }

    ctx_t::node_info_t const*
    ctx_t::availableNodesInfoBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_available_nodes_info.data();
    }

    ctx_t::node_info_t const*
    ctx_t::availableNodesInfoEnd()   const SIXTRL_NOEXCEPT
    {
        ctx_t::node_info_t const* ptr_end =
            this->availableNodesInfoBegin();

        if( ptr_end != nullptr )
        {
            std::advance( ptr_end, this->numAvailableNodes() );
        }

        return ptr_end;
    }

    ctx_t::node_info_t const*
    ctx_t::defaultNodeInfo() const SIXTRL_NOEXCEPT
    {
        return this->availableNodesInfoBegin();
    }

    ctx_t::node_id_t ctx_t::defaultNodeId() const SIXTRL_NOEXCEPT
    {
        NS(ComputeNodeId) default_node_id;

        ctx_t::node_info_t const* default_node_info = this->defaultNodeInfo();
        NS(ComputeNodeId_preset)( &default_node_id );

        if( default_node_info != nullptr )
        {
            default_node_id = default_node_info->id;
        }

        return default_node_id;
    }

    bool ctx_t::isNodeIndexAvailable(
         st_size_t const node_index ) const SIXTRL_NOEXCEPT
    {
        return ( node_index < this->numAvailableNodes() );
    }

    bool ctx_t::isNodeIdAvailable(
        ctx_t::node_id_t const node_id ) const SIXTRL_NOEXCEPT
    {
        platform_id_t const platform_index =
            NS(ComputeNodeId_get_platform_id)( &node_id );

        device_id_t const device_index =
            NS(ComputeNodeId_get_device_id)( &node_id );

        return ( this->numAvailableNodes() >
                 this->findAvailableNodesIndex( platform_index, device_index ) );
    }

    bool ctx_t::isNodeIdAvailable(
        ctx_t::platform_id_t const platform_index,
        ctx_t::device_id_t  const device_index ) const SIXTRL_NOEXCEPT
    {
        return ( this->numAvailableNodes() > this->findAvailableNodesIndex(
            platform_index, device_index ) );
    }

    bool ctx_t::isNodeIdAvailable(
        char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        return ( this->numAvailableNodes() >
                 this->findAvailableNodesIndex( node_id_str ) );
    }

    bool ctx_t::isDefaultNode(
        char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        node_id_t const default_node_id = this->defaultNodeId();

        return ( NS(ComputeNodeId_are_equal)(
            this->ptrAvailableNodesId( node_id_str ), &default_node_id ) );
    }

    bool ctx_t::isDefaultNode(
        ctx_t::node_id_t const node_id ) const SIXTRL_NOEXCEPT
    {
        node_id_t const default_node_id = this->defaultNodeId();
        return ( NS(ComputeNodeId_are_equal)( &node_id, &default_node_id ) );
    }

    bool ctx_t::isDefaultNode(
        st_size_t const node_index ) const SIXTRL_NOEXCEPT
    {
        node_id_t const default_node_id = this->defaultNodeId();

        return ( NS(ComputeNodeId_are_equal)(
            this->ptrAvailableNodesId( node_index ), &default_node_id ) );
    }

    bool ctx_t::isDefaultNode(
        ctx_t::platform_id_t const platform_index,
        ctx_t::device_id_t const device_index ) const SIXTRL_NOEXCEPT
    {
        node_id_t const default_node_id = this->defaultNodeId();

        return ( NS(ComputeNodeId_are_equal)(
            this->ptrAvailableNodesId( platform_index, device_index ),
                                       &default_node_id ) );
    }

    ctx_t::node_id_t const* ctx_t::ptrAvailableNodesId(
        st_size_t const index ) const SIXTRL_NOEXCEPT
    {
        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_id[ index ] : nullptr;
    }

    ctx_t::node_id_t const* ctx_t::ptrAvailableNodesId(
        ctx_t::platform_id_t const platform_index,
        ctx_t::device_id_t   const device_index ) const SIXTRL_NOEXCEPT
    {
        size_type const index =
            this->findAvailableNodesIndex( platform_index, device_index );

        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_id[ index ] : nullptr;
    }

    ctx_t::node_id_t const* ctx_t::ptrAvailableNodesId(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        size_type const index = this->findAvailableNodesIndex( node_id_str );

        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_id[ index ] : nullptr;
    }

    ctx_t::node_info_t const* ctx_t::ptrAvailableNodesInfo(
        size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_info[ index ] : nullptr;
    }

    ctx_t::node_info_t const* ctx_t::ptrAvailableNodesInfo(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        size_type const index = this->findAvailableNodesIndex( node_id_str );

        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_info[ index ] : nullptr;
    }

    ctx_t::node_info_t const* ctx_t::ptrAvailableNodesInfo(
        ctx_t::node_id_t const node_id ) const SIXTRL_NOEXCEPT
    {
        size_type const index = this->findAvailableNodesIndex(
                                    NS(ComputeNodeId_get_platform_id)( &node_id ),
                                    NS(ComputeNodeId_get_device_id)( &node_id ) );

        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_info[ index ] : nullptr;
    }

    bool ctx_t::isAvailableNodeAMDPlatform(
        st_size_t const index ) const SIXTRL_NOEXCEPT
    {
        bool is_amd_platform = false;

        ctx_t::node_info_t const* node_info =
            this->ptrAvailableNodesInfo( index );

        if( node_info != nullptr )
        {
            char _temp[ 5 ] = { '\0', '\0', '\0', '\0', '\0' };

            std::strncpy( &_temp[ 0 ], ::NS(ComputeNodeInfo_get_platform)(
                node_info ), 4u );

            std::transform( &_temp[ 0 ], &_temp[ 4 ], &_temp[ 0 ],
                [](unsigned char c){ return std::tolower(c); } );

            is_amd_platform = ( 0 == std::strncmp( &_temp[ 0 ], "amd ", 4u ) );
        }

        return is_amd_platform;
    }

    bool ctx_t::hasSelectedNode() const SIXTRL_NOEXCEPT
    {
        return ( ( this->m_selected_node_index >= 0 ) &&
                 ( this->numAvailableNodes() > static_cast< size_type >(
                       this->m_selected_node_index ) ) );
    }

    cl::Device const* ctx_t::selectedNodeDevice() const SIXTRL_NOEXCEPT
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

    cl::Device* ctx_t::selectedNodeDevice() SIXTRL_NOEXCEPT
    {
        return const_cast< cl::Device* >(
            static_cast< ClContextBase const& >( *this ).selectedNodeDevice() );
    }

    ctx_t::node_id_t const*
    ctx_t::ptrSelectedNodeId() const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesId( static_cast< size_type >(
            this->m_selected_node_index ) );
    }

    ctx_t::node_info_t const*
    ctx_t::ptrSelectedNodeInfo() const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesInfo( static_cast< size_type >(
            this->m_selected_node_index ) );
    }

    st_size_t ctx_t::selectedNodeIndex() const SIXTRL_NOEXCEPT
    {
        using size_t = st_size_t;

        return ( this->hasSelectedNode() )
            ? static_cast< size_t >( this->m_selected_node_index )
            : this->numAvailableNodes();
    }

    std::string ctx_t::selectedNodeIdStr() const SIXTRL_NOEXCEPT
    {
        char node_id_str[ 32 ];
        std::memset( &node_id_str[ 0 ], ( int )'\0', 32 );

        if( this->selectedNodeIdStr( &node_id_str[ 0 ], 32 ) )
        {
            return std::string( node_id_str );
        }

        return std::string{ "" };
    }

    bool ctx_t::selectedNodeIdStr( char* SIXTRL_RESTRICT node_id_str,
        st_size_t const max_str_length ) const SIXTRL_NOEXCEPT
    {
        return ( 0 == NS(ComputeNodeId_to_string)(
            this->ptrSelectedNodeId(), node_id_str, max_str_length ) );
    }

    bool ctx_t::selectNode( size_type const node_index )
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

    bool ctx_t::selectNode( node_id_t const node_id )
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

    bool ctx_t::selectNode(
         ctx_t::platform_id_t const platform_idx,
         ctx_t::device_id_t   const device_idx )
    {
        bool success = false;

        if( this->doSelectNode( this->findAvailableNodesIndex(
                platform_idx, device_idx ) ) )
        {
            success = this->doInitDefaultKernels();
        }

        return success;
    }

    bool ctx_t::selectNode( char const* node_id_str )
    {
        bool success = false;

        if( this->doSelectNode( this->findAvailableNodesIndex( node_id_str ) ) )
        {
            success = this->doInitDefaultKernels();
        }

        return success;
    }

    bool ctx_t::doSelectNode( size_type const node_index )
    {
        return this->doSelectNodeBaseImpl( node_index );
    }

    st_status_t ctx_t::doSetStatusFlags(
        ctx_t::status_flag_t const value )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        cl::CommandQueue* queue = this->openClQueue();

        if( ( queue != nullptr ) &&
            ( this->doGetPtrLocalStatusFlags() != nullptr ) )
        {
            *this->doGetPtrLocalStatusFlags() = value;

            cl_int const ret = queue->enqueueWriteBuffer(
                this->internalStatusFlagsBuffer(), CL_TRUE, 0,
                    sizeof( ctx_t::status_flag_t ),
                        this->doGetPtrLocalStatusFlags() );

            if( ret == CL_SUCCESS )
            {
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    st_status_t ctx_t::doFetchStatusFlags(
        ctx_t::status_flag_t* SIXTRL_RESTRICT ptr_status_flags )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        cl::CommandQueue* queue = this->openClQueue();

        if( ( queue != nullptr ) &&
            ( this->doGetPtrLocalStatusFlags() != nullptr ) )
        {
            *this->doGetPtrLocalStatusFlags() =
                st::ARCH_DEBUGGING_GENERAL_FAILURE;

            cl_int const ret = queue->enqueueReadBuffer(
                this->internalStatusFlagsBuffer(), CL_TRUE, 0,
                    sizeof( ctx_t::status_flag_t ),
                        this->doGetPtrLocalStatusFlags() );

            if( ret == CL_SUCCESS )
            {
                status = ::NS(DebugReg_get_stored_arch_status)(
                    *this->doGetPtrLocalStatusFlags() );
            }

            if( ( ptr_status_flags != SIXTRL_NULLPTR ) &&
                ( status == st::ARCH_STATUS_SUCCESS ) )
            {
                *ptr_status_flags = *this->doGetPtrLocalStatusFlags();
            }

            queue->finish();
        }

        if( ( status == st::ARCH_STATUS_SUCCESS ) &&
            ( ptr_status_flags != nullptr ) &&
            ( this->doGetPtrLocalStatusFlags() != nullptr ) )
        {
            *ptr_status_flags = *this->doGetPtrLocalStatusFlags();
        }

        return status;
    }

    st_status_t ctx_t::doAssignStatusFlagsArg(
        cl::Buffer& SIXTRL_RESTRICT_REF status_flags_arg )
    {
        return this->doAssignStatusFlagsArgBaseImpl( status_flags_arg );
    }


    st_status_t ctx_t::doAssignStatusFlagsArgBaseImpl(
            cl::Buffer& SIXTRL_RESTRICT_REF status_flags_arg )
    {
        using kernel_id_t = ctx_t::kernel_id_t;
        using size_t = st_size_t;

        st_status_t status = st::ARCH_STATUS_SUCCESS;

        if( this->has_remapping_kernel() )
        {
             kernel_id_t const kernel_id = this->remapping_kernel_id();

            if( ( kernel_id != st::ARCH_ILLEGAL_KERNEL_ID ) &&
                ( kernel_id >= kernel_id_t{ 0 } ) && ( kernel_id <
                    static_cast< size_t >( this->numAvailableKernels() ) ) )
            {
                size_t const num_args = this->kernelNumArgs( kernel_id );

                if( num_args >= ctx_t::MIN_NUM_REMAP_BUFFER_ARGS )
                {
                    if( ( this->debugMode() ) &&
                        ( num_args > ctx_t::MIN_NUM_REMAP_BUFFER_ARGS ) )
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
                else if( num_args < ctx_t::MIN_NUM_REMAP_BUFFER_ARGS )
                {
                    status = st::ARCH_STATUS_GENERAL_FAILURE;
                }
            }
        }

        return status;
    }

    st_status_t ctx_t::doAssignSlotSizeArg(
        st_size_t const slot_size )
    {
         return this->doAssignSlotSizeArgBaseImpl( slot_size );
    }

    st_status_t ctx_t::doAssignSlotSizeArgBaseImpl(
        st_size_t const slot_size )
    {
        using size_t = st_size_t;
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( slot_size > size_t{ 0 } )
        {
            status = st::ARCH_STATUS_SUCCESS;

            if( this->has_remapping_kernel() )
            {
                ctx_t::kernel_id_t const kernel_id =
                    this->remapping_kernel_id();

                if( ( kernel_id != st::ARCH_ILLEGAL_KERNEL_ID ) &&
                    ( kernel_id >= ctx_t::kernel_id_t{ 0 } ) &&
                    ( kernel_id <  static_cast< size_t >(
                            this->numAvailableKernels() ) ) )
                {
                    if( this->kernelNumArgs( kernel_id ) >=
                        ctx_t::MIN_NUM_REMAP_BUFFER_ARGS )
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

    void ctx_t::doSetConfigStr(
        const char *const SIXTRL_RESTRICT config_str )
    {
        if( ( config_str != nullptr ) &&
            ( std::strlen( config_str ) > st_size_t{ 0 } ) )
        {
            this->m_config_str = std::string{ config_str };
        }
        else
        {
            this->m_config_str.clear();
        }

        return;
    }

    bool ctx_t::doSelectNodeBaseImpl( size_type const index )
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

    void ctx_t::printNodesInfo() const SIXTRL_NOEXCEPT
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

    char const* ctx_t::configStr() const SIXTRL_NOEXCEPT
    {
        return this->m_config_str.c_str();
    }

    st_status_t ctx_t::reinit_default_programs()
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( !this->hasSelectedNode() )
        {
            this->clear();
            this->doInitDefaultPrograms();
            this->doInitDefaultKernels();

            status = st::ARCH_STATUS_SUCCESS;
        }

        return status;
    }

    void ctx_t::clear()
    {
        this->doClear();
        this->doInitDefaultPrograms();
        return;
    }

    void ctx_t::setDefaultCompileOptions(
        std::string const& compile_options_str )
    {
        this->setDefaultCompileOptions( compile_options_str.c_str() );
        return;
    }


    void ctx_t::setDefaultCompileOptions( char const* compile_options_str )
    {
        SIXTRL_ASSERT( compile_options_str != nullptr );
        this->m_default_compile_options = compile_options_str;
        return;
    }

    char const* ctx_t::defaultCompileOptions() const SIXTRL_NOEXCEPT
    {
        return this->m_default_compile_options.c_str();
    }

    ctx_t::program_id_t ctx_t::addProgramCode(
        std::string const& source_code )
    {
        return this->addProgramCode(
            source_code.c_str(), this->defaultCompileOptions() );
    }

    ctx_t::program_id_t ctx_t::addProgramCode( char const* source_code )
    {
        return this->addProgramCode(
            source_code, this->defaultCompileOptions() );
    }

    ctx_t::program_id_t ctx_t::addProgramCode(
        std::string const& source_code, std::string const& compile_options )
    {
        ctx_t::program_id_t program_id = st::ARCH_ILLEGAL_PROGRAM_ID;

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

    ctx_t::program_id_t
    ctx_t::addProgramCode(
        char const* source_code, char const* compile_options )
    {
        std::string const str_source_code( ( source_code != nullptr )
                                           ? std::string( source_code ) : std::string() );

        std::string const str_compile_options( ( compile_options != nullptr )
                                               ? std::string( compile_options )
                                               : this->m_default_compile_options );

        return this->addProgramCode( str_source_code, str_compile_options );
    }

    ctx_t::program_id_t ctx_t::addProgramFile( std::string const& path_to_prg,
        program_path_type_t const path_type )
    {
        return this->addProgramFile( path_to_prg.c_str(),
            this->m_default_compile_options.c_str(), path_type );
    }

    ctx_t::program_id_t ctx_t::addProgramFile( char const* path_to_prg,
        ctx_t::program_path_type_t const path_type )
    {
        return this->addProgramFile( path_to_prg,
            this->m_default_compile_options.c_str(), path_type );
    }

    ctx_t::program_id_t ctx_t::addProgramFile(
        char const* SIXTRL_RESTRICT in_path_to_program,
        char const* SIXTRL_RESTRICT in_compile_options,
        ctx_t::program_path_type_t const path_type )
    {
        ctx_t::program_id_t program_id = st::ARCH_ILLEGAL_PROGRAM_ID;
        std::string path_to_program;

        if( ( in_path_to_program != nullptr ) &&
            ( std::strlen( in_path_to_program ) > st_size_t{ 0 } ) )
        {
            if( path_type == ctx_t::PROGRAM_PATH_ABSOLUTE )
            {
                path_to_program = in_path_to_program;
            }
            else if( path_type == ctx_t::PROGRAM_PATH_RELATIVE )
            {
                path_to_program  = this->m_default_path_to_kernel_dir;
                path_to_program += in_path_to_program;
            }
        }

        if( !path_to_program.empty() )
        {
            std::fstream program_file( path_to_program, std::ios::in );

            if( program_file.is_open() )
            {
                std::string const source_code(
                    ( std::istreambuf_iterator< char >( program_file ) ),
                    std::istreambuf_iterator< char >() );

                if( !source_code.empty() )
                {
                    std::string const compile_options =
                        ( ( in_compile_options != nullptr ) &&
                          ( std::strlen( in_compile_options ) > st_size_t{ 0 } )
                        )
                        ? in_compile_options
                        : this->m_default_compile_options;

                    program_id = this->addProgramCode(
                        source_code, compile_options );

                    if( program_id >= program_id_t{ 0 } )
                    {
                        this->m_program_data[ program_id ].m_file_path =
                            path_to_program;
                    }
                }
            }
        }

        return program_id;
    }

    ctx_t::program_id_t ctx_t::addProgramFile(
        std::string const& SIXTRL_RESTRICT_REF path_to_program,
        std::string const& SIXTRL_RESTRICT_REF compile_options,
        ctx_t::program_path_type_t const path_type )
    {
        return this->addProgramFile( path_to_program.c_str(),
            compile_options.c_str(), path_type );
    }

    bool ctx_t::compileProgram( ctx_t::program_id_t const program_id )
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

    char const* ctx_t::programSourceCode(
        ctx_t::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) )
            ? this->m_program_data[ program_id ].m_source_code.c_str()
            : nullptr;
    }

    bool ctx_t::programHasFilePath(
         ctx_t::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) &&
                 ( !this->m_program_data[ program_id ].m_file_path.empty() ) );
    }

    char const* ctx_t::programPathToFile(
        ctx_t::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) )
            ? this->m_program_data[ program_id ].m_file_path.c_str()
            : nullptr;
    }

    char const* ctx_t::programCompileOptions(
        ctx_t::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) )
            ? this->m_program_data[ program_id ].m_compile_options.c_str()
            : nullptr;
    }

    char const* ctx_t::programCompileReport(
        ctx_t::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) )
            ? this->m_program_data[ program_id ].m_compile_report.c_str()
            : nullptr;
    }

    bool ctx_t::isProgramCompiled(
        ctx_t::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) )
            ? this->m_program_data[ program_id ].m_compiled : false;
    }

    st_size_t ctx_t::numAvailablePrograms() const SIXTRL_NOEXCEPT
    {
        return this->m_program_data.size();
    }

    ctx_t::kernel_id_t ctx_t::enableKernel(
        std::string const& kernel_name,
        ctx_t::program_id_t const program_id )
    {
        return this->enableKernel( kernel_name.c_str(), program_id );
    }

    ctx_t::kernel_id_t
    ctx_t::enableKernel(
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
                SIXTRL_ASSERT( max_work_group_size >= pref_work_group_size );
                SIXTRL_ASSERT( max_work_group_size <= size_type{ 0x7fffffff } );

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

    ctx_t::kernel_id_t ctx_t::findKernelByName(
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

    bool ctx_t::has_remapping_program() const SIXTRL_NOEXCEPT
    {
        return ( ( this->m_remap_program_id >= program_id_t{ 0 } ) &&
                 ( static_cast< size_type >( this->m_remap_program_id ) <
                   this->numAvailablePrograms() ) );
    }

    ctx_t::program_id_t
    ctx_t::remapping_program_id() const SIXTRL_NOEXCEPT
    {
        return ( this->has_remapping_program() )
            ? this->m_remap_program_id : st::ARCH_ILLEGAL_PROGRAM_ID;
    }

    bool ctx_t::has_remapping_kernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
                 ( this->m_remap_kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( this->m_remap_kernel_id ) <
                   this->m_cl_kernels.size() ) );
    }

    char const* ctx_t::kernelFunctionName(
        ctx_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
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

    st_size_t ctx_t::kernelLocalMemSize(
        ctx_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
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

    st_size_t ctx_t::kernelNumArgs(
        ctx_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
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

    st_size_t ctx_t::kernelWorkGroupSize(
        ctx_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
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

    st_size_t ctx_t::kernelMaxWorkGroupSize(
        ctx_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
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

    bool ctx_t::setKernelWorkGroupSize(
        ctx_t::kernel_id_t const kernel_id,
        st_size_t work_group_size ) SIXTRL_NOEXCEPT
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

    st_size_t ctx_t::kernelPreferredWorkGroupSizeMultiple(
        ctx_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
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

    ctx_t::program_id_t ctx_t::programIdByKernelId(
        ctx_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
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

    st_size_t ctx_t::kernelExecCounter(
        ctx_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->numAvailableKernels() ) )
            ? this->m_kernel_data[ kernel_id ].m_exec_count
            : size_type{ 0 };
    }

    ClArgument* ctx_t::ptrKernelArgument(
        ctx_t::kernel_id_t const kernel_id,
        st_size_t const arg_index ) SIXTRL_NOEXCEPT
    {
        using ctx_t = ClContextBase;
        using   ptr_t = ClArgument*;

        return const_cast< ptr_t >( static_cast< ctx_t const& >(
            *this ).ptrKernelArgument( kernel_id, arg_index ) );
    }

    ClArgument const* ctx_t::ptrKernelArgument(
        ctx_t::kernel_id_t const kernel_id,
        st_size_t const arg_index ) const SIXTRL_NOEXCEPT
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

    ctx_t::kernel_arg_type_t ctx_t::kernelArgumentType(
        ctx_t::kernel_id_t const kernel_id,
        st_size_t const arg_index) const SIXTRL_NOEXCEPT
    {
        return ( ( static_cast< size_t >( kernel_id ) <
                   this->m_kernel_data.size() ) &&
                 ( arg_index < this->m_kernel_data[ kernel_id ].m_num_args ) )
            ? this->m_kernel_data[ kernel_id ].m_arg_types[ arg_index ]
            : st::ctx_t::ARG_TYPE_INVALID;
    }

    void ctx_t::assignKernelArgument(
        ctx_t::kernel_id_t const kernel_id, st_size_t const index,
        ClArgument& SIXTRL_RESTRICT_REF arg )
    {
        SIXTRL_ASSERT( arg.context() == this );
        SIXTRL_ASSERT( arg.size() > st_size_t{ 0 } );

        this->m_kernel_data.at( kernel_id ).setKernelArg(
            ctx_t::ARG_TYPE_CL_ARGUMENT, index, &arg );

        cl::Kernel* kernel = this->openClKernel( kernel_id );
        if( kernel != nullptr ) kernel->setArg( index, arg.openClBuffer() );

    }

    void ctx_t::assignKernelArgumentRawPtr(
        ctx_t::kernel_id_t const kernel_id,
        st_size_t const arg_index,
        st_size_t const arg_size, void* ptr )
    {
        SIXTRL_ASSERT( arg_size > st_size_t{ 0 } );
        SIXTRL_ASSERT( ptr != nullptr );

        this->m_kernel_data.at( kernel_id ).setKernelArg(
            ctx_t::ARG_TYPE_RAW_PTR, arg_index, ptr );

        cl::Kernel* kernel = this->openClKernel( kernel_id );
        if( kernel != nullptr ) kernel->setArg( arg_index, arg_size, ptr );
    }

    void ctx_t::assignKernelArgumentClBuffer(
            ctx_t::kernel_id_t const kernel_id,
            st_size_t const arg_index,
            cl::Buffer& SIXTRL_RESTRICT_REF cl_buffer_arg )
    {
        this->m_kernel_data.at( kernel_id ).setKernelArg(
            ctx_t::ARG_TYPE_CL_BUFFER, arg_index, &cl_buffer_arg );

        cl::Kernel* kernel = this->openClKernel( kernel_id );
        if( kernel != nullptr ) kernel->setArg( arg_index, cl_buffer_arg );
    }

    void ctx_t::assignKernelArgumentRawPtr(
            ctx_t::kernel_id_t const kernel_id, st_size_t const arg_index,
            st_size_t const arg_size,
            SIXTRL_ARGPTR_DEC void const* ptr ) SIXTRL_NOEXCEPT
        {
            SIXTRL_ASSERT( kernel_id >= ctx_t::kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< ctx_t::size_type >(
                kernel_id ) < this->numAvailableKernels() );

            this->m_kernel_data[ kernel_id ].setKernelArg(
                    ctx_t::ARG_TYPE_RAW_PTR, arg_index, nullptr );

            cl::Kernel* cxx_kernel = this->openClKernel( kernel_id );

            if( cxx_kernel != nullptr )
            {
                ::cl_kernel kernel = cxx_kernel->operator()();
                cl_int const ret = ::clSetKernelArg(
                    kernel, arg_index, arg_size, ptr );

                SIXTRL_ASSERT( ret == CL_SUCCESS );
                ( void )ret;
            }
        }

    void ctx_t::resetKernelArguments(
        ctx_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
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

    void ctx_t::resetSingleKernelArgument(
        ctx_t::kernel_id_t const kernel_id,
        st_size_t const arg_index ) SIXTRL_NOEXCEPT
    {
        cl::Kernel* kernel = this->openClKernel( kernel_id );
        if( kernel != nullptr ) kernel->setArg(
                arg_index, this->m_default_kernel_arg );

        this->m_kernel_data.at( kernel_id ).setKernelArg(
            ctx_t::ARG_TYPE_NONE, arg_index, nullptr );
    }

    st_size_t ctx_t::calculateKernelNumWorkItems(
        ctx_t::kernel_id_t const kernel_id,
        st_size_t const min_num_work_items ) const SIXTRL_NOEXCEPT
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

    bool ctx_t::runKernel(
        ctx_t::kernel_id_t const kernel_id,
        st_size_t min_num_work_items )
    {
        return this->runKernel( kernel_id, min_num_work_items,
                         this->kernelWorkGroupSize( kernel_id ) );
    }

    bool ctx_t::runKernel( ctx_t::kernel_id_t const kernel_id,
        st_size_t const min_num_work_items,
        st_size_t work_group_size )
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

    double ctx_t::lastExecTime(
        ctx_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->numAvailableKernels() ) )
            ? this->m_kernel_data[ kernel_id ].m_last_exec_time : double{ 0 };
    }

    double ctx_t::minExecTime(
        ctx_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->numAvailableKernels() ) )
            ? this->m_kernel_data[ kernel_id ].m_min_exec_time : double{ 0 };
    }

    double ctx_t::maxExecTime(
        ctx_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->numAvailableKernels() ) )
            ? this->m_kernel_data[ kernel_id ].m_max_exec_time : double{ 0 };
    }

    double ctx_t::avgExecTime(
        ctx_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->numAvailableKernels() ) )
            ? this->m_kernel_data[ kernel_id ].avgExecTime() : double{ 0 };
    }

    st_size_t ctx_t::lastExecWorkGroupSize(
        ctx_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->numAvailableKernels() ) )
            ? this->m_kernel_data[ kernel_id ].m_last_work_group_size
            : size_type{ 0 };
    }

    st_size_t ctx_t::lastExecNumWorkItems(
        ctx_t::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->numAvailableKernels() ) )
            ? this->m_kernel_data[ kernel_id ].m_last_num_of_threads
            : size_type{ 0 };
    }

    void ctx_t::resetKernelExecTiming(
         ctx_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        if( ( kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( kernel_id ) <
              this->numAvailableKernels() ) )
        {
            this->m_kernel_data[ kernel_id ].resetTiming();
        }

        return;
    }

    void ctx_t::addKernelExecTime( double const time,
        ctx_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        if( ( kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( kernel_id ) <
              this->numAvailableKernels() ) )
        {
            this->m_kernel_data[ kernel_id ].addExecTime( time );
        }

        return;
    }

    ctx_t::kernel_id_t
    ctx_t::remapping_kernel_id() const SIXTRL_NOEXCEPT
    {
        return ( this->has_remapping_kernel() )
            ? this->m_remap_kernel_id : st::ARCH_ILLEGAL_KERNEL_ID;
    }

    bool ctx_t::set_remapping_kernel_id(
        ctx_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
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

    st_size_t
    ctx_t::numAvailableKernels() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_cl_kernels.size() ==
                       this->m_kernel_data.size() );

        return this->m_cl_kernels.size();
    }

    cl::Program* ctx_t::openClProgram(
        ctx_t::program_id_t const program_id ) SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
        ( this->m_cl_programs.size() >
            static_cast< size_type >( program_id ) ) )
        ? &this->m_cl_programs[ program_id ] : nullptr;
    }

    cl::Kernel* ctx_t::openClKernel(
        ctx_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
            ( this->m_cl_kernels.size() > static_cast< size_type >( kernel_id ) ) )
            ? &this->m_cl_kernels[ kernel_id ] : nullptr;
    }

    cl::CommandQueue* ctx_t::openClQueue() SIXTRL_NOEXCEPT
    {
        return &this->m_cl_queue;
    }

    std::uintptr_t ctx_t::openClQueueAddr() const SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< std::uintptr_t >( &this->m_cl_queue() );
    }

    cl::Context* ctx_t::openClContext() SIXTRL_NOEXCEPT
    {
        return &this->m_cl_context;
    }

    std::uintptr_t ctx_t::openClContextAddr() const SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< std::uintptr_t >( &this->m_cl_context() );
    }

    bool ctx_t::debugMode() const  SIXTRL_NOEXCEPT
    {
        return this->m_debug_mode;
    }

    void ctx_t::enableDebugMode()  SIXTRL_NOEXCEPT
    {
        if( ( !this->debugMode() ) && ( !this->hasSelectedNode() ) )
        {
            this->m_debug_mode = true;
            this->clear();
            this->doInitDefaultPrograms();
        }

        return;
    }

    void ctx_t::disableDebugMode() SIXTRL_NOEXCEPT
    {
        if( ( this->debugMode() ) && ( !this->hasSelectedNode() ) )
        {
            this->m_debug_mode = false;
            this->clear();
            this->doInitDefaultPrograms();
        }

        return;
    }

    ctx_t::status_flag_t ctx_t::status_flags()
    {
        st_status_t const status = this->doFetchStatusFlags(
            this->doGetPtrLocalStatusFlags() );

        return ( status == st::ARCH_STATUS_SUCCESS )
            ? *this->doGetPtrLocalStatusFlags()
            : st::ARCH_DEBUGGING_GENERAL_FAILURE;
    }

    ctx_t::status_flag_t ctx_t::set_status_flags(
        ctx_t::status_flag_t const status_flags )
    {
        return this->doSetStatusFlags( status_flags );
    }

    st_status_t ctx_t::prepare_status_flags_for_use()
    {
        st_status_t const status = this->doSetStatusFlags(
            st::ARCH_DEBUGGING_REGISTER_EMPTY );

        SIXTRL_ASSERT( ( status != st::ARCH_STATUS_SUCCESS ) ||
            ( ( this->doGetPtrLocalStatusFlags() != nullptr ) &&
              ( *this->doGetPtrLocalStatusFlags() ==
                st::ARCH_DEBUGGING_REGISTER_EMPTY ) ) );

        return status;
    }

    st_status_t ctx_t::eval_status_flags_after_use()
    {
        st_status_t status = this->doFetchStatusFlags(
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

    st_status_t ctx_t::assign_slot_size_arg(
        st_size_t const slot_size )
    {
        return this->doAssignSlotSizeArg( slot_size );
    }

    st_status_t ctx_t::assign_status_flags_arg(
        ctx_t::cl_buffer_t& SIXTRL_RESTRICT_REF success_flag_arg )
    {
        return this->doAssignStatusFlagsArg( success_flag_arg );
    }

    /* --------------------------------------------------------------------- */

    st_size_t ctx_t::num_feature_flags() const SIXTRL_NOEXCEPT
    {
        return this->m_feature_flags.size();
    }

    ctx_t::feature_flags_iter_t
    ctx_t::feature_flags_begin() const SIXTRL_NOEXCEPT
    {
        return this->m_feature_flags.begin();
    }

    ctx_t::feature_flags_iter_t
    ctx_t::feature_flags_end() const SIXTRL_NOEXCEPT
    {
        return this->m_feature_flags.end();
    }

    bool ctx_t::has_feature_flag(
        std::string const& SIXTRL_RESTRICT_REF str ) const SIXTRL_NOEXCEPT
    {
        return ( ( !str.empty() ) &&
                 ( this->m_feature_flags.find( str ) !=
                   this->m_feature_flags.end() ) );
    }

    bool ctx_t::has_feature_flag(
        char const* SIXTRL_RESTRICT str ) const SIXTRL_NOEXCEPT
    {
        return ( ( str != nullptr ) && ( std::strlen( str ) > 0u ) &&
                 ( this->m_feature_flags.find( str ) !=
                   this->m_feature_flags.end() ) );
    }

    std::string const& ctx_t::feature_flag_str(
        std::string const& SIXTRL_RESTRICT_REF str ) const
    {
        auto it = this->m_feature_flags.end();

        if( !str.empty() )
        {
            it = this->m_feature_flags.find( str );
        }

        if( it == this->m_feature_flags.end() )
        {
            std::string error_msg( "feature_flag \"" );
            error_msg += str;
            error_msg += "\" not found";

            throw std::runtime_error( error_msg.c_str() );
        }

        return it->second;
    }

    std::string const& ctx_t::feature_flag_str(
        char const* SIXTRL_RESTRICT str ) const
    {
        auto it = this->m_feature_flags.end();
        bool valid_str = false;

        if( ( str != nullptr ) && ( std::strlen( str ) > 0 ) )
        {
            valid_str = true;
            it = this->m_feature_flags.find( str );
        }

        if( it == this->m_feature_flags.end() )
        {
            std::string error_msg( "feature_flag " );

            if( valid_str )
            {
                error_msg += "\"";
                error_msg += str;
                error_msg += "\" not found";
            }
            else
            {
                error_msg += " not found, input str empty / illegal";
            }

            throw std::runtime_error( error_msg.c_str() );
        }

        return it->second;
    }

    char const* ctx_t::feature_flag(
        std::string const& SIXTRL_RESTRICT_REF str ) const SIXTRL_NOEXCEPT
    {
        char const* flag_value = nullptr;

        if( !str.empty() )
        {
            auto it = this->m_feature_flags.find( str );

            if( it != this->m_feature_flags.end() )
            {
                flag_value = it->second.c_str();
            }
        }

        return flag_value;
    }

    char const* ctx_t::feature_flag(
        char const* SIXTRL_RESTRICT str ) const SIXTRL_NOEXCEPT
    {
        char const* flag_value = nullptr;

        if( ( str != nullptr ) && ( std::strlen( str ) > 0u ) )
        {
            auto it = this->m_feature_flags.find( str );

            if( it != this->m_feature_flags.end() )
            {
                flag_value = it->second.c_str();
            }
        }

        return flag_value;
    }

    void ctx_t::set_feature_flag(
        std::string const& SIXTRL_RESTRICT_REF str,
        std::string const& SIXTRL_RESTRICT_REF flag_value )
    {
        if( !str.empty() )
        {
            this->m_feature_flags.emplace( std::make_pair( str, flag_value ) );
        }
    }

    void ctx_t::set_feature_flag( char const* SIXTRL_RESTRICT str,
        char const* SIXTRL_RESTRICT flag_value )
    {
        if( ( str != nullptr ) && ( std::strlen( str ) > 0u ) &&
            ( flag_value != nullptr ) )
        {
            this->m_feature_flags.emplace( std::make_pair( str, flag_value ) );
        }
    }

    std::string ctx_t::feature_flag_repr(
        std::string const& SIXTRL_RESTRICT_REF str,
        std::string const& SIXTRL_RESTRICT_REF prefix,
        std::string const& SIXTRL_RESTRICT_REF sep  ) const
    {
        std::ostringstream a2str;

        if( this->has_feature_flag( str ) )
        {
            a2str << prefix << str << sep << this->feature_flag_str( str );
        }

        return a2str.str();
    }

    std::string ctx_t::feature_flag_repr(
        char const* SIXTRL_RESTRICT str,
        char const* SIXTRL_RESTRICT prefix,
        char const* SIXTRL_RESTRICT sep ) const
    {
        if( ( str != nullptr ) && ( std::strlen( str ) > 0u ) )
        {
            std::string const prefix_str = ( prefix != nullptr )
                ? std::string{ prefix } : std::string{ "-D" };

            std::string const sep_str = ( sep != nullptr )
                ? std::string{ sep } : std::string{ "=" };

            return this->feature_flag_repr(
                std::string{ str }, prefix_str, sep_str );
        }

        return std::string{};
    }

    ctx_t::size_type ctx_t::feature_flag_repr_required_capacity(
        char const* SIXTRL_RESTRICT str, char const* SIXTRL_RESTRICT prefix,
        char const* SIXTRL_RESTRICT sep ) const
    {
        std::string const temp_str_repr(
            this->feature_flag_repr( str, prefix, sep ) );

        return ( !temp_str_repr.empty() ) ? ( temp_str_repr.size() + 1u ) : 0u;
    }

    st_status_t ctx_t::feature_flag_repr_as_cstr(
        char* out_str, size_type out_str_capacity,
        char const* SIXTRL_RESTRICT str,
        char const* SIXTRL_RESTRICT prefix,
        char const* SIXTRL_RESTRICT sep ) const
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( out_str != nullptr ) && ( out_str_capacity > 0u ) )
        {
            std::string const temp_str_repr(
                this->feature_flag_repr( str, prefix, sep ) );

            std::memset( out_str, ( int )'\0', out_str_capacity );

            if( ( !temp_str_repr.empty() ) &&
                ( temp_str_repr.size() < out_str_capacity ) )
            {
                std::strncpy( out_str, temp_str_repr.c_str(),
                              out_str_capacity - 1u );

                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    /* --------------------------------------------------------------------- */

    st_status_t ctx_t::GetAllowedNodesFromEnvVariable(
        std::vector< ctx_t::node_id_t >& allowed_node_ids,
        char const* SIXTRL_RESTRICT env_variable_name )
    {
        st_status_t status = st::ARCH_STATUS_SUCCESS;

        allowed_node_ids.clear();
        char const* env_var_begin = nullptr;

        if( env_variable_name != nullptr )
        {
            env_var_begin = std::getenv( env_variable_name );
        }
        else
        {
            env_var_begin = std::getenv( "SIXTRACKLIB_DEVICES" );
        }

        if( ( env_var_begin != nullptr ) &&
            ( std::strlen( env_var_begin ) > std::size_t{ 0 } ) )
        {
            std::regex expr( "(^\\s*|\\s*\\,\\s*|\\s*\\;\\s*)"
                             "(opencl\\:|2\\:)?(\\d+\\.\\d+)" );

            auto expr_it  = std::cregex_iterator( env_var_begin,
                    env_var_begin + std::strlen( env_var_begin ), expr );

            auto expr_end = (std::cregex_iterator());

            ctx_t::node_id_t tnode_id;

            for( ; expr_it != expr_end ; ++expr_it )
            {
                std::string const node_id_str( ( *expr_it )[ 3 ] );

                if( ( expr_it->size() == std::size_t{ 4 } ) &&
                    ( 0 == ::NS(ComputeNodeId_from_string)(
                        &tnode_id, node_id_str.c_str() ) ) )
                {
                    allowed_node_ids.push_back( tnode_id );
                }
            }

            if( !allowed_node_ids.empty() )
            {
                auto cmp_node_ids =
                []( ctx_t::node_id_t const& SIXTRL_RESTRICT_REF lhs,
                    ctx_t::node_id_t const& SIXTRL_RESTRICT_REF rhs )
                {
                    return (
                        ( ::NS(ComputeNodeId_get_platform_id)( &lhs ) <
                            ::NS(ComputeNodeId_get_platform_id)( &rhs ) ) ||
                        ( ( ::NS(ComputeNodeId_get_platform_id)( &lhs ) ==
                            ::NS(ComputeNodeId_get_platform_id)( &rhs ) ) &&
                            ( ::NS(ComputeNodeId_get_device_id)( &lhs ) <
                            ::NS(ComputeNodeId_get_device_id)( &rhs ) ) ) );
                };

                std::sort( allowed_node_ids.begin(), allowed_node_ids.end(),
                           cmp_node_ids );

                allowed_node_ids.erase(
                    std::unique( allowed_node_ids.begin(), allowed_node_ids.end(),
                    []( ctx_t::node_id_t const& SIXTRL_RESTRICT_REF lhs,
                        ctx_t::node_id_t const& SIXTRL_RESTRICT_REF rhs )
                    {
                        return ::NS(ComputeNodeId_are_equal)( &lhs, &rhs );
                    } ), allowed_node_ids.end() );

                SIXTRL_ASSERT( std::is_sorted( allowed_node_ids.begin(),
                                allowed_node_ids.end(), cmp_node_ids ) );
            }
        }

        return status;
    }

    st_status_t ctx_t::GetAllAvailableNodes(
            std::vector< cl::Platform>& available_platforms,
            std::vector< cl::Device >&  available_devices,
            std::vector< ctx_t::node_id_t >* ptr_available_nodes_id,
            std::vector< ctx_t::node_info_t >* ptr_available_nodes_info )
    {
        st_status_t status = st::ARCH_STATUS_SUCCESS;

        if( ptr_available_nodes_id != nullptr )
        {
            ptr_available_nodes_id->clear();
            ptr_available_nodes_id->reserve( 100 );
        }

        if( ptr_available_nodes_info != nullptr )
        {
            ptr_available_nodes_info->clear();
            ptr_available_nodes_info->reserve( 100 );
        }

        available_platforms.clear();
        available_platforms.reserve( 10 );

        available_devices.clear();
        available_devices.reserve( 100 );

        std::vector< cl::Device > temp_devices;
        temp_devices.clear();
        temp_devices.reserve( 100 );

        ctx_t::platform_id_t platform_idx = ctx_t::platform_id_t{ 0 };
        ctx_t::device_id_t device_idx = ctx_t::device_id_t{ 0 };

        cl::Platform::get( &available_platforms );
        std::string const arch_str( "opencl" );

        for( auto const& platform : available_platforms )
        {
            temp_devices.clear();
            device_idx = 0;

            std::string const platform_name =
                platform.getInfo< CL_PLATFORM_NAME >();

            #if defined( SIXTRL_OPENCL_CXX_ENABLES_HOST_EXCEPTIONS ) && \
                         SIXTRL_OPENCL_CXX_ENABLES_HOST_EXCEPTIONS == 1
            try
            {
            #endif /* OpenCL 1.x C++ Host Exceptions enabled */

            platform.getDevices( CL_DEVICE_TYPE_ALL, &temp_devices );

            #if defined( SIXTRL_OPENCL_CXX_ENABLES_HOST_EXCEPTIONS ) && \
                         SIXTRL_OPENCL_CXX_ENABLES_HOST_EXCEPTIONS == 1
            }
            catch( cl::Error const& e )
            {
                #if !defined( NDEBUG )
                std::cerr << "Error while probing devices for platform "
                          << platform_name << " --> skipping"
                          << std::endl;
                #endif /* !defined( NDEBUG ) */
            }
            #endif /* OpenCL 1.x C++ Host Exceptions enabled */

            bool added_at_least_one_device = false;

            for( auto const& device : temp_devices )
            {
                ctx_t::node_id_t temp_node_id;

                ::NS(ComputeNodeId_preset)( &temp_node_id );

                ::NS(ComputeNodeId_set_platform_id)(
                    &temp_node_id, platform_idx );

                ::NS(ComputeNodeId_set_device_id)(
                    &temp_node_id, device_idx++ );

                available_devices.push_back( device );
                added_at_least_one_device = true;

                if( ptr_available_nodes_id != nullptr )
                {
                    ptr_available_nodes_id->push_back( temp_node_id );

                    SIXTRL_ASSERT( ptr_available_nodes_id->size() ==
                                   available_devices.size() );
                }

                if( ptr_available_nodes_info != nullptr )
                {
                    std::string name;
                    std::string description;

                    cl_int ret = device.getInfo( CL_DEVICE_NAME, &name );
                    ret |= device.getInfo( CL_DEVICE_EXTENSIONS, &description );

                    ptr_available_nodes_info->push_back(
                        ctx_t::node_info_t{} );

                    ctx_t::node_info_t* ptr_node_info =
                        &ptr_available_nodes_info->back();

                    ::NS(ComputeNodeInfo_preset)( ptr_node_info );

                    if( nullptr != NS(ComputeNodeInfo_reserve)( ptr_node_info,
                        arch_str.size(), platform_name.size(), name.size(),
                            description.size() ) )
                    {
                        ptr_node_info->id = temp_node_id;

                        std::strncpy( ptr_node_info->arch,
                                      arch_str.c_str(), arch_str.size() );

                        std::strncpy( ptr_node_info->name,
                                      name.c_str(), name.size() );

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

                        SIXTRL_ASSERT( ptr_available_nodes_info->size() ==
                                       available_devices.size() );
                    }
                }
            }

            if( added_at_least_one_device ) ++platform_idx;
        }

        return status;
    }

    st_status_t ctx_t::GetAvailableNodes(
            std::vector< ctx_t::node_id_t>& available_nodes_id,
            std::vector< ctx_t::node_info_t >* ptr_available_nodes_info,
            std::vector< cl::Device >* ptr_available_devices,
            char const* SIXTRL_RESTRICT env_variable_name,
            char const* SIXTRL_RESTRICT filter_str )
    {
        using node_id_t = ctx_t::node_id_t;

        ( void )filter_str;

        char const* env_var_begin = nullptr;

        if( ( env_variable_name != nullptr ) &&
            ( std::strlen( env_variable_name ) > std::size_t{ 0 } ) )
        {
            env_var_begin = std::getenv( env_variable_name );
        }
        else
        {
            env_var_begin = std::getenv( "SIXTRACKLIB_DEVICES" );
        }

        available_nodes_id.clear();

        if( ptr_available_nodes_info != nullptr )
                ptr_available_nodes_info->clear();

        if( ptr_available_devices != nullptr )
            ptr_available_devices->clear();

        std::vector< cl::Device > all_available_devices;
        std::vector< cl::Platform > all_available_platforms;
        std::vector< node_id_t > all_available_nodes_id;
        std::vector< ctx_t::node_info_t > all_available_nodes_info;

        st_status_t status = ctx_t::GetAllAvailableNodes(
            all_available_platforms, all_available_devices,
                &all_available_nodes_id, &all_available_nodes_info );

        if( status != st::ARCH_STATUS_SUCCESS ) return status;
        if( all_available_devices.empty() ) return status;

        SIXTRL_ASSERT( all_available_devices.size() >=
                       all_available_platforms.size() );

        SIXTRL_ASSERT( all_available_devices.size() ==
                       all_available_nodes_id.size() );

        SIXTRL_ASSERT( all_available_devices.size() ==
                       all_available_nodes_info.size() );

        std::vector< ctx_t::node_id_t > allowed_node_ids;

        bool first_available_node = false;
        bool ext_list_of_nodes    = false;

        if( ( env_var_begin != nullptr ) &&
            ( std::strlen( env_var_begin ) > std::size_t{ 0 } ) )
        {
            if( std::strcmp( env_var_begin, "first" ) == 0 )
            {
                first_available_node = true;
            }
            else if( std::strcmp( env_var_begin, "all" ) != 0 )
            {
                ext_list_of_nodes = true;
            }
        }

        if( ext_list_of_nodes )
        {
            status = ctx_t::GetAllowedNodesFromEnvVariable(
                allowed_node_ids, env_variable_name );
        }

        if( status != st::ARCH_STATUS_SUCCESS ) return status;

        st_size_t const max_expected_num_nodes = ( ext_list_of_nodes )
            ? allowed_node_ids.size() : all_available_devices.size();

        if( max_expected_num_nodes > st_size_t{ 0 } )
        {
            available_nodes_id.reserve( max_expected_num_nodes );

            if( ptr_available_nodes_info != nullptr )
                ptr_available_nodes_info->reserve( max_expected_num_nodes );

            if( ptr_available_devices != nullptr )
                ptr_available_devices->reserve( max_expected_num_nodes );

            if( ext_list_of_nodes )
            {
                auto cmp_node_ids = [](
                    node_id_t const& SIXTRL_RESTRICT_REF lhs,
                    node_id_t const& SIXTRL_RESTRICT_REF rhs )
                {
                    return (
                        ( ::NS(ComputeNodeId_get_platform_id)( &lhs ) <
                          ::NS(ComputeNodeId_get_platform_id)( &rhs ) ) ||
                        ( ( ::NS(ComputeNodeId_get_platform_id)( &lhs ) ==
                            ::NS(ComputeNodeId_get_platform_id)( &rhs ) ) &&
                          ( ::NS(ComputeNodeId_get_device_id)( &lhs ) <
                            ::NS(ComputeNodeId_get_device_id)( &rhs ) ) ) );
                };

                SIXTRL_ASSERT( !allowed_node_ids.empty() );
                SIXTRL_ASSERT( std::is_sorted( allowed_node_ids.begin(),
                       allowed_node_ids.end(), cmp_node_ids ) );

                st_size_t const num_available_nodes =
                    all_available_devices.size();

                for( st_size_t ii = st_size_t{ 0 } ;
                        ii < num_available_nodes ; ++ii )
                {
                    ctx_t::node_id_t const& node_id =
                        all_available_nodes_id[ ii ];

                    if( std::binary_search( allowed_node_ids.begin(),
                            allowed_node_ids.end(), node_id, cmp_node_ids ) )
                    {
                        if( ptr_available_nodes_info != nullptr )
                        {
                            ptr_available_nodes_info->push_back(
                                all_available_nodes_info[ ii ] );

                            ::NS(ComputeNodeInfo_preset)(
                                &all_available_nodes_info[ ii ] );

                        }

                        if( ptr_available_devices != nullptr )
                        {
                            ptr_available_devices->push_back(
                                all_available_devices[ ii ] );
                        }

                        available_nodes_id.push_back( node_id );
                    }
                }

                SIXTRL_ASSERT( available_nodes_id.size() <=
                    allowed_node_ids.size() );

            }
            else if( first_available_node )
            {
                SIXTRL_ASSERT( !all_available_devices.empty() );
                SIXTRL_ASSERT( !all_available_nodes_id.empty() );
                SIXTRL_ASSERT( !all_available_nodes_info.empty() );

                available_nodes_id.push_back( all_available_nodes_id.front() );

                if( ptr_available_nodes_info != nullptr )
                {
                    ptr_available_nodes_info->push_back(
                        all_available_nodes_info.front() );

                    ::NS(ComputeNodeInfo_preset)(
                        &all_available_nodes_info.front() );
                }

                if( ptr_available_devices != nullptr )
                {
                    ptr_available_devices->push_back(
                        all_available_devices.front() );
                }
            }
            else
            {
                available_nodes_id.swap( all_available_nodes_id );

                if( ptr_available_nodes_info != nullptr )
                {
                    ptr_available_nodes_info->swap( all_available_nodes_info );
                }

                if( ptr_available_devices != nullptr )
                {
                    ptr_available_devices->swap( all_available_devices );
                }
            }
        }

        SIXTRL_ASSERT( ( ptr_available_devices == nullptr ) ||
            ( ptr_available_devices->size() == available_nodes_id.size() ) );

        SIXTRL_ASSERT( ( ptr_available_nodes_info == nullptr ) ||
            ( ptr_available_nodes_info->size() == available_nodes_id.size() ) );

        if( !all_available_nodes_info.empty() )
        {
            for( auto& node_info : all_available_nodes_info )
            {
                ::NS(ComputeNodeInfo_free)( &node_info );
            }
        }

        return status;
    }

    ctx_t::kernel_data_list_t const&
    ctx_t::kernelData() const SIXTRL_NOEXCEPT
    {
        return this->m_kernel_data;
    }

    ctx_t::program_data_list_t const&
    ctx_t::programData() const SIXTRL_NOEXCEPT
    {
        return this->m_program_data;
    }

    void ctx_t::setLastWorkGroupSize(
         st_size_t const work_group_size,
         ctx_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
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

    void ctx_t::setLastNumWorkItems(
        st_size_t const num_work_items,
        ctx_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
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

    st_size_t ctx_t::findAvailableNodesIndex(
        ctx_t::platform_id_t const platform_index,
        ctx_t::device_id_t const device_index ) const SIXTRL_NOEXCEPT
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

    st_size_t ctx_t::findAvailableNodesIndex(
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

    ctx_t::status_flag_t*
    ctx_t::doGetPtrLocalStatusFlags() SIXTRL_NOEXCEPT
    {
        return &this->m_local_status_flags;
    }

    ctx_t::status_flag_t const*
    ctx_t::doGetPtrLocalStatusFlags() const SIXTRL_NOEXCEPT
    {
        return &this->m_local_status_flags;
    }

    void ctx_t::doParseConfigString(
        const char *const SIXTRL_RESTRICT config_str )
    {
        this->doParseConfigStringBaseImpl( config_str );
        return;
    }

    void ctx_t::doParseConfigStringBaseImpl(
        const char *const SIXTRL_RESTRICT config_str )
    {
        ( void )config_str;
        return;
    }

    void ctx_t::doClear()
    {
        this->doClearBaseImpl();
        return;
    }

    void ctx_t::doClearBaseImpl() SIXTRL_NOEXCEPT
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

    st_status_t ctx_t::doInitDefaultFeatureFlags()
    {
        return this->doInitDefaultFeatureFlagsBaseImpl();
    }

    st_status_t ctx_t::doInitDefaultFeatureFlagsBaseImpl()
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        this->set_feature_flag( "_GPUCODE", "1" );
        this->set_feature_flag( "SIXTRL_BUFFER_ARGPTR_DEC", "__private" );
        this->set_feature_flag( "SIXTRL_BUFFER_DATAPTR_DEC", "__global" );

        if( this->num_feature_flags() >= st_size_t{ 2 } )
        {
            status = st::ARCH_STATUS_SUCCESS;
        }

        return status;
    }

    bool ctx_t::doInitDefaultPrograms()
    {
        return this->doInitDefaultProgramsBaseImpl();
    }

    bool ctx_t::doInitDefaultProgramsBaseImpl()
    {
        bool success = false;

        std::string path_to_remap_kernel_program(
            this->default_path_to_kernel_dir_str() );

        if( !this->debugMode() )
        {
            path_to_remap_kernel_program += "managed_buffer_remap.cl";
        }
        else
        {
            path_to_remap_kernel_program += "managed_buffer_remap_debug.cl";
        }

        std::ostringstream remap_program_compile_options;

        if( !this->m_default_compile_options.empty() )
        {
            remap_program_compile_options <<
                this->defaultCompileOptions() << " ";
        }

        remap_program_compile_options
            << this->feature_flag_repr( "_GPUCODE" ) << " "
            << this->feature_flag_repr( "SIXTRL_BUFFER_ARGPTR_DEC" ) << " "
            << this->feature_flag_repr( "SIXTRL_BUFFER_DATAPTR_DEC" ) << " "
            << "-I " << NS(PATH_TO_SIXTRL_INCLUDE_DIR);

        program_id_t const remap_program_id = this->addProgramFile(
            path_to_remap_kernel_program, remap_program_compile_options.str() );

        if( remap_program_id >= program_id_t{ 0 } )
        {
            this->m_remap_program_id = remap_program_id;
            success = true;
        }

        return success;
    }

    bool ctx_t::doInitDefaultKernels()
    {
        return this->doInitDefaultKernelsBaseImpl();
    }

    bool ctx_t::doInitDefaultKernelsBaseImpl()
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

    bool ctx_t::doCompileProgram(
         cl::Program& cl_program, ctx_t::program_data_t& program_data )
    {
        return this->doCompileProgramBaseImpl( cl_program, program_data );
    }

    bool ctx_t::doCompileProgramBaseImpl(
         cl::Program& cl_program, ctx_t::program_data_t& program_data )
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
}

/* ------------------------------------------------------------------------- */
/* -----             Implementation of C Wrapper functions              ---- */
/* ------------------------------------------------------------------------- */

::NS(arch_size_t) NS(OpenCL_get_num_all_nodes)( void )
{
    return st::ctx_t::NUM_ALL_NODES();
}

::NS(arch_size_t) NS(OpenCL_get_all_nodes)(
    NS(ComputeNodeId)* SIXTRL_RESTRICT out_node_ids_begin,
    NS(arch_size_t) const max_num_node_ids )
{
    return st::ctx_t::GET_ALL_NODES(
        out_node_ids_begin, max_num_node_ids );
}

void NS(OpenCL_print_all_nodes)( void )
{
    st::ctx_t::PRINT_ALL_NODES();
}

::NS(arch_size_t) NS(OpenCL_get_all_nodes_required_str_capacity)( void )
{
    return st::ctx_t::GET_ALL_NODES_REQUIRED_STRING_CAPACITY();
}

::NS(arch_status_t) NS(OpenCL_get_all_nodes_as_string)(
    char* SIXTRL_RESTRICT out_node_info_str,
    ::NS(arch_size_t) const out_node_info_str_capacity )
{
    st::st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

    if( ( out_node_info_str != nullptr ) &&
        ( out_node_info_str_capacity > st::ctx_t::size_type{ 0 } ) )
    {
        std::memset( out_node_info_str, ( int )'\0',
                     out_node_info_str_capacity );

        std::string const temp_str(
            st::ctx_t::PRINT_ALL_NODES_TO_STRING() );

        std::strncpy( out_node_info_str, temp_str.c_str(),
            out_node_info_str_capacity - st::ctx_t::size_type{ 1 } );

        status = st::ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

::NS(arch_size_t) NS(OpenCL_num_available_nodes)(
    char const* SIXTRL_RESTRICT env_variable_name )
{
    return st::ctx_t::NUM_AVAILABLE_NODES( nullptr, env_variable_name );
}

::NS(arch_size_t) NS(OpenCL_num_available_nodes_detailed)(
    char const* SIXTRL_RESTRICT filter_str,
    char const* SIXTRL_RESTRICT env_variable_name )
{
    return st::ctx_t::NUM_AVAILABLE_NODES(
        filter_str, env_variable_name );
}

::NS(arch_size_t) NS(OpenCL_get_available_nodes)(
    ::NS(ComputeNodeId)* SIXTRL_RESTRICT out_node_ids_begin,
    ::NS(arch_size_t) const max_num_node_ids )
{
    return st::ctx_t::GET_AVAILABLE_NODES(
        out_node_ids_begin, max_num_node_ids );
}

::NS(arch_size_t) NS(OpenCL_get_available_nodes_detailed)(
    ::NS(ComputeNodeId)* SIXTRL_RESTRICT out_node_ids_begin,
    ::NS(arch_size_t) const max_num_node_ids,
    ::NS(arch_size_t) const skip_first_num_nodes,
    char const* SIXTRL_RESTRICT filter_str,
    char const* SIXTRL_RESTRICT env_variable_name )
{
    return st::ctx_t::GET_AVAILABLE_NODES(
        out_node_ids_begin, max_num_node_ids, skip_first_num_nodes,
            filter_str, env_variable_name );
}

void NS(OpenCL_print_available_nodes)( void )
{
    st::ctx_t::PRINT_AVAILABLE_NODES();
}

void NS(OpenCL_print_available_nodes_detailed)(
    char const* SIXTRL_RESTRICT filter_str,
    char const* SIXTRL_RESTRICT env_variable_name )
{
    st::ctx_t::PRINT_AVAILABLE_NODES( filter_str, env_variable_name );
}

::NS(arch_size_t) NS(OpenCL_get_available_nodes_required_str_capacity)(
    char const* SIXTRL_RESTRICT filter_str,
    char const* SIXTRL_RESTRICT env_variable_name )
{
    return st::ctx_t::GET_AVAILABLE_NODES_REQUIRED_STRING_CAPACITY(
        filter_str, env_variable_name );
}

::NS(arch_status_t) NS(OpenCL_get_available_nodes_as_string)(
    char* SIXTRL_RESTRICT out_node_info_str,
    ::NS(arch_size_t) const out_node_info_str_capacity,
    char const* SIXTRL_RESTRICT filter_str,
    char const* SIXTRL_RESTRICT env_variable_name )
{
    using ctx_t = st::ClContextBase;
    st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

    if( ( out_node_info_str != nullptr ) &&
        ( out_node_info_str_capacity > st_size_t{ 0 } ) )
    {
        std::memset( out_node_info_str, ( int )'\0',
                     out_node_info_str_capacity );

        std::string const temp_str( ctx_t::PRINT_AVAILABLE_NODES_TO_STRING(
            filter_str, env_variable_name ) );

        std::strncpy( out_node_info_str, temp_str.c_str(),
                      out_node_info_str_capacity - st_size_t{ 1 } );

        status = st::ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* ************************************************************************* */

::NS(ClContextBase)* NS(ClContextBase_create)()
{
    ::NS(ClContextBase)* ptr_base_ctx = new st::ClContextBase;
    return ptr_base_ctx;
}

::NS(arch_size_t) NS(ClContextBase_get_num_available_nodes)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->numAvailableNodes() : ::NS(arch_size_t){ 0 };
}

::NS(context_node_info_t) const*
NS(ClContextBase_get_available_nodes_info_begin)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->availableNodesInfoBegin() : nullptr;
}

::NS(context_node_info_t) const* NS(ClContextBase_get_available_nodes_info_end)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->availableNodesInfoEnd() : nullptr;
}

::NS(context_node_info_t) const* NS(ClContextBase_get_default_node_info)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT context )
{
    return ( context != nullptr ) ? context->defaultNodeInfo() : nullptr;
}

::NS(context_node_id_t) NS(ClContextBase_get_default_node_id)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT context )
{
    st::ctx_t::node_id_t default_node_id;
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
    ::NS(arch_size_t) const node_index )
{
    return ( ctx != nullptr )
        ? ctx->ptrAvailableNodesInfo( node_index ) : nullptr;
}

bool NS(ClContextBase_is_available_node_amd_platform)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_size_t) const node_index )
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
    ::NS(arch_size_t) const node_index )
{
    return ( ( ctx != nullptr ) &&
             ( ctx->isNodeIndexAvailable( node_index ) ) );
}

bool NS(ClContextBase_is_platform_device_tuple_available)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(comp_node_id_num_t) const platform_idx,
    ::NS(comp_node_id_num_t) const device_idx )
{
    return ( ( ctx != nullptr ) &&
             ( ctx->isNodeIdAvailable( platform_idx, device_idx ) ) );
}

bool NS(ClContextBase_is_node_id_default_node)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    const ::NS(context_node_id_t) *const SIXTRL_RESTRICT node_id )
{
    return ( ( ctx != nullptr ) && ( node_id != nullptr ) &&
             ( ctx->isDefaultNode( *node_id ) ) );
}

bool NS(ClContextBase_is_node_id_str_default_node)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT node_id_str )
{
    return ( ( ctx != nullptr ) && ( ctx->isDefaultNode( node_id_str ) ) );
}

bool NS(ClContextBase_is_platform_device_tuple_default_node)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(comp_node_id_num_t) const platform_idx,
    ::NS(comp_node_id_num_t) const device_idx )
{
    return ( ( ctx != nullptr ) &&
             ( ctx->isDefaultNode( platform_idx, device_idx ) ) );
}

bool NS(ClContextBase_is_node_index_default_node)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_size_t) const node_index )
{
    return ( ( ctx != nullptr ) && ( ctx->isDefaultNode( node_index ) ) );
}

bool NS(ClContextBase_has_selected_node)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) && ( ctx->hasSelectedNode() ) );
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

::NS(arch_size_t) NS(ClContextBase_get_selected_node_index)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    using size_type = ::NS(arch_size_t);
    using limits_t  = std::numeric_limits< size_type >;

    return ( ctx != nullptr ) ? ctx->selectedNodeIndex() : limits_t::max();
}

bool NS(ClContextBase_get_selected_node_id_str)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    char* SIXTRL_RESTRICT node_id_str, ::NS(arch_size_t) const max_length )
{
    return ( ( ctx != nullptr ) &&
             ( ctx->selectedNodeIdStr( node_id_str, max_length ) ) );
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

::NS(arch_status_t) NS(ClContextBase_reinit_default_programs)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->reinit_default_programs() : st::ARCH_STATUS_GENERAL_FAILURE;
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
    return ( ( ctx != nullptr ) && ( ctx->selectNode( *node_id ) ) );
}

bool NS(ClContextBase_select_node_by_index)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_size_t) const index )
{
    return ( ( ctx != nullptr ) && ( ctx->selectNode( index ) ) );
}

::NS(ClContextBase)* NS(ClContextBase_new_on_selected_node_id_str)(
    char const* SIXTRL_RESTRICT node_id_str )
{
    ::NS(ClContextBase)* ctx = ::NS(ClContextBase_create)();

    if( ctx != nullptr )
    {
        using node_id_t = st::ctx_t::node_id_t;
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

void NS(ClContextBase_set_default_compile_options)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT default_compile_options )
{
    if( ctx != nullptr ) ctx->setDefaultCompileOptions(
            default_compile_options );
}

char const* NS(ClContextBase_default_compile_options)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx ) SIXTRL_NOEXCEPT
{
    return ( ctx != nullptr ) ? ctx->defaultCompileOptions() : nullptr;
}

::NS(arch_size_t) NS(ClContextBase_get_num_available_programs)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->numAvailablePrograms() : ( ::NS(arch_size_t) )0u;
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

::NS(arch_size_t)
NS(ClContextBase_get_required_program_source_code_capacity)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_program_id_t) const program_id )
{
    ::NS(arch_size_t) capacity = ::NS(arch_size_t){ 0 };

    if( ctx != nullptr )
    {
        char const* ptr_code = ctx->programSourceCode( program_id );

        if( ptr_code != nullptr )
        {
            capacity = ::NS(arch_size_t){ 1 } + std::strlen( ptr_code );
        }
    }

    return capacity;
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

::NS(arch_size_t) NS(ClContextBase_get_required_program_path_capacity)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_program_id_t) const program_id )
{
    ::NS(arch_size_t) capacity = ::NS(arch_size_t){ 0 };

    if( ctx != nullptr )
    {
        char const* ptr_path = ctx->programPathToFile( program_id );

        if( ptr_path != nullptr )
        {
            capacity = ::NS(arch_size_t){ 1 } + std::strlen( ptr_path );
        }
    }

    return capacity;
}

char const* NS(ClContextBase_get_program_path_to_file)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_program_id_t) const program_id )
{
    return ( ctx != nullptr ) ? ctx->programPathToFile( program_id ) : nullptr;
}

::NS(arch_size_t)
NS(ClContextBase_get_required_program_compile_options_capacity)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_program_id_t) const program_id )
{
    ::NS(arch_size_t) capacity = ::NS(arch_size_t){ 0 };

    if( ctx != nullptr )
    {
        char const* ptr_options = ctx->programCompileOptions( program_id );

        if( ptr_options != nullptr )
        {
            capacity = ::NS(arch_size_t){ 1 } + std::strlen( ptr_options );
        }
    }

    return capacity;
}

char const* NS(ClContextBase_get_program_compile_options)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_program_id_t) const program_id )
{
    return ( ctx != nullptr )
        ? ctx->programCompileOptions( program_id ) : nullptr;
}

::NS(arch_size_t) NS(ClContextBase_get_required_program_compile_report_capacity)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_program_id_t) const program_id )
{
    ::NS(arch_size_t) capacity = ::NS(arch_size_t){ 0 };

    if( ctx != nullptr )
    {
        char const* ptr_report = ctx->programCompileReport( program_id );

        if( ptr_report != nullptr )
        {
            capacity = ::NS(arch_size_t){ 1 } + std::strlen( ptr_report );
        }
    }

    return capacity;
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

::NS(arch_size_t) NS(ClContextBase_get_num_available_kernels)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->numAvailableKernels() : ( ::NS(arch_size_t) )0u;
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

::NS(arch_size_t) NS(ClContextBase_get_kernel_local_mem_size)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->kernelLocalMemSize( kernel_id )
        : NS(arch_size_t){ 0 };
}

::NS(arch_size_t) NS(ClContextBase_get_kernel_num_args)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->kernelNumArgs( kernel_id ) : ::NS(arch_size_t){ 0 };
}

::NS(arch_size_t) NS(ClContextBase_get_kernel_work_group_size)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->kernelWorkGroupSize( kernel_id ) : ::NS(arch_size_t){ 0 };
}

::NS(arch_size_t) NS(ClContextBase_get_kernel_max_work_group_size)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->kernelMaxWorkGroupSize( kernel_id ) : ::NS(arch_size_t){ 0 };
}

bool NS(ClContextBase_set_kernel_work_group_size)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(arch_size_t) const work_group_size )
{
    return ( ctx != nullptr )
        ? ctx->setKernelWorkGroupSize( kernel_id, work_group_size ) : false;
}

::NS(arch_size_t)
NS(ClContextBase_get_kernel_preferred_work_group_size_multiple)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->kernelPreferredWorkGroupSizeMultiple( kernel_id )
        : ::NS(arch_size_t){ 0 };
}

::NS(arch_size_t) NS(ClContextBase_get_kernel_exec_counter)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->kernelExecCounter( kernel_id ) : ::NS(arch_size_t){ 0 };
}

::NS(ClArgument)* NS(ClContextBase_get_ptr_kernel_argument)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(arch_size_t) const arg_index )
{
    return ( ctx != nullptr )
        ? ctx->ptrKernelArgument( kernel_id, arg_index ) : nullptr;
}

::NS(kernel_arg_type_t) NS(ClContextBase_get_kernel_argument_type)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(arch_size_t) const arg_index )
{
    return ( ctx != nullptr ) ? ctx->kernelArgumentType( kernel_id, arg_index )
        : st::ctx_t::ARG_TYPE_INVALID;
}

::NS(ClArgument) const* NS(ClContextBase_get_const_ptr_kernel_argument)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(arch_size_t) const arg_index )
{
    return ( ctx != nullptr )
        ? ctx->ptrKernelArgument( kernel_id, arg_index ) : nullptr;
}

void NS(ClContextBase_assign_kernel_argument)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx, ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(arch_size_t) const arg_index,
    ::NS(ClArgument)* SIXTRL_RESTRICT ptr_arg )
{
    if( ( ctx != nullptr ) && ( ptr_arg != nullptr ) )
    {
        ctx->assignKernelArgument( kernel_id, arg_index, *ptr_arg );
    }
}

void NS(ClContextBase_reset_kernel_arguments)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    if( ctx != nullptr ) ctx->resetKernelArguments( kernel_id );
}

void NS(ClContextBase_reset_single_kernel_argument)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(arch_size_t) const arg_index )
{
    if( ctx != nullptr ) ctx->resetSingleKernelArgument( kernel_id, arg_index );
}

void NS(ClContextBase_assign_kernel_argument_raw_ptr)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id, ::NS(arch_size_t) const arg_idx,
    ::NS(arch_size_t) const arg_size, void const* ptr )
{
    if( ctx != nullptr ) ctx->assignKernelArgumentRawPtr(
            kernel_id, arg_idx, arg_size, ptr );
}

void NS(ClContextBase_assign_kernel_argument_value)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(arch_size_t) const arg_index, void* SIXTRL_RESTRICT arg_data,
    ::NS(arch_size_t) const arg_data_size )
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

::NS(arch_size_t) NS(ClContextBase_calculate_kernel_num_work_items)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(arch_size_t) const min_num_work_items )
{
    return ( ctx != nullptr )
        ? ctx->calculateKernelNumWorkItems( kernel_id, min_num_work_items )
        : st::ctx_t::size_type{ 0 };
}

bool NS(ClContextBase_run_kernel)( ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(arch_size_t) num_work_items )
{
    return ( ( ctx != nullptr ) &&
             ( ctx->runKernel( kernel_id, num_work_items ) ) );
}

bool NS(ClContextBase_run_kernel_wgsize)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id,
    ::NS(arch_size_t) const num_work_items,
    ::NS(arch_size_t) const work_group_size )
{
    return ( ( ctx != nullptr ) &&
             ( ctx->runKernel( kernel_id, num_work_items, work_group_size ) ) );
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

NS(arch_size_t) NS(ClContextBase_get_last_exec_work_group_size)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->lastExecWorkGroupSize( kernel_id )
        : ::NS(arch_size_t){ 0 };
}

NS(arch_size_t) NS(ClContextBase_get_last_exec_num_work_items)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->lastExecNumWorkItems( kernel_id )
        : ::NS(arch_size_t){ 0 };
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
    return ( ( ctx != nullptr ) && ( ctx->has_remapping_program() ) );
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

uintptr_t NS(ClContextBase_get_queue_addr)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx ) SIXTRL_NOEXCEPT
{
    return ( ctx != nullptr ) ? ctx->openClQueueAddr() : std::uintptr_t{ 0 };
}

cl_context NS(ClContextBase_get_opencl_context)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx )
{
    cl::Context* ptr_context = ( ctx != nullptr )
        ? ctx->openClContext() : nullptr;

    if( ptr_context != nullptr )
    {
        return ptr_context->operator()();
    }

    return cl_context{};
}

uintptr_t NS(ClContextBase_get_opencl_context_addr)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx ) SIXTRL_NOEXCEPT
{
    return ( ctx != nullptr ) ? ctx->openClContextAddr() : std::uintptr_t{ 0 };
}

void NS(ClContextBase_delete)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx )
{
    delete ctx;
}

bool NS(ClContextBase_is_debug_mode_enabled)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) && ( ctx->debugMode() ) );
}

void NS(ClContextBase_enable_debug_mode)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->enableDebugMode();
}

void NS(ClContextBase_disable_debug_mode)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->disableDebugMode();
}


::NS(arch_size_t) NS(ClContextBase_num_feature_flags)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx )SIXTRL_NOEXCEPT
{
    return ( ctx != nullptr )
        ? ctx->num_feature_flags() : ::NS(arch_size_t){ 0 };
}

bool NS(ClContextBase_has_feature_flag)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT str ) SIXTRL_NOEXCEPT
{
    return ( ( ctx != nullptr ) && ( ctx->has_feature_flag)( str ) );
}

char const* NS(ClContextBase_feature_flag)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT str ) SIXTRL_NOEXCEPT
{
    return ( ctx != nullptr ) ? ctx->feature_flag( str ) : SIXTRL_NULLPTR;
}

void NS(ClContextBase_set_feature_flag)(
    ::NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT str, char const* flag_val )
{
    if( ctx != nullptr ) ctx->set_feature_flag( str, flag_val );
}

::NS(arch_size_t) NS(ClContextBase_feature_flag_repr_required_capacity)(
    const ::NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT str,
    char const* SIXTRL_RESTRICT prefix,
    char const* SIXTRL_RESTRICT sep )
{
    return ( ctx != nullptr )
        ? ctx->feature_flag_repr_required_capacity( str, prefix, sep )
        : ::NS(arch_size_t){ 0 };
}

::NS(arch_status_t)  NS(ClContextBase_feature_flag_repr_as_cstr)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    char* out_str, NS(arch_size_t) const out_str_capacity,
    char const* SIXTRL_RESTRICT str,
    char const* SIXTRL_RESTRICT prefix,
    char const* SIXTRL_RESTRICT sep )
{
    return ( ctx != nullptr )
        ? ctx->feature_flag_repr_as_cstr(
            out_str, out_str_capacity, str, prefix, sep )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

#endif /* !defined( __CUDACC__ )  */

/* end: sixtracklib/opencl/internal/cl_context_base.cpp */


