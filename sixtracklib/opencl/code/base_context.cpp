#include "sixtracklib/opencl/private/base_context.h"

#if !defined( __CUDACC__ )

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iterator>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/compute_arch.h"

#include <CL/cl.hpp>

namespace SIXTRL_NAMESPACE
{
    ClContextBase::ClContextBase() :
        m_cl_programs(),
        m_cl_kernels(),
        m_cl_buffers(),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_available_devices(),
        m_program_data(),
        m_kernel_data(),
        m_default_compile_options(),
        m_cl_context(),
        m_cl_queue(),
        m_selected_node_index( int64_t{ -1 } )
    {
        using _this_t = ClContextBase;

        _this_t::UpdateAvailableNodes(
            this->m_available_nodes_id, this->m_available_nodes_info,
            this->m_available_devices );

        SIXTRL_ASSERT( this->m_available_nodes_id.size() ==
                       this->m_available_nodes_info.size() );
    }

    ClContextBase::ClContextBase( ClContextBase::size_type const node_index ) :
        m_cl_programs(),
        m_cl_kernels(),
        m_cl_buffers(),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_available_devices(),
        m_program_data(),
        m_kernel_data(),
        m_default_compile_options(),
        m_cl_context(),
        m_cl_queue(),
        m_selected_node_index( int64_t{ -1 } )
    {
        using _this_t = ClContextBase;

        _this_t::UpdateAvailableNodes(
            this->m_available_nodes_id, this->m_available_nodes_info,
            this->m_available_devices );

        SIXTRL_ASSERT( this->m_available_nodes_id.size() ==
                       this->m_available_nodes_info.size() );

        SIXTRL_ASSERT( this->m_available_devices.size() ==
                       this->m_available_nodes_id.size() );

        if( node_index < this->numAvailableNodes() )
        {
            this->selectNode( node_index );
        }
    }

    ClContextBase::ClContextBase( ClContextBase::node_id_t const node_id ) :
        m_cl_programs(),
        m_cl_kernels(),
        m_cl_buffers(),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_available_devices(),
        m_program_data(),
        m_kernel_data(),
        m_default_compile_options(),
        m_cl_context(),
        m_cl_queue(),
        m_selected_node_index( int64_t{ -1 } )
    {
        using _this_t = ClContextBase;

        _this_t::UpdateAvailableNodes(
            this->m_available_nodes_id, this->m_available_nodes_info,
            this->m_available_devices );

        SIXTRL_ASSERT( this->m_available_nodes_id.size() ==
                       this->m_available_nodes_info.size() );

        SIXTRL_ASSERT( this->m_available_devices.size() ==
                       this->m_available_nodes_id.size() );

        size_type const node_index = this->findAvailableNodesIndex(
            NS(ComputeNodeId_get_platform_id)( &node_id ),
            NS(ComputeNodeId_get_device_id)( &node_id ) );

        if( node_index < this->numAvailableNodes() )
        {
            this->selectNode( node_id );
        }
    }

    ClContextBase::ClContextBase( char const* node_id_str ) :
        m_cl_programs(),
        m_cl_kernels(),
        m_cl_buffers(),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_available_devices(),
        m_program_data(),
        m_kernel_data(),
        m_default_compile_options(),
        m_cl_context(),
        m_cl_queue(),
        m_selected_node_index( int64_t{ -1 } )
    {
        using _this_t = ClContextBase;

        _this_t::UpdateAvailableNodes(
            this->m_available_nodes_id, this->m_available_nodes_info,
            this->m_available_devices );

        SIXTRL_ASSERT( this->m_available_nodes_id.size() ==
                       this->m_available_nodes_info.size() );

        SIXTRL_ASSERT( this->m_available_devices.size() ==
                       this->m_available_nodes_id.size() );

        size_type const node_index =
            this->findAvailableNodesIndex( node_id_str );

        if( node_index < this->numAvailableNodes() )
        {
            this->selectNode( node_id_str );
        }
    }

    ClContextBase::ClContextBase(
        ClContextBase::platform_id_t const platform_idx,
        ClContextBase::device_id_t const device_idx ) :
        m_cl_programs(),
        m_cl_kernels(),
        m_cl_buffers(),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_available_devices(),
        m_program_data(),
        m_kernel_data(),
        m_default_compile_options(),
        m_cl_context(),
        m_cl_queue(),
        m_selected_node_index( int64_t{ -1 } )
    {
        using _this_t = ClContextBase;

        _this_t::UpdateAvailableNodes(
            this->m_available_nodes_id, this->m_available_nodes_info,
            this->m_available_devices );

        SIXTRL_ASSERT( this->m_available_nodes_id.size() ==
                       this->m_available_nodes_info.size() );

        SIXTRL_ASSERT( this->m_available_devices.size() ==
                       this->m_available_nodes_id.size() );

        size_type const node_index =
            this->findAvailableNodesIndex( platform_idx, device_idx );

        if( node_index < this->numAvailableNodes() )
        {
            this->selectNode( node_index );
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

    ClContextBase::size_type
    ClContextBase::numAvailableNodes() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_available_nodes_id.size() ==
                       this->m_available_nodes_info.size() );

        return this->m_available_nodes_id.size();
    }

    ClContextBase::node_info_t const*
    ClContextBase::availableNodesInfoBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_available_nodes_info.data();
    }

    ClContextBase::node_info_t const*
    ClContextBase::availableNodesInfoEnd()   const SIXTRL_NOEXCEPT
    {
        ClContextBase::node_info_t const* ptr_end =
            this->availableNodesInfoBegin();

        if( ptr_end != nullptr )
        {
            std::advance( ptr_end, this->numAvailableNodes() );
        }

        return ptr_end;
    }

    bool ClContextBase::isNodeIdAvailable(
        ClContextBase::node_id_t const node_id ) const SIXTRL_NOEXCEPT
    {
        platform_id_t const platform_index =
            NS(ComputeNodeId_get_platform_id)( &node_id );

        device_id_t const device_index =
            NS(ComputeNodeId_get_device_id)( &node_id );

        return ( this->numAvailableNodes() >
                 this->findAvailableNodesIndex( platform_index, device_index ) );
    }

    bool ClContextBase::isNodeIdAvailable(
        ClContextBase::platform_id_t const platform_index,
        ClContextBase::device_id_t  const device_index ) const SIXTRL_NOEXCEPT
    {
        return ( this->numAvailableNodes() >
                 this->findAvailableNodesIndex( platform_index, device_index ) );
    }

    bool ClContextBase::isNodeIdAvailable( char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        return ( this->numAvailableNodes() >
                 this->findAvailableNodesIndex( node_id_str ) );
    }


    ClContextBase::node_id_t const* ClContextBase::ptrAvailableNodesId(
        ClContextBase::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_id[ index ] : nullptr;
    }

    ClContextBase::node_id_t const* ClContextBase::ptrAvailableNodesId(
        ClContextBase::platform_id_t const platform_index,
        ClContextBase::device_id_t   const device_index ) const SIXTRL_NOEXCEPT
    {
        size_type const index =
            this->findAvailableNodesIndex( platform_index, device_index );

        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_id[ index ] : nullptr;
    }

    ClContextBase::node_id_t const* ClContextBase::ptrAvailableNodesId(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        size_type const index = this->findAvailableNodesIndex( node_id_str );

        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_id[ index ] : nullptr;
    }

    ClContextBase::node_info_t const*
    ClContextBase::ptrAvailableNodesInfo(
        size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_info[ index ] : nullptr;
    }

    ClContextBase::node_info_t const*
    ClContextBase::ptrAvailableNodesInfo(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        size_type const index = this->findAvailableNodesIndex( node_id_str );

        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_info[ index ] : nullptr;
    }

    ClContextBase::node_info_t const*
    ClContextBase::ptrAvailableNodesInfo(
        ClContextBase::node_id_t const node_id ) const SIXTRL_NOEXCEPT
    {
        size_type const index = this->findAvailableNodesIndex(
                                    NS(ComputeNodeId_get_platform_id)( &node_id ),
                                    NS(ComputeNodeId_get_device_id)( &node_id ) );

        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_info[ index ] : nullptr;
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

    ClContextBase::node_id_t const*
    ClContextBase::ptrSelectedNodeId() const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesId( static_cast< size_type >(
            this->m_selected_node_index ) );
    }

    ClContextBase::node_info_t const*
    ClContextBase::ptrSelectedNodeInfo() const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesInfo( static_cast< size_type >(
            this->m_selected_node_index ) );
    }

    bool ClContextBase::selectNode( node_id_t const node_id )
    {
        platform_id_t const platform_idx =
            NS(ComputeNodeId_get_platform_id)( &node_id );

        device_id_t const device_idx =
            NS(ComputeNodeId_get_device_id)( &node_id );

        return this->selectNode( this->findAvailableNodesIndex(
                                     platform_idx, device_idx ) );
    }

    bool ClContextBase::selectNode( char const* node_id_str )
    {
        return this->selectNode( this->findAvailableNodesIndex( node_id_str ) );
    }

    bool ClContextBase::selectNode( size_type const index )
    {
        bool success = false;

        if( ( !this->hasSelectedNode() ) &&
            ( index < this->numAvailableNodes() ) )
        {
            SIXTRL_ASSERT( this->m_cl_programs.empty() );
            SIXTRL_ASSERT( this->m_selected_node_index < int64_t{ 0 } );
            SIXTRL_ASSERT( this->m_available_devices.size() ==
                           this->numAvailableNodes() );

            cl::Device  device  = this->m_available_devices[ index ];
            cl::Context context( device );
            cl::CommandQueue queue( context, device,
                                    CL_QUEUE_PROFILING_ENABLE );

            this->m_cl_context = context;
            this->m_cl_queue   = queue;

            this->m_cl_programs.clear();
            this->m_cl_kernels.clear();
            this->m_kernel_data.clear();

            this->m_selected_node_index = index;

            success = true;

            if( !this->m_program_data.empty() )
            {
                size_type const num_programs = this->m_program_data.size();

                for( auto& program_data : this->m_program_data )
                {
                    program_data.m_kernels.clear();
                    program_data.m_compiled = false;
                    program_data.m_compile_report.clear();

                    this->m_cl_programs.emplace_back(
                        this->m_cl_context, program_data.m_source_code );

                    SIXTRL_ASSERT( this->m_cl_programs.size() <= num_programs );

                    success &= this->doCompileProgram(
                        this->m_cl_programs.back(), program_data );
                }
            }
        }

        return success;
    }

    void ClContextBase::clear()
    {
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

    ClContextBase::program_id_t
    ClContextBase::addProgramCode( std::string const& source_code )
    {
        return this->addProgramCode( source_code.c_str(),
                                     this->defaultCompileOptions() );
    }

    ClContextBase::program_id_t
    ClContextBase::addProgramCode( char const* source_code )
    {
        return this->addProgramCode( source_code,
                                     this->defaultCompileOptions() );
    }

    ClContextBase::program_id_t
    ClContextBase::addProgramCode(
        std::string const& source_code, std::string const& compile_options )
    {
        program_id_t program_id = program_id_t { -1 };

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

    ClContextBase::program_id_t
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

    ClContextBase::program_id_t
    ClContextBase::addProgramFile( std::string const& path_to_program )
    {
        return this->addProgramFile(
                   path_to_program, this->m_default_compile_options );
    }

    ClContextBase::program_id_t
    ClContextBase::addProgramFile( char const* path_to_program )
    {
        return ( path_to_program != nullptr )
               ? this->addProgramFile( std::string( path_to_program ) )
               : program_id_t { -1 };
    }

    ClContextBase::program_id_t
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

        return program_id_t{ -1 };
    }

    ClContextBase::program_id_t
    ClContextBase::addProgramFile(
        char const* path_to_program, char const* compile_options )
    {
        std::string options_str = ( compile_options != nullptr )
                                  ? std::string( compile_options )
                                  : this->m_default_compile_options;

        return ( path_to_program != nullptr )
               ? this->addProgramFile( std::string( path_to_program ), options_str )
               : program_id_t { -1 };
    }

    bool ClContextBase::compileProgram(
        ClContextBase::program_id_t const program_id )
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
        ClContextBase::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) )
            ? this->m_program_data[ program_id ].m_source_code.c_str()
            : nullptr;
    }

    bool ClContextBase::programHasFilePath(
         ClContextBase::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) &&
                 ( !this->m_program_data[ program_id ].m_file_path.empty() ) );
    }

    char const* ClContextBase::programPathToFile(
        ClContextBase::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) )
            ? this->m_program_data[ program_id ].m_file_path.c_str()
            : nullptr;
    }

    char const* ClContextBase::programCompileOptions(
        ClContextBase::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) )
            ? this->m_program_data[ program_id ].m_compile_options.c_str()
            : nullptr;
    }

    char const* ClContextBase::programCompileReport(
        ClContextBase::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) )
            ? this->m_program_data[ program_id ].m_compile_report.c_str()
            : nullptr;
    }

    bool ClContextBase::isProgramCompiled(
        ClContextBase::program_id_t const program_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
                 ( this->m_program_data.size() >
                   static_cast< size_type >( program_id ) ) )
            ? this->m_program_data[ program_id ].m_compiled : false;
    }

    ClContextBase::size_type
    ClContextBase::numAvailablePrograms() const SIXTRL_NOEXCEPT
    {
        return this->m_program_data.size();
    }

    ClContextBase::kernel_id_t ClContextBase::enableKernel(
        std::string const& kernel_name,
        ClContextBase::program_id_t const program_id )
    {
        return this->enableKernel( kernel_name.c_str(), program_id );
    }

    ClContextBase::kernel_id_t
    ClContextBase::enableKernel(
        char const* kernel_name, program_id_t const program_id )
    {
        kernel_id_t kernel_id = kernel_id_t{ -1 };

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
                this->m_kernel_data.back().m_num_args =
                    kernel.getInfo< CL_KERNEL_NUM_ARGS >();

                cl::Device& selected_device = this->m_available_devices.at(
                    this->m_selected_node_index );

                this->m_kernel_data.back().m_work_group_size =
                    kernel.getWorkGroupInfo< CL_KERNEL_WORK_GROUP_SIZE >(
                        selected_device );

                this->m_kernel_data.back().m_preferred_work_group_multiple =
                    kernel.getWorkGroupInfo<
                         CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE >(
                             selected_device );

                this->m_kernel_data.back().m_local_mem_size =
                    kernel.getWorkGroupInfo< CL_KERNEL_LOCAL_MEM_SIZE >(
                        selected_device );

                program_data.m_kernels.push_back( kernel_id );
            }
        }

        return kernel_id;
    }


    bool ClContextBase::hasRemappingKernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
                 ( this->m_remap_kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( this->m_remap_kernel_id ) <
                   this->m_cl_kernels.size() ) );
    }

    char const* ClContextBase::kernelFunctionName(
        ClContextBase::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
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

    ClContextBase::size_type ClContextBase::kernelLocalMemSize(
        ClContextBase::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
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

    ClContextBase::size_type ClContextBase::kernelNumArgs(
        ClContextBase::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
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

    ClContextBase::size_type ClContextBase::kernelWorkGroupSize(
        ClContextBase::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_cl_kernels.size() ==
                       this->m_kernel_data.size() );

        return ( ( this->hasSelectedNode() ) &&
                 ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->m_kernel_data.size() ) )
            ? this->m_kernel_data[ kernel_id ].m_work_group_size
            : size_type{ 0 };
    }

    ClContextBase::size_type ClContextBase::kernelPreferredWorkGroupSizeMultiple(
        ClContextBase::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
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

    ClContextBase::program_id_t ClContextBase::kernelProgramId(
        ClContextBase::kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_cl_kernels.size() ==
                       this->m_kernel_data.size() );

        return ( ( this->hasSelectedNode() ) &&
                 ( kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( kernel_id ) <
                   this->m_kernel_data.size() ) )
            ? this->m_kernel_data[ kernel_id ].m_program_id
            : program_id_t{ -1 };
    }

    ClContextBase::kernel_id_t
    ClContextBase::remappingKernelId() const SIXTRL_NOEXCEPT
    {
        return ( this->hasRemappingKernel() )
            ? this->m_remap_kernel_id : kernel_id_t{ -1 };
    }

    bool ClContextBase::setRemappingKernelId(
        ClContextBase::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        bool success = false;

        if( ( this->hasSelectedNode() ) &&
            ( kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( kernel_id ) <
              this->m_cl_kernels.size() ) )
        {
            this->m_remap_kernel_id = kernel_id;
            success = true;
        }

        return success;
    }

    ClContextBase::size_type
    ClContextBase::numAvailableKernels() const SIXTRL_NOEXCEPT
    {
        return this->m_cl_kernels.size();
    }

    cl::Program* ClContextBase::openClProgram(
        ClContextBase::program_id_t const program_id ) SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
        ( this->m_cl_programs.size() >
            static_cast< size_type >( program_id ) ) )
        ? &this->m_cl_programs[ program_id ] : nullptr;
    }

    cl::Kernel* ClContextBase::openClKernel(
        ClContextBase::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
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

    ClContextBase::size_type
    ClContextBase::findAvailableNodesIndex(
        ClContextBase::platform_id_t const platform_index,
        ClContextBase::device_id_t const device_index ) const SIXTRL_NOEXCEPT
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

    ClContextBase::size_type
    ClContextBase::findAvailableNodesIndex(
        char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        if( ( node_id_str != nullptr ) &&
                ( std::strlen( node_id_str ) > 3u ) )
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

    bool ClContextBase::doCompileProgram(
         cl::Program& cl_program, ClContextBase::program_data_t& program_data )
    {
        bool success = false;

        if( (  this->hasSelectedNode() ) &&
            ( !program_data.m_compiled ) &&
            (  program_data.m_kernels.empty() ) &&
            ( !program_data.m_compile_options.empty() ) )
        {
            cl_int ret = cl_program.build( program_data.m_compile_options.c_str() );

            if( ret == CL_SUCCESS )
            {
                success = program_data.m_compiled = true;
            }
            else
            {
                SIXTRL_ASSERT( this->m_selected_node_index >= 0 );
                SIXTRL_ASSERT( this->m_available_devices.size() ==
                               this->numAvailableNodes() );

                program_data.m_compile_report = cl_program.getBuildInfo<
                        CL_PROGRAM_BUILD_LOG >( this->m_available_devices.at(
                            this->m_selected_node_index ) );
            }
        }

        return success;
    }

    void ClContextBase::UpdateAvailableNodes(
        std::vector< ClContextBase::node_id_t>& available_nodes_id,
        std::vector< ClContextBase::node_info_t >& available_nodes_info,
        std::vector< cl::Device >& available_devices,
        const char *const filter_str )
    {
        ( void )filter_str;

        platform_id_t platform_index = 0;
        device_id_t   device_index   = 0;

        std::vector< cl::Device > devices;
        std::vector< cl::Platform > platforms;

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
            platform.getDevices( CL_DEVICE_TYPE_ALL, &devices );

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

SIXTRL_HOST_FN NS(ClContextBase)* NS(ClContextBase_create)()
{
    NS(ClContextBase)* ptr_base_ctx = new SIXTRL_NAMESPACE::ClContextBase;
    return ptr_base_ctx;
}

SIXTRL_HOST_FN NS(context_size_t) NS(ClContextBase_get_num_available_nodes)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->numAvailableNodes() : NS(context_size_t){ 0 };
}

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(ClContextBase_get_available_nodes_info_begin)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->availableNodesInfoBegin() : nullptr;

}

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(ClContextBase_get_available_nodes_info_end)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->availableNodesInfoEnd() : nullptr;
}

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(ClContextBase_get_available_node_info_by_index)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    NS(context_size_t) const node_index )
{
    return ( ctx != nullptr )
        ? ctx->ptrAvailableNodesInfo( node_index ) : nullptr;
}

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(ClContextBase_get_available_node_info_by_node_id)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    const NS(context_node_id_t) *const SIXTRL_RESTRICT node_id )
{
    return ( ( ctx != nullptr ) && ( node_id != nullptr ) )
        ? ctx->ptrAvailableNodesInfo( *node_id ) : nullptr;
}

SIXTRL_HOST_FN bool NS(ClContextBase_is_node_id_available)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    const NS(context_node_id_t) *const SIXTRL_RESTRICT node_id )
{
    return ( ( ctx != nullptr ) && ( node_id != nullptr ) &&
             ( ctx->isNodeIdAvailable( *node_id ) ) );
}

SIXTRL_HOST_FN bool NS(ClContextBase_is_node_id_str_available)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT node_id_str )
{
    return ( ( ctx != nullptr ) && ( node_id_str != nullptr ) &&
             ( ctx->isNodeIdAvailable( node_id_str ) ) );
}

SIXTRL_HOST_FN bool NS(ClContextBase_has_selected_node)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->hasSelectedNode() : false;
}

SIXTRL_HOST_FN cl_device_id NS(ClContextBase_get_selected_node_device)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    cl::Device const* ptr_device = ( ctx != nullptr )
        ? ctx->selectedNodeDevice() : nullptr;

    if( ptr_device != nullptr )
    {
        return ptr_device->operator()();
    }

    return cl_device_id{};
}

SIXTRL_HOST_FN NS(context_node_info_t) const* NS(ClContextBase_selected_get_node_info)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->ptrSelectedNodeInfo() : nullptr;
}

SIXTRL_HOST_FN NS(context_node_id_t) const* NS(ClContextBase_selected_get_node_id)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->ptrSelectedNodeId() : nullptr;
}

SIXTRL_HOST_FN void NS(ClContextBase_clear)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->clear();
    return;
}

SIXTRL_HOST_FN bool NS(ClContextBase_select_node)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT node_id_str )
{
    return ( ctx != nullptr ) ? ctx->selectNode( node_id_str ) : false;
}

SIXTRL_HOST_FN bool NS(ClContextBase_select_node_by_node_id)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    NS(context_node_id_t) const node_id )
{
    return ( ctx != nullptr ) ? ctx->selectNode( node_id ) : false;
}

SIXTRL_HOST_FN bool NS(ClContextBase_select_node_by_index)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    NS(context_size_t) const index )
{
    return ( ctx != nullptr ) ? ctx->selectNode( index ) : false;
}

SIXTRL_HOST_FN NS(ClContextBase)*
NS(ClContextBase_new)( char const* SIXTRL_RESTRICT node_id_str )
{
    NS(ClContextBase)* ctx = NS(ClContextBase_create)();

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

SIXTRL_HOST_FN int NS(ClContextBase_add_program_file)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT path_to_program_file,
    char const* SIXTRL_RESTRICT compile_options )
{
    return ( ctx != nullptr )
        ? ctx->addProgramFile( path_to_program_file, compile_options )
        : -1;
}

SIXTRL_HOST_FN bool NS(ClContextBase_compile_program)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx, int const program_id )
{
    return ( ctx != nullptr ) ? ctx->compileProgram( program_id ) : false;
}

SIXTRL_HOST_FN char const* NS(ClContextBase_get_program_source_code)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const program_id )
{
    return ( ctx != nullptr ) ? ctx->programSourceCode( program_id ) : nullptr;
}

SIXTRL_HOST_FN bool NS(ClContextBase_has_program_file_path)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const program_id )
{
    return ( ctx != nullptr ) ? ctx->programHasFilePath( program_id ) : false;
}

SIXTRL_HOST_FN char const* NS(ClContextBase_get_program_path_to_file)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const program_id )
{
    return ( ctx != nullptr ) ? ctx->programPathToFile( program_id ) : nullptr;
}

SIXTRL_HOST_FN char const* NS(ClContextBase_get_program_compile_options)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const program_id )
{
    return ( ctx != nullptr )
        ? ctx->programCompileOptions( program_id ) : nullptr;
}

SIXTRL_HOST_FN char const* NS(ClContextBase_get_program_compile_report)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const program_id )
{
    return ( ctx != nullptr )
        ? ctx->programCompileReport( program_id ) : nullptr;
}

SIXTRL_HOST_FN bool NS(ClContextBase_is_program_compiled)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const program_id )
{
    return ( ctx != nullptr ) ? ctx->isProgramCompiled( program_id ) : false;
}

SIXTRL_HOST_FN int NS(ClContextBase_enable_kernel)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    char const* kernel_name,
    int const program_id )
{
    return ( ctx != nullptr )
        ? ctx->enableKernel( kernel_name, program_id ) : -1;
}

SIXTRL_HOST_FN char const* NS(ClContextBase_get_kernel_function_name)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->kernelFunctionName( kernel_id ) : nullptr;
}

SIXTRL_HOST_FN NS(context_size_t) NS(ClContextBase_get_kernel_local_mem_size)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->kernelLocalMemSize( kernel_id ) : NS(context_size_t){ 0 };
}

SIXTRL_HOST_FN NS(context_size_t) NS(ClContextBase_get_kernel_num_args)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->kernelNumArgs( kernel_id )
        : NS(context_size_t){ 0 };
}

SIXTRL_HOST_FN NS(context_size_t) NS(ClContextBase_get_kernel_work_group_size)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->kernelWorkGroupSize( kernel_id ) : NS(context_size_t){ 0 };
}

SIXTRL_HOST_FN NS(context_size_t)
NS(ClContextBase_get_kernel_preferred_work_group_size_multiple)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->kernelPreferredWorkGroupSizeMultiple( kernel_id )
        : NS(context_size_t){ 0 };
}

SIXTRL_HOST_FN int NS(ClContextBase_get_kernel_get_program_id)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->kernelProgramId( kernel_id ) : int{ -1 };
}

SIXTRL_HOST_FN bool NS(ClContextBase_has_remapping_kernel)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->hasRemappingKernel() : false;
}

SIXTRL_HOST_FN int NS(ClContextBase_get_remapping_kernel_id)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->remappingKernelId() : -1;
}

SIXTRL_HOST_FN bool NS(ClContextBase_set_remapping_kernel_id)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx, int const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->setRemappingKernelId( kernel_id ) : false;
}

SIXTRL_HOST_FN cl_program NS(ClContextBase_get_program)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    int const program_id )
{
    cl::Program* ptr_program = ( ctx != nullptr )
        ? ctx->openClProgram( program_id ) : nullptr;

    if( ptr_program != nullptr )
    {
        return ptr_program->operator()();
    }

    return cl_program{};
}

SIXTRL_HOST_FN cl_kernel NS(ClContextBase_get_kernel)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    int const kernel_id )
{
    cl::Kernel* ptr_kernel = ( ctx != nullptr )
        ? ctx->openClKernel( kernel_id ) : nullptr;

    if( ptr_kernel != nullptr )
    {
        return ptr_kernel->operator()();
    }

    return cl_kernel{};
}

SIXTRL_HOST_FN cl_command_queue NS(ClContextBase_get_queue)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx )
{
    cl::CommandQueue* ptr_queue = ( ctx != nullptr )
        ? ctx->openClQueue() : nullptr;

    if( ptr_queue != nullptr )
    {
        return ptr_queue->operator()();
    }

    return cl_command_queue{};
}

SIXTRL_HOST_FN cl_context NS(ClContextBase_get_opencl_context)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx )
{
    cl::Context* ptr_context = ( ctx != nullptr )
        ? ctx->openClContext() : nullptr;

    if( ptr_context != nullptr )
    {
        ptr_context->operator()();
    }

    return cl_context{};
}


SIXTRL_HOST_FN void NS(ClContextBase_delete)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx )
{
    delete ctx;
    return;
}

#endif /* !defined( __CUDACC__ )  */

/* end: sixtracklib/opencl/code/cl_context_base.cpp */


