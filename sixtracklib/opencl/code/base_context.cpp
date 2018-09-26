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
#include <sstream>
#include <vector>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/compute_arch.h"

#include <CL/cl.hpp>

namespace SIXTRL_NAMESPACE
{
    CLContextBase::CLContextBase() :
        m_cl_programs(),
        m_cl_kernels(),
        m_cl_buffers(),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_available_devices(),
        m_program_data(),
        m_default_compile_options(),
        m_cl_context(),
        m_cl_queue(),
        m_selected_node_index( int64_t{ -1 } )
    {
        using _this_t = CLContextBase;

        _this_t::UpdateAvailableNodes(
            this->m_available_nodes_id, this->m_available_nodes_info,
            this->m_available_devices );

        SIXTRL_ASSERT( this->m_available_nodes_id.size() ==
                       this->m_available_nodes_info.size() );
    }

    CLContextBase::~CLContextBase() SIXTRL_NOEXCEPT
    {

    }

    CLContextBase::size_type
    CLContextBase::numAvailableNodes() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_available_nodes_id.size() ==
                       this->m_available_nodes_info.size() );

        return this->m_available_nodes_id.size();
    }

    CLContextBase::node_info_t const*
    CLContextBase::availableNodesInfoBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_available_nodes_info.data();
    }

    CLContextBase::node_info_t const*
    CLContextBase::availableNodesInfoEnd()   const SIXTRL_NOEXCEPT
    {
        CLContextBase::node_info_t const* ptr_end =
            this->availableNodesInfoBegin();

        if( ptr_end != nullptr )
        {
            std::advance( ptr_end, this->numAvailableNodes() );
        }

        return ptr_end;
    }

    bool CLContextBase::isNodeIdAvailable(
        CLContextBase::node_id_t const node_id ) const SIXTRL_NOEXCEPT
    {
        platform_id_t const platform_index =
            NS(ComputeNodeId_get_platform_id)( &node_id );

        device_id_t const device_index =
            NS(ComputeNodeId_get_device_id)( &node_id );

        return ( this->numAvailableNodes() >
                 this->findAvailableNodesIndex( platform_index, device_index ) );
    }

    bool CLContextBase::isNodeIdAvailable(
        CLContextBase::platform_id_t const platform_index,
        CLContextBase::device_id_t  const device_index ) const SIXTRL_NOEXCEPT
    {
        return ( this->numAvailableNodes() >
                 this->findAvailableNodesIndex( platform_index, device_index ) );
    }

    bool CLContextBase::isNodeIdAvailable( char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        return ( this->numAvailableNodes() >
                 this->findAvailableNodesIndex( node_id_str ) );
    }


    CLContextBase::node_id_t const* CLContextBase::ptrAvailableNodesId(
        CLContextBase::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_id[ index ] : nullptr;
    }

    CLContextBase::node_id_t const* CLContextBase::ptrAvailableNodesId(
        CLContextBase::platform_id_t const platform_index,
        CLContextBase::device_id_t   const device_index ) const SIXTRL_NOEXCEPT
    {
        size_type const index =
            this->findAvailableNodesIndex( platform_index, device_index );

        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_id[ index ] : nullptr;
    }

    CLContextBase::node_id_t const* CLContextBase::ptrAvailableNodesId(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        size_type const index = this->findAvailableNodesIndex( node_id_str );

        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_id[ index ] : nullptr;
    }

    CLContextBase::node_info_t const*
    CLContextBase::ptrAvailableNodesInfo(
        size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_info[ index ] : nullptr;
    }

    CLContextBase::node_info_t const*
    CLContextBase::ptrAvailableNodesInfo(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        size_type const index = this->findAvailableNodesIndex( node_id_str );

        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_info[ index ] : nullptr;
    }

    CLContextBase::node_info_t const*
    CLContextBase::ptrAvailableNodesInfo(
        CLContextBase::node_id_t const node_id ) const SIXTRL_NOEXCEPT
    {
        size_type const index = this->findAvailableNodesIndex(
                                    NS(ComputeNodeId_get_platform_id)( &node_id ),
                                    NS(ComputeNodeId_get_device_id)( &node_id ) );

        return ( index < this->numAvailableNodes() )
               ? &this->m_available_nodes_info[ index ] : nullptr;
    }

    bool CLContextBase::hasSelectedNode() const SIXTRL_NOEXCEPT
    {
        return ( ( this->m_selected_node_index >= 0 ) &&
                 ( this->numAvailableNodes() > static_cast< size_type >(
                       this->m_selected_node_index ) ) );
    }

    CLContextBase::node_id_t const*
    CLContextBase::ptrSelectedNodeId()     const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesId( static_cast< size_type >(
                                              this->m_selected_node_index ) );
    }

    CLContextBase::node_info_t const*
    CLContextBase::ptrSelectedNodeInfo()   const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesInfo( static_cast< size_type >(
                                                this->m_selected_node_index ) );
    }

    bool CLContextBase::selectNode( node_id_t const node_id )
    {
        platform_id_t const platform_idx =
            NS(ComputeNodeId_get_platform_id)( &node_id );

        device_id_t const device_idx =
            NS(ComputeNodeId_get_device_id)( &node_id );

        return this->selectNode( this->findAvailableNodesIndex(
                                     platform_idx, device_idx ) );
    }

    bool CLContextBase::selectNode( char const* node_id_str )
    {
        return this->selectNode( this->findAvailableNodesIndex( node_id_str ) );
    }

    bool CLContextBase::selectNode( size_type const index )
    {
        bool success = false;

        if( this->hasSelectedNode() )
        {
            SIXTRL_ASSERT( this->m_selected_node_index >= int64_t { 0 } );

            if( ( index != static_cast< size_type >( this->m_selected_node_index ) ) &&
                ( static_cast< size_type >( index ) < this->numAvailableNodes() ) )
            {
                cl::Device device = this->m_available_devices.at( index );

                cl::Context context( device );
                cl::CommandQueue queue( context, device, CL_QUEUE_PROFILING_ENABLE );

                this->m_cl_context = context;
                this->m_cl_queue   = queue;

                this->m_cl_programs.clear();
                this->m_cl_kernels.clear();

                if( !this->m_program_data.empty() )
                {
                    program_id_t program_id = program_id_t{ 0 };
                    program_id_t const num_programs = this->m_program_data.size();

                    for( ; program_id < num_programs ; ++program_id )
                    {
                        this->m_program_data[ program_id ].m_kernels.clear();

                        cl::Program program( this->m_cl_context,
                             this->m_program_data[ program_id ].m_source_code );

                        this->m_cl_programs.push_back( program );

                        bool const compiled = this->compileProgram( program_id );
                        this->m_program_data[ program_id ].m_compiled = compiled;
                    }

                    success = ( this->m_program_data.size() ==
                                this->m_cl_programs.size() );
                }
            }
        }

        return success;
    }

    void CLContextBase::clear()
    {
        return;
    }

    void CLContextBase::setDefaultCompileOptions(
        std::string const& compile_options_str )
    {
        this->setDefaultCompileOptions( compile_options_str.c_str() );
        return;
    }


    void CLContextBase::setDefaultCompileOptions( char const* compile_options_str )
    {
        SIXTRL_ASSERT( compile_options_str != nullptr );
        this->m_default_compile_options = compile_options_str;
        return;
    }

    char const* CLContextBase::defaultCompileOptions() const SIXTRL_NOEXCEPT
    {
        return this->m_default_compile_options.c_str();
    }

    CLContextBase::program_id_t
    CLContextBase::addProgramCode( std::string const& source_code )
    {
        return this->addProgramCode( source_code.c_str(),
                                     this->defaultCompileOptions() );
    }

    CLContextBase::program_id_t
    CLContextBase::addProgramCode( char const* source_code )
    {
        return this->addProgramCode( source_code,
                                     this->defaultCompileOptions() );
    }

    CLContextBase::program_id_t
    CLContextBase::addProgramCode(
        std::string const& source_code, std::string const& compile_options )
    {
        program_id_t program_id = program_id_t { -1 };

        if( ( !source_code.empty() ) &&
                ( ( ( !this->hasSelectedNode() ) &&
                    ( this->m_cl_programs.size() <= this->m_program_data.size() ) ) ||
                  (  this->m_cl_programs.size() == this->m_program_data.size() ) ) )
        {
            program_id = this->m_program_data.size();
            this->m_program_data.push_back( program_data_t {} );
            this->m_program_data.back().m_source_code = source_code;
            this->m_program_data.back().m_compile_options = compile_options;

            if( this->hasSelectedNode() )
            {
                cl::Program program( this->m_cl_context, source_code );
                this->m_cl_programs.push_back( program );

                SIXTRL_ASSERT( this->m_cl_programs.size() ==
                               this->m_program_data.size() );

                this->m_program_data.back().m_compiled =
                    this->compileProgram( program_id );
            }
        }

        return program_id;
    }

    CLContextBase::program_id_t
    CLContextBase::addProgramCode(
        char const* source_code, char const* compile_options )
    {
        std::string const str_source_code( ( source_code != nullptr )
                                           ? std::string( source_code ) : std::string() );

        std::string const str_compile_options( ( compile_options != nullptr )
                                               ? std::string( compile_options )
                                               : this->m_default_compile_options );

        return this->addProgramCode( str_source_code, str_compile_options );
    }

    CLContextBase::program_id_t
    CLContextBase::addProgramFile( std::string const& path_to_program )
    {
        return this->addProgramFile(
                   path_to_program, this->m_default_compile_options );
    }

    CLContextBase::program_id_t
    CLContextBase::addProgramFile( char const* path_to_program )
    {
        return ( path_to_program != nullptr )
               ? this->addProgramFile( std::string( path_to_program ) )
               : program_id_t { -1 };
    }

    CLContextBase::program_id_t
    CLContextBase::addProgramFile(
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

        return program_id_t { -1 };
    }

    CLContextBase::program_id_t
    CLContextBase::addProgramFile(
        char const* path_to_program, char const* compile_options )
    {
        std::string options_str = ( compile_options != nullptr )
                                  ? std::string( compile_options )
                                  : this->m_default_compile_options;

        return ( path_to_program != nullptr )
               ? this->addProgramCode( std::string( path_to_program ), options_str )
               : program_id_t { -1 };
    }

    CLContextBase::size_type
    CLContextBase::numAvailablePrograms() const SIXTRL_NOEXCEPT
    {
        return this->m_program_data.size();
    }

    CLContextBase::kernel_id_t CLContextBase::enableKernel(
        std::string const& kernel_name,
        CLContextBase::program_id_t const program_id )
    {
        return this->enableKernel( kernel_name.c_str(), program_id );
    }

    CLContextBase::kernel_id_t
    CLContextBase::enableKernel(
        char const* kernel_name, program_id_t const program_id )
    {
        kernel_id_t kernel_id = kernel_id_t { -1 };

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

            if( !program_data.m_compiled )
            {
                add_kernel = this->compileProgram( program_id );
            }

            if( ( !add_kernel ) && ( program_data.m_compiled ) )
            {
                add_kernel = true;

                auto it  = program_data.m_kernels.begin();
                auto end = program_data.m_kernels.end();

                for( ; it != end ; ++it )
                {
                    if( 0 == it->second.compare( kernel_name ) )
                    {
                        add_kernel = false;
                        break;
                    }
                }
            }

            if( ( add_kernel ) && ( program_data.m_compiled ) )
            {
                kernel_id_t kernel_id = static_cast< kernel_id_t >(
                                            this->m_cl_kernels.size() );

                cl::Kernel kernel(
                    this->m_cl_programs[ program_id ], kernel_name );

                this->m_cl_kernels.push_back( kernel );

                program_data.m_kernels[ kernel_id ] = std::string( kernel_name );
            }
        }

        return kernel_id;
    }

    CLContextBase::size_type
    CLContextBase::numAvailableKernels() const SIXTRL_NOEXCEPT
    {
        return this->m_cl_kernels.size();
    }

    cl::Program* CLContextBase::program(
        CLContextBase::program_id_t const program_id ) SIXTRL_NOEXCEPT
    {
        return ( ( program_id >= program_id_t{ 0 } ) &&
        ( this->m_cl_programs.size() >
            static_cast< size_type >( program_id ) ) )
        ? &this->m_cl_programs[ program_id ] : nullptr;
    }

    cl::Kernel* CLContextBase::kernel(
        CLContextBase::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        return ( ( kernel_id >= kernel_id_t{ 0 } ) &&
            ( this->m_cl_kernels.size() > static_cast< size_type >( kernel_id ) ) )
            ? &this->m_cl_kernels[ kernel_id ] : nullptr;
    }

    cl::CommandQueue* CLContextBase::queue() SIXTRL_NOEXCEPT
    {
        return &this->m_cl_queue;
    }

    CLContextBase::size_type
    CLContextBase::findAvailableNodesIndex(
        CLContextBase::platform_id_t const platform_index,
        CLContextBase::device_id_t const device_index ) const SIXTRL_NOEXCEPT
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

    CLContextBase::size_type
    CLContextBase::findAvailableNodesIndex(
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

    bool CLContextBase::compileProgram(
         CLContextBase::program_id_t const program_id )
    {
        bool success = false;

        if( (  program_id >= program_id_t{ 0 } ) &&
            (  static_cast< size_type >( program_id ) <
               this->m_program_data.size() ) &&
            (  this->m_program_data.size() == this->m_cl_programs.size() ) &&
            ( !this->m_program_data[ program_id ].m_compiled ) )
        {
            cl_int const cl_ret = this->m_cl_programs[ program_id ].build(
                this->m_program_data[ program_id ].m_compile_options.c_str() );

            if( cl_ret == CL_SUCCESS )
            {
                success = this->m_program_data[ program_id ].m_compiled = true;
            }
        }

        return success;
    }

    void CLContextBase::UpdateAvailableNodes(
        std::vector< CLContextBase::node_id_t>& available_nodes_id,
        std::vector< CLContextBase::node_info_t >& available_nodes_info,
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

            return;
        }
    }
}

/* ------------------------------------------------------------------------- */
/* -----             Implementation of C Wrapper functions              ---- */
/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN NS(CLContextBase)* NS(CLContextBase_create)()
{
    NS(CLContextBase)* ptr_base_ctx = new SIXTRL_NAMESPACE::CLContextBase;
    return ptr_base_ctx;
}

SIXTRL_HOST_FN NS(context_size_t) NS(CLContextBase_get_num_available_nodes)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->numAvailableNodes() : NS(context_size_t){ 0 };
}

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(CLContextBase_get_available_nodes_info_begin)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->availableNodesInfoBegin() : nullptr;

}

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(CLContextBase_get_available_nodes_info_end)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->availableNodesInfoEnd() : nullptr;
}

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(CLContextBase_get_available_node_info_by_index)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT ctx,
    NS(context_size_t) const node_index )
{
    return ( ctx != nullptr )
        ? ctx->ptrAvailableNodesInfo( node_index ) : nullptr;
}

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(CLContextBase_get_available_node_info_by_node_id)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT ctx,
    const NS(context_node_id_t) *const SIXTRL_RESTRICT node_id )
{
    return ( ( ctx != nullptr ) && ( node_id != nullptr ) )
        ? ctx->ptrAvailableNodesInfo( *node_id ) : nullptr;
}

SIXTRL_HOST_FN bool NS(CLContextBase_has_selected_node)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->hasSelectedNode() : false;
}

SIXTRL_HOST_FN NS(context_node_info_t) const* NS(CLContextBase_selected_get_node_info)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->ptrSelectedNodeInfo() : nullptr;
}

SIXTRL_HOST_FN NS(context_node_id_t) const* NS(CLContextBase_selected_get_node_id)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->ptrSelectedNodeId() : nullptr;
}

SIXTRL_HOST_FN void NS(CLContextBase_clear)(
    NS(CLContextBase)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->clear();
    return;
}

SIXTRL_HOST_FN bool NS(CLContextBase_select_node)(
    NS(CLContextBase)* SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT node_id_str )
{
    return ( ctx != nullptr ) ? ctx->selectNode( node_id_str ) : false;
}

SIXTRL_HOST_FN bool NS(CLContextBase_select_node_by_node_id)(
    NS(CLContextBase)* SIXTRL_RESTRICT ctx,
    NS(context_node_id_t) const node_id )
{
    return ( ctx != nullptr ) ? ctx->selectNode( node_id ) : false;
}

SIXTRL_HOST_FN bool NS(CLContextBase_select_node_by_index)(
    NS(CLContextBase)* SIXTRL_RESTRICT ctx,
    NS(context_size_t) const index )
{
    return ( ctx != nullptr ) ? ctx->selectNode( index ) : false;
}

SIXTRL_HOST_FN NS(CLContextBase)*
NS(CLContextBase_new)( char const* SIXTRL_RESTRICT node_id_str )
{
    NS(CLContextBase)* ctx = NS(CLContextBase_create)();

    if( ctx != nullptr )
    {
        using node_id_t = NS(CLContextBase)::node_id_t;
        node_id_t const* ptr_node_id = ctx->ptrAvailableNodesId( node_id_str );

        if( ( ptr_node_id == nullptr ) || ( !ctx->selectNode( *ptr_node_id ) ) )
        {
            delete ctx;
            ctx = nullptr;
        }
    }

    return ctx;
}

SIXTRL_HOST_FN void NS(CLContextBase_delete)(
    NS(CLContextBase)* SIXTRL_RESTRICT ctx )
{
    delete ctx;
    return;
}

#endif /* !defined( __CUDACC__ )  */

/* end: sixtracklib/opencl/code/cl_context_base.cpp */


