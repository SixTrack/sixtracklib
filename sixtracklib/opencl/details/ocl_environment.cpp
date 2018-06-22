#include "sixtracklib/opencl/ocl_environment.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <iostream>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include <CL/cl.hpp>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/path.h"

#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/compute_arch.h"

NS(OclEnvironment)::NS(OclEnvironment)() :
    m_ocl_platform_devices(),
    m_node_id_to_info_map(), 
    m_available_nodes( 1 ),
    m_selected_nodes( 1 ), 
    m_contexts( 1 ),
    m_programs( 1 ),
    m_queues( 1 ),
    m_kernels( 1 ),
    m_buffers( 1 ),
    m_success_flags( 1 ),
    m_preferred_work_group_multi( 1 ),
    m_local_workgroup_size( 1 ),
    m_particles_data_buffer_size( 0u ),
    m_beam_elements_data_buffer_size( 0u ),
    m_elem_by_elem_data_buffer_size( 0u ),
    m_num_of_particles( 0u ),
    m_num_of_platforms( 0u )
{
    NS(comp_node_id_num_t) next_platform_id = 0;
    std::string const arch( "opencl" );
    
    this->m_available_nodes.clear();
    this->m_selected_nodes.clear();
    this->m_contexts.clear();
    this->m_programs.clear();
    this->m_queues.clear();
    this->m_kernels.clear();
    this->m_buffers.clear();
    this->m_success_flags.clear();
    this->m_preferred_work_group_multi.clear();
    this->m_local_workgroup_size.clear();
    
    
    
    std::vector< cl::Platform > platforms;
    cl::Platform::get( &platforms );
        
    for( auto const& platform : platforms )
    {
        std::vector< cl::Device > devices;
        platform.getDevices( CL_DEVICE_TYPE_ALL, &devices );
        
        NS(comp_node_id_num_t) next_device_id   = 0;
        
        std::string const platform_name = 
            platform.getInfo< CL_PLATFORM_NAME >();
        
        std::cout << platform_name << std::endl;
            
        for( auto const& device : devices )
        {
            std::string name;
            std::string description;
            
            cl_int ret = device.getInfo( CL_DEVICE_NAME, &name );
            ret |= device.getInfo( CL_DEVICE_EXTENSIONS, &description );
            
            NS(ComputeNodeId) node_id;
            NS(ComputeNodeId_set_platform_id)( &node_id, next_platform_id );
            NS(ComputeNodeId_set_device_id)( &node_id, next_device_id++ );
            
            NS(ComputeNodeInfo) node_info;
            NS(ComputeNodeInfo_preset)( &node_info );
            
            this->m_ocl_platform_devices.insert(
                std::make_pair( node_id, std::make_pair( platform, device ) ) );
            
            auto insert_result = this->m_node_id_to_info_map.insert(
                std::make_pair( node_id, node_info ) );
            
            if( insert_result.second )
            {
                NS(ComputeNodeInfo)* ptr_info = &insert_result.first->second;                
                
                if( nullptr != NS(ComputeNodeInfo_reserve)( 
                    ptr_info, arch.size(), platform_name.size(),
                    name.size(), description.size() ) )
                {
                    ptr_info->id = node_id;
                    
                    std::strncpy( ptr_info->arch, arch.c_str(), arch.size() );
                    std::strncpy( ptr_info->name, name.c_str(), name.size() );
                
                    if( !platform_name.empty() )
                    {
                        std::strncpy( ptr_info->platform, platform_name.c_str(),
                                      platform_name.size() );                        
                    }
                    
                    if( !description.empty() )
                    {
                        std::strncpy( ptr_info->description, 
                                      description.c_str(), 
                                      description.size() );
                    }
                }
            }
            
            this->m_available_nodes.push_back( node_id );
        }
        
        if( !devices.empty() )
        {
            ++next_platform_id;
            ++this->m_num_of_platforms;
        }
    }
}

NS(OclEnvironment)::~NS(OclEnvironment)() noexcept
{
    for( auto& key_value : this->m_node_id_to_info_map )
    {
        NS(ComputeNodeInfo_free)( &key_value.second );
    }
}

NS(OclEnvironment)::node_id_t const* 
NS(OclEnvironment)::constAvailableNodesBegin() const noexcept
{
    return this->m_available_nodes.data();
}

NS(OclEnvironment)::node_id_t const* 
NS(OclEnvironment)::constAvailableNodesEnd() const noexcept
{
    return ( this->m_available_nodes.size() > 0u )
         ? ( this->m_available_nodes.data() + this->m_available_nodes.size() )
         : ( this->m_available_nodes.data() );
}

std::size_t NS(OclEnvironment)::numAvailableNodes() const noexcept
{
    return this->m_available_nodes.size();
}

NS(OclEnvironment)::node_info_t const* 
NS(OclEnvironment)::getPtrNodeInfo( 
    NS(OclEnvironment)::node_id_t const id ) const noexcept
{
    NS(OclEnvironment)::node_info_t const* ptr_info = nullptr;
    
    auto find_it = this->m_node_id_to_info_map.find( id );
    
    if( find_it != this->m_node_id_to_info_map.end() )
    {
        ptr_info = &find_it->second;
    }
    
    return ptr_info;
}

bool NS(OclEnvironment)::prepareParticlesTracking(
        NS(Blocks) const& SIXTRL_RESTRICT_REF particles_buffer, 
        NS(Blocks) const& SIXTRL_RESTRICT_REF beam_elements,
        NS(Blocks) const* SIXTRL_RESTRICT elem_by_elem_buffer, 
        NS(ComputeNodeId) const* SIXTRL_RESTRICT selected_nodes_begin,
        std::size_t const num_of_selected_nodes )
{
    bool success = false;
    
    if( ( selected_nodes_begin != nullptr ) &&
        ( num_of_selected_nodes > 0u ) &&
        ( NS(Blocks_are_serialized)( &particles_buffer ) ) &&
        ( NS(Blocks_get_num_of_blocks)( &particles_buffer ) > 0 ) &&
        ( NS(Blocks_get_total_num_bytes)( &particles_buffer ) > 0u ) &&
        ( NS(Blocks_are_serialized)( &beam_elements ) ) &&
        ( NS(Blocks_get_num_of_blocks)( &beam_elements ) > 0 ) &&
        ( NS(Blocks_get_total_num_bytes)( &beam_elements ) > 0u ) &&
        ( ( elem_by_elem_buffer == nullptr ) ||
          ( ( NS(Blocks_are_serialized)( elem_by_elem_buffer ) ) &&
            ( NS(Blocks_get_num_of_blocks)( elem_by_elem_buffer ) > 0 ) &&
            ( NS(Blocks_get_total_num_bytes)( elem_by_elem_buffer ) > 0u ) ) 
        ) )
    {
        this->m_num_of_particles = uint64_t{ 0 };
        
        auto particles_it  = NS(Blocks_get_const_block_infos_begin)( &particles_buffer );
        auto particles_end = NS(Blocks_get_const_block_infos_end)( &particles_buffer );
        
        for( ; particles_it != particles_end ; ++particles_it )
        {
            NS(Particles) const* particles = 
                NS(Blocks_get_const_particles)( particles_it );
                
            this->m_num_of_particles += 
                NS(Particles_get_num_particles)( particles );
        }
        
        NS(ComputeNodeId) const* sel_node_it = selected_nodes_begin;
        NS(ComputeNodeId) const* selected_nodes_end = selected_nodes_begin;
        std::advance( selected_nodes_end, num_of_selected_nodes );
        
        std::string PATH_TO_SOURCE_DIR( st_PATH_TO_BASE_DIR );
        PATH_TO_SOURCE_DIR += "sixtracklib/";
        
        std::vector< std::string > const paths_to_kernel_files{
            PATH_TO_SOURCE_DIR + std::string{ "_impl/namespace_begin.h" },
            PATH_TO_SOURCE_DIR + std::string{ "_impl/definitions.h" },
            PATH_TO_SOURCE_DIR + std::string{ "common/blocks.h" },
            PATH_TO_SOURCE_DIR + std::string{ "common/impl/particles_type.h" },
            PATH_TO_SOURCE_DIR + std::string{ "common/impl/particles_api.h" },
            PATH_TO_SOURCE_DIR + std::string{ "common/particles.h" },
            PATH_TO_SOURCE_DIR + std::string{ "common/impl/beam_elements_type.h" },
            PATH_TO_SOURCE_DIR + std::string{ "common/impl/beam_elements_api.h" },
            PATH_TO_SOURCE_DIR + std::string{ "common/beam_elements.h" },
            PATH_TO_SOURCE_DIR + std::string{ "common/track.h" },
            PATH_TO_SOURCE_DIR + std::string{ "common/impl/track_api.h" },
            PATH_TO_SOURCE_DIR + std::string{ "opencl/track_particles_kernel.cl" },
            PATH_TO_SOURCE_DIR + std::string{ "_impl/namespace_end.h" }
        };
        
        std::string kernel_source( 2048 * 1024, '\0' );
        kernel_source.clear();
        
        for( auto const& path : paths_to_kernel_files )
        {
            std::ifstream const one_kernel_file( path, std::ios::in );
            
            std::istreambuf_iterator< char > one_kernel_file_begin( 
                one_kernel_file.rdbuf() );
            std::istreambuf_iterator< char > end_of_file;
            
            kernel_source.insert( kernel_source.end(), 
                                one_kernel_file_begin, end_of_file );
        }
        
        this->m_selected_nodes.clear();
        this->m_selected_nodes.reserve( num_of_selected_nodes );
        
        this->m_contexts.clear();
        this->m_contexts.reserve( num_of_selected_nodes );
        
        this->m_programs.clear();
        this->m_programs.reserve( num_of_selected_nodes );
        
        this->m_queues.clear();
        this->m_queues.reserve( num_of_selected_nodes );
        
        this->m_kernels.clear();
        this->m_kernels.reserve( num_of_selected_nodes );
        
        this->m_buffers.clear();
        this->m_buffers.reserve( num_of_selected_nodes );
        
        this->m_success_flags.clear();
        this->m_success_flags.resize( num_of_selected_nodes, int64_t{ 0 } );
        
        this->m_local_workgroup_size.clear();
        this->m_local_workgroup_size.reserve( num_of_selected_nodes );
        
        this->m_preferred_work_group_multi.clear();
        this->m_preferred_work_group_multi.reserve( num_of_selected_nodes );
        
        this->m_particles_data_buffer_size = 
            NS(Blocks_get_total_num_bytes)( &particles_buffer );
            
        this->m_beam_elements_data_buffer_size =
            NS(Blocks_get_total_num_bytes)( &particles_buffer );
            
        if( elem_by_elem_buffer != nullptr )
        {
            this->m_elem_by_elem_data_buffer_size = 
                NS(Blocks_get_total_num_bytes)( elem_by_elem_buffer );
        }
        else
        {
            this->m_elem_by_elem_data_buffer_size = 4 * sizeof( uint64_t );
        }
        
        std::string const compile_options( "-D_GPUCODE=1 -D__NAMESPACE=st_" );
        
        success = true;
        
        for( ; sel_node_it != selected_nodes_end ; ++sel_node_it )
        {
            auto find_it = this->m_ocl_platform_devices.find( *sel_node_it );
            
            if( find_it != this->m_ocl_platform_devices.end() )
            {
                cl_int ret = CL_SUCCESS;
                
                cl::Platform platform( find_it->second.first );
                cl::Device   device(   find_it->second.second );
                
                this->m_selected_nodes.push_back( *sel_node_it );
                this->m_contexts.push_back( cl::Context( device ) );
                cl::Context& context = this->m_contexts.back();
                
                /* -------------------------------------------------------- */
                
                this->m_buffers.emplace_back( std::vector< cl::Buffer >( 4 ) );
                std::vector< cl::Buffer >& buffers = this->m_buffers.back();
                
                buffers[ 0 ] = cl::Buffer( context, CL_MEM_READ_WRITE, 
                    this->m_particles_data_buffer_size, 0, &ret );
                
                success &= ( ret == CL_SUCCESS );
                
                buffers[ 1 ] = cl::Buffer( context, CL_MEM_READ_WRITE,
                    this->m_beam_elements_data_buffer_size, 0, &ret );
                
                success &= ( ret == CL_SUCCESS );
                
                buffers[ 2 ] = cl::Buffer( context, CL_MEM_READ_WRITE,
                     this->m_elem_by_elem_data_buffer_size, 0, &ret );
                
                success &= ( ret == CL_SUCCESS );
                
                buffers[ 3 ] = cl::Buffer( context, CL_MEM_READ_WRITE, 
                    sizeof( int64_t ), 0, &ret );
                
                success &= ( ret == CL_SUCCESS );
                
                if( !success ) break;
                
                /* -------------------------------------------------------- */
                
                this->m_programs.push_back( 
                    cl::Program( context, kernel_source ) );
                
                cl::Program& program = this->m_programs.back();
                
                if( ( program.build( compile_options.c_str() ) ) != CL_SUCCESS )
                {
                    std::string const build_log = 
                        program.getBuildInfo< CL_PROGRAM_BUILD_LOG >( device );
                    
                    std::cout << "Error building: " << build_log << "\r\n";
                    
                    success = false;
                }
                
                if( !success ) break;
                
                /* -------------------------------------------------------- */
                
                this->m_kernels.emplace_back( std::vector< cl::Kernel >( 2 ) );
                std::vector< cl::Kernel >& kernels = this->m_kernels.back();
                
                success &= ( kernels.size() == 2u );
                
                kernels[ 0 ] = cl::Kernel( program, 
                       "Track_remap_serialized_blocks_buffer", &ret );                
                
                success &= ( ret == CL_SUCCESS );
                
                kernels[ 1 ] = cl::Kernel( 
                    program, "Track_particles_kernel_opencl", &ret );
                
                success &= ( ret == CL_SUCCESS );
                
                cl::Kernel& track_kernel = kernels[ 1 ];
                
                this->m_local_workgroup_size.push_back(
                    track_kernel.getWorkGroupInfo<
                        CL_KERNEL_WORK_GROUP_SIZE >( device ) );
                
                this->m_preferred_work_group_multi.push_back(
                    track_kernel.getWorkGroupInfo< 
                        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE >( 
                            device ) );
                
                /* -------------------------------------------------------- */
                
                this->m_queues.push_back( 
                    cl::CommandQueue( context, device, 
                                      CL_QUEUE_PROFILING_ENABLE, &ret ) );
                
                success &= ( ret == CL_SUCCESS );
            }
        }
    }
    
    return success;
}


bool NS(OclEnvironment)::runParticlesTracking(
    uint64_t const num_of_turns, 
    struct NS(Blocks)& SIXTRL_RESTRICT_REF particles_buffer, 
    struct NS(Blocks)& SIXTRL_RESTRICT_REF beam_elements,
    struct NS(Blocks)* SIXTRL_RESTRICT elem_by_elem_buffer )
{
    bool success = false;
    
    if( (  this->m_selected_nodes.empty() ) &&
        ( !this->m_available_nodes.empty() ) )
    {
        cl_int ret = CL_SUCCESS;
        cl::Platform default_platform = cl::Platform::getDefault( &ret );
        
        if( ret == CL_SUCCESS )
        {
            cl_platform_id platform_id = default_platform();
            
            std::vector< cl::Device > devices;
            default_platform.getDevices(  CL_DEVICE_TYPE_DEFAULT, &devices );
            
            NS(ComputeNodeId) default_node_id;
            NS(ComputeNodeId_preset)( &default_node_id );
            
            if( !devices.empty() )
            {
                cl_device_id const device_id = devices.front()();
                
                for( auto const& id_ocl_pair : this->m_ocl_platform_devices )
                {
                    if( ( id_ocl_pair.second.first() == platform_id ) &&
                        ( id_ocl_pair.second.second() == device_id ) )
                    {
                        default_node_id = id_ocl_pair.first;
                        break;
                    }                    
                }
            }
            
            if( NS(ComputeNodeId_is_valid)( &default_node_id ) )                
            {
                this->prepareParticlesTracking( particles_buffer, 
                    beam_elements, elem_by_elem_buffer, &default_node_id, 1u );
            }
        }            
    }
    
    if( !this->m_selected_nodes.empty() )        
    {
        std::size_t const num_of_nodes = this->m_selected_nodes.size();
        
        /* Temporary until we figured out what to do with multiple compute 
         * nodes at the same time */
        
        if( num_of_nodes != 1u )
        {
            return false;
        }
        
        SIXTRL_ASSERT( num_of_nodes <= this->m_available_nodes.size() );
        SIXTRL_ASSERT( num_of_nodes == this->m_contexts.size() );
        SIXTRL_ASSERT( num_of_nodes == this->m_programs.size() );
        SIXTRL_ASSERT( num_of_nodes == this->m_queues.size() );
        SIXTRL_ASSERT( num_of_nodes == this->m_kernels.size() );
        SIXTRL_ASSERT( num_of_nodes == this->m_buffers.size() );
        SIXTRL_ASSERT( num_of_nodes == this->m_success_flags.size() );
        
        SIXTRL_ASSERT( num_of_nodes == 
                       this->m_preferred_work_group_multi.size() );
        
        SIXTRL_ASSERT( num_of_nodes ==
                       this->m_local_workgroup_size.size() );
        
        for( std::size_t jj = 0 ; jj < num_of_nodes ; ++jj )
        {
            std::vector< cl::Kernel >& kernels = this->m_kernels[ jj ];
            
            SIXTRL_ASSERT( kernels.size() == 2u );
            
            cl::Kernel& remap_kernel = kernels[ 0 ];
            
            std::vector< cl::Buffer >& buffers = this->m_buffers[ jj ];
            SIXTRL_ASSERT( buffers.size() == 4u );
            
            cl_int ret  = CL_SUCCESS;
            ret |= remap_kernel.setArg( 0, buffers[ 0 ] );
            ret |= remap_kernel.setArg( 1, buffers[ 1 ] );
            ret |= remap_kernel.setArg( 2, buffers[ 2 ] );
            ret |= remap_kernel.setArg( 3, buffers[ 3 ] );
            
            success = ( ret == CL_SUCCESS );
            
            cl::Kernel& track_kernel = kernels[ 1 ];
            
            ret  = track_kernel.setArg( 0, num_of_turns );
            ret |= track_kernel.setArg( 1, buffers[ 0 ] );
            ret |= track_kernel.setArg( 2, buffers[ 1 ] );
            ret |= track_kernel.setArg( 3, buffers[ 2 ] );
            ret |= track_kernel.setArg( 4, buffers[ 3 ] );
            
            success &= ( ret == CL_SUCCESS );
            
            if( !success ) break;
            
            /* ------------------------------------------------------------- */
            
            size_t global_work_size = this->m_num_of_particles;
            
            size_t const local_work_group_size = 
                this->m_local_workgroup_size[ jj ];
            
            if( ( global_work_size % local_work_group_size ) != 0u )
            {
                global_work_size /= local_work_group_size;
                ++global_work_size;
                
                global_work_size *= local_work_group_size;
            }
            
            /* ------------------------------------------------------------- */
            
            cl::CommandQueue& queue = this->m_queues[ jj ];
                        
            uint64_t dummy_elem_by_elem_buffer[ 4 ] = { 0u, 0u, 0u, 0u };
            
            unsigned char* particles_data_buffer = 
                NS(Blocks_get_data_begin)( &particles_buffer );
                
            unsigned char* beam_elements_data_buffer =
                NS(Blocks_get_data_begin)( &beam_elements );
                
            unsigned char* elem_by_elem_data_buffer = 
                ( elem_by_elem_buffer != nullptr )
                    ? static_cast< unsigned char* >( 
                        NS(Blocks_get_data_begin)( elem_by_elem_buffer ) )
                    : reinterpret_cast< unsigned char* >( 
                        &dummy_elem_by_elem_buffer[ 0 ] );
            
            cl_long success_flag = 0;
            
            SIXTRL_ASSERT( particles_data_buffer     != nullptr );
            SIXTRL_ASSERT( beam_elements_data_buffer != nullptr );
            SIXTRL_ASSERT( elem_by_elem_data_buffer  != nullptr );
            
            ret =  queue.enqueueWriteBuffer( 
                        buffers[ 0 ], CL_TRUE, 0,
                        this->m_particles_data_buffer_size, 
                        particles_data_buffer );
            
            ret |= queue.enqueueWriteBuffer(
                        buffers[ 1 ], CL_TRUE, 0, 
                        this->m_beam_elements_data_buffer_size,
                        beam_elements_data_buffer );
            
            ret |= queue.enqueueWriteBuffer(
                        buffers[ 2 ], CL_TRUE, 0, 
                        this->m_elem_by_elem_data_buffer_size,
                        elem_by_elem_data_buffer );
            
            ret |= queue.enqueueWriteBuffer( buffers[ 3 ], CL_TRUE, 0, 
                        sizeof( cl_long ), &success_flag );
            
            success &= ( ret == CL_SUCCESS );
            
            if( !success ) break;
            
            /* ------------------------------------------------------------- */
            
            cl::Event event_remap;
    
            ret = queue.enqueueNDRangeKernel(
                        remap_kernel, cl::NullRange, 
                        cl::NDRange( this->m_preferred_work_group_multi[ jj ] ),
                        cl::NDRange( this->m_preferred_work_group_multi[ jj ] ), 
                        nullptr, &event_remap );    
            
            success &= ( ret == CL_SUCCESS );
            
            ret = queue.flush();
            success &= ( ret == CL_SUCCESS );
            
            ret = event_remap.wait();
            success &= ( ret == CL_SUCCESS );
            
            /* ------------------------------------------------------------- */
            
            ret = queue.enqueueReadBuffer( buffers[ 3 ], CL_TRUE, 0,
                                   sizeof( cl_long ), &success_flag );
    
            success &= ( ( success_flag == 0 ) && ( ret == CL_SUCCESS ) );
            if( !success ) break;
            
            /* ------------------------------------------------------------- */
            
            cl::Event event_track;
    
            ret = queue.enqueueNDRangeKernel(
                        track_kernel, cl::NullRange, cl::NDRange( global_work_size ),
                        cl::NDRange( local_work_group_size ), nullptr, &event_track );
            
            success &= ( ret == CL_SUCCESS );
            
            ret = queue.flush();
            success &= ( ret == CL_SUCCESS );
            
            ret = event_track.wait();
            success &= ( ret == CL_SUCCESS );
            
            /* ------------------------------------------------------------- */
            
            ret  = queue.enqueueReadBuffer( buffers[ 3 ], CL_TRUE, 
                    0, sizeof( cl_long ), &success_flag );
            
            success &= ( ( success_flag == 0 ) && ( ret == CL_SUCCESS ) );
            
            ret  = queue.enqueueReadBuffer(
                    buffers[ 0 ], CL_TRUE, 0, 
                    this->m_particles_data_buffer_size, 
                    particles_data_buffer );
            
            success &= ( ret == CL_SUCCESS );
            
            if( success )
            {
                success = ( 0 == st_Blocks_unserialize( 
                    &particles_buffer, particles_data_buffer ) );
            }
            
            ret = queue.enqueueReadBuffer(
                    buffers[ 2 ], CL_TRUE, 0, 
                    this->m_elem_by_elem_data_buffer_size, 
                    elem_by_elem_data_buffer );
            
            success &= ( ret == CL_SUCCESS );
            
            if( ( success ) && ( elem_by_elem_buffer != nullptr ) )
            {
                success = ( 0 == NS(Blocks_unserialize)(
                    elem_by_elem_buffer, elem_by_elem_data_buffer ) );
            }
            
            if( !success ) break;
        }
    }
    
    return success;
}

/* ------------------------------------------------------------------------- */
/* ---  Wrapper functions with extern "C" binding for the C- interface   --- */
/* ------------------------------------------------------------------------- */

NS(OclEnvironment)* NS(OclEnvironment_new)()
{
    NS(OclEnvironment)* ptr_to_ocl_env = new NS(OclEnvironment);
    return ptr_to_ocl_env;
}

void NS(OclEnvironment_free)( NS(OclEnvironment)* SIXTRL_RESTRICT ocl_env )
{
    if( ocl_env != nullptr )
    {
        delete ocl_env;
        ocl_env = nullptr;
    }
    
    return;
}

SIXTRL_HOST_FN NS(ComputeNodeId) const* 
NS(OclEnvironment_get_available_nodes_begin)( 
    const NS(OclEnvironment) *const SIXTRL_RESTRICT ocl_env );

NS(ComputeNodeId) const* NS(OclEnvironment_get_available_nodes_begin)( 
    const NS(OclEnvironment) *const SIXTRL_RESTRICT ocl_env )
{
    return ( ocl_env != nullptr ) 
        ? ocl_env->constAvailableNodesBegin() : nullptr;
}

NS(ComputeNodeId) const* NS(OclEnvironment_get_available_nodes_end)( 
    const NS(OclEnvironment) *const SIXTRL_RESTRICT ocl_env )
{
    return ( ocl_env != nullptr ) 
        ? ocl_env->constAvailableNodesEnd() : nullptr;
}

SIXTRL_SIZE_T NS(OclEnvironment_get_num_available_nodes)(
    const NS(OclEnvironment) *const SIXTRL_RESTRICT ocl_env )
{
    return ( ocl_env != nullptr )
        ? ocl_env->numAvailableNodes() : ( SIXTRL_SIZE_T )0u;
}


NS(ComputeNodeInfo) const* NS(OclEnvironment_get_ptr_node_info)( 
    const NS(OclEnvironment) *const SIXTRL_RESTRICT ocl_env, 
    NS(ComputeNodeId) const* SIXTRL_RESTRICT node_id )
{
    return ( ( ocl_env != nullptr ) && ( node_id != nullptr ) )
        ? ocl_env->getPtrNodeInfo( *node_id ) : nullptr;
}

int NS(OclEnvironment_prepare_particles_tracking)(
    NS(OclEnvironment)* SIXTRL_RESTRICT ocl_env,
    const struct NS(Blocks) *const SIXTRL_RESTRICT particles_buffer, 
    const struct NS(Blocks) *const SIXTRL_RESTRICT beam_elements,
    const struct NS(Blocks) *const SIXTRL_RESTRICT elem_by_elem_buffer,
    NS(ComputeNodeId) const* selected_nodes_begin,
    size_t const num_of_selected_nodes )
{
    return ( ( ocl_env != nullptr ) && ( particles_buffer != nullptr ) &&
             ( beam_elements != nullptr ) )
        ? ocl_env->prepareParticlesTracking(
            *particles_buffer, *beam_elements, elem_by_elem_buffer,
            selected_nodes_begin, num_of_selected_nodes )
        : false;
}

int NS(OclEnvironment_run_particle_tracking)(
    NS(OclEnvironment)* SIXTRL_RESTRICT ocl_env,
    uint64_t const num_of_turns,
    struct NS(Blocks)* SIXTRL_RESTRICT particles_buffer, 
    struct NS(Blocks)* SIXTRL_RESTRICT beam_elements,
    struct NS(Blocks)* SIXTRL_RESTRICT elem_by_elem_buffer )
{
    return ( ( ocl_env != nullptr ) && ( particles_buffer != nullptr ) &&
             ( beam_elements != nullptr ) )
        ? ocl_env->runParticlesTracking( num_of_turns, 
                *particles_buffer, *beam_elements, elem_by_elem_buffer )
        : false;
}

/* end: sixtracklib/opencl/details/ocl_environment.cpp */
