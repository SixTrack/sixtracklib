#include "sixtracklib/opencl/ocl_environment.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include <CL/cl.hpp>

NS(OclEnvironment)::NS(OclEnvironment)() :
    m_ocl_platform_devices(),
    m_node_id_to_info_map(), 
    m_available_nodes( 100 ),
    m_selected_nodes( 100 )
{
    NS(comp_node_id_num_t) next_platform_id = 0;
    std::string const arch( "opencl" );
    
    this->m_available_nodes.clear();
    this->m_selected_nodes.clear();
    
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
            
            std::cout << " -> " << node_id.platform_id 
                      << " / " 
                      << node_id.device_id
                      << " :: " << name << std::endl;
            
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
        
        ++next_platform_id;
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

/* ------------------------------------------------------------------------- */
/* ---  Wrapper functions with extern "C" binding for the C- interface   --- */
/* ------------------------------------------------------------------------- */

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

/* end: sixtracklib/opencl/details/ocl_environment.cpp */
