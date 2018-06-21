#ifndef SIXTRACKLIB_OPENCL_CL_ENVIRONMENT_H__
#define SIXTRACKLIB_OPENCL_CL_ENVIRONMENT_H__

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/compute_arch.h"

#if defined( __cplusplus )

#include <map>
#include <string>
#include <vector>
#include <utility>

#include <CL/cl.hpp>

class NS(OclEnvironment)
{
    public:
    
    typedef NS(ComputeNodeId)   node_id_t;
    typedef NS(ComputeNodeInfo) node_info_t;
    
    NS(OclEnvironment)();
    
    NS(OclEnvironment)( NS(OclEnvironment) const& other ) = delete;
    NS(OclEnvironment)( NS(OclEnvironment)&& other ) = delete;
    
    NS(OclEnvironment)& operator=( NS(OclEnvironment) const& rhs ) = delete;
    NS(OclEnvironment)& operator=( NS(OclEnvironment)&& rhs ) = delete;
    
    virtual ~NS(OclEnvironment)() noexcept;
    
    node_id_t const* constAvailableNodesBegin() const noexcept;
    node_id_t const* constAvailableNodesEnd()   const noexcept;
    std::size_t numAvailableNodes() const noexcept;
    
    node_info_t const* getPtrNodeInfo( node_id_t const id ) const noexcept;
    
    private:
    
    using ocl_platform_dev_pair_t = 
        std::pair< cl::Platform, cl::Device >;
        
    using ocl_platform_dev_map_t  = 
        std::map< node_id_t, ocl_platform_dev_pair_t >;
        
    using ocl_node_id_to_node_info_map_t = std::map< node_id_t, node_info_t >;
        
    ocl_platform_dev_map_t          m_ocl_platform_devices;    
    ocl_node_id_to_node_info_map_t  m_node_id_to_info_map;
    
    std::vector< node_id_t >        m_available_nodes;
    std::vector< node_id_t >        m_selected_nodes;
};

#else

#include <CL/cl.h>

typedef void NS(OclEnvironment);

#endif /* defined( __cplusplus ) */
    
#if defined( __cplusplus )

extern "C" 
{

#endif /* !defined( __cplusplus ) */

SIXTRL_HOST_FN NS(ComputeNodeId) const* 
NS(OclEnvironment_get_available_nodes_begin)( 
    const NS(OclEnvironment) *const SIXTRL_RESTRICT ocl_env );

SIXTRL_HOST_FN NS(ComputeNodeId) const* 
NS(OclEnvironment_get_available_nodes_end)( 
    const NS(OclEnvironment) *const SIXTRL_RESTRICT ocl_env );

SIXTRL_HOST_FN SIXTRL_SIZE_T NS(OclEnvironment_get_num_available_nodes)(
    const NS(OclEnvironment) *const SIXTRL_RESTRICT ocl_env );

SIXTRL_HOST_FN NS(ComputeNodeInfo) const* 
NS(OclEnvironment_get_ptr_node_info)( 
    const NS(OclEnvironment) *const SIXTRL_RESTRICT ocl_env, 
    NS(ComputeNodeId) const* SIXTRL_RESTRICT  node_id );

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */
    
#endif /* SIXTRACKLIB_OPENCL_CL_ENVIRONMENT_H__ */
    
/* end: sixtracklib/opencl/ocl_environment.h */
