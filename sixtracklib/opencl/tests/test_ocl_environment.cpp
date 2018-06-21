#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <CL/cl.hpp>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/compute_arch.h"
#include "sixtracklib/opencl/ocl_environment.h"

TEST( OpenCLOclEnvironmentTests, BasicUsage )
{
    std::vector< cl::Platform > platforms;
    cl::Platform::get( &platforms );
    
    st_OclEnvironment ocl_environment;
    
    NS(ComputeNodeId) const* nodes_it  = 
        st_OclEnvironment_get_available_nodes_begin( &ocl_environment );
        
    NS(ComputeNodeId) const* nodes_end = 
        st_OclEnvironment_get_available_nodes_end( &ocl_environment );
    
    for( ; nodes_it != nodes_end ; ++nodes_it )
    {
        auto node_info = st_OclEnvironment_get_ptr_node_info( 
            &ocl_environment, nodes_it );
        
        ASSERT_TRUE( node_info != nullptr );
        
        std::cout 
            << " --------------------------------------------------------\r\n"
            << "platform_id  : " 
            << st_ComputeNodeInfo_get_platform_id( node_info ) << "\r\n"
            << "device_id    : " 
            << st_ComputeNodeInfo_get_device_id( node_info ) << "\r\n"
            << "\r\n"
            << "architecture : "
            << st_ComputeNodeInfo_get_arch( node_info ) << "\r\n"
            << "platform     : "
            << st_ComputeNodeInfo_get_platform( node_info ) << "\r\n"
            << "device name  : "
            << st_ComputeNodeInfo_get_name( node_info ) << "\r\n"
            << "description  : " 
            << st_ComputeNodeInfo_get_description( node_info ) << "\r\n"
            << "\r\n";
    }    
}

/* end: sixtracklib/opencl/tests/test_ocl_environment.cpp */
