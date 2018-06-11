#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>

#include <CL/cl.hpp>

int main()
{
    std::vector< cl::Platform> platforms;
    cl::Platform::get( &platforms );
    
    if( !platforms.empty() )
    {
        std::cerr << "No OpenCL platforms available" << std::endl;
        return 1;
    }
    
    for( auto const& platform : platforms )
    {
        std::cout << "platform
        std::cout << "platform : " << platform.getInfo< CL_PLATFORM_NAME >() << "\r\n";
    }
    
    
