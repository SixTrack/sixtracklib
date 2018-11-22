#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>
#include <CL/cl.hpp>

#include "sixtracklib/testlib.h"

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/path.h"
#include "sixtracklib/common/impl/managed_buffer.h"

TEST( C99_OpenCLManagedBuffer, InitWithGenericObjDataCopyToDeviceCopyBackCmp )
{
    using type_id_t     = ::st_object_type_id_t;
    using object_t      = ::st_Object;
    using ptr_to_obj_t  = object_t*;
    using size_t        = ::st_buffer_size_t;
    using raw_t         = unsigned char;

    static SIXTRL_CONSTEXPR_OR_CONST size_t SLOTS_ID    = size_t{ 3 };
    static SIXTRL_CONSTEXPR_OR_CONST size_t OBJECTS_ID  = size_t{ 4 };
    static SIXTRL_CONSTEXPR_OR_CONST size_t DATAPTRS_ID = size_t{ 5 };
    static SIXTRL_CONSTEXPR_OR_CONST size_t GARBAGE_ID  = size_t{ 6 };

    size_t const max_num_objects  = size_t{ 2 };
    size_t const max_num_slots    = size_t{ 8 };
    size_t const max_num_dataptrs = size_t{ 4 };
    size_t const max_num_garbage  = size_t{ 1 };
    size_t const slot_size        = ::st_BUFFER_DEFAULT_SLOT_SIZE;

    size_t const buf_size = st_ManagedBuffer_calculate_buffer_length( nullptr,
        max_num_objects, max_num_slots, max_num_dataptrs, max_num_garbage );

    size_t current_orig_buffer_size = size_t{ 0 };

    ASSERT_TRUE( buf_size > size_t{ 0 } );

    std::vector< raw_t > orig_buffer( buf_size, raw_t{ 0 } );
    std::vector< raw_t > copy_buffer( buf_size, raw_t{ 0 } );

    int success = ::st_ManagedBuffer_init( orig_buffer.data(),
        &current_orig_buffer_size, max_num_objects, max_num_slots,
            max_num_dataptrs, max_num_garbage, orig_buffer.size(), slot_size );

    ASSERT_TRUE( success == 0 );
    ASSERT_TRUE( current_orig_buffer_size <= orig_buffer.size() );
    ASSERT_TRUE( current_orig_buffer_size <= copy_buffer.size() );

    ptr_to_obj_t obj_it  = reinterpret_cast< ptr_to_obj_t >(
        st_ManagedBuffer_get_ptr_to_section_data(
            orig_buffer.data(), OBJECTS_ID, slot_size ) );

    ptr_to_obj_t  obj_end = obj_it;
    std::advance( obj_end, ::st_ManagedBuffer_get_section_num_entities(
        orig_buffer.data(), OBJECTS_ID, slot_size );

    /* --------------------------------------------------------------------- */

    std::vector< cl::Platform > platform;
    cl::Platform::get( &platform );

    if( platform.empty() )
    {
        std::cout << "Unable to perform unit-test as no OpenCL "
                  << "platforms have been found."
                  << std::endl;

        return 0;
    }

    std::vector< cl::Device > devices;

    for( auto const& p : platform )
    {
        std::vector< cl::Device > temp_devices;

        p.getDevices( CL_DEVICE_TYPE_ALL, &temp_devices );

        for( auto const& d : temp_devices )
        {
            if( !d.getInfo< CL_DEVICE_AVAILABLE >() ) continue;
            devices.push_back( d );
        }
    }

    std::ostringstream a2str;

    a2str << "#if !defined( SIXTRL_NO_INCLUDES ) \r\n"
        << "    #include \"sixtracklib/_impl/definitions.h\"\r\n"
        << "    #include \"sixtracklib/common/impl/managed_buffer_minimal.h\"\r\n"
        << "    #include \"sixtracklib/common/impl/managed_buffer_remap.h\"\r\n"
        << "#endif /* !defined( SIXTRL_NO_INCLUDES ) */\r\n"
        << "\r\n"
        << "#include \"test_managed_buffer_opencl_kernel.cl\" \r\n"
        << "\r\n";

    std::string const PROGRAM_SOURCE_CODE = a2str.str();

    a2str.str( "" );
    a2str << " -D_GPUCODE=1"
          << " -D__NAMESPACE=st_"
          << " -I" << ::st_PATH_TO_BASE_DIR )
          << " -I" << ::st_PATH_TO_BASE_DIR << "tests/sixtracklib/opencl";

    std::string const COMPILE_OPTIONS = a2str.str();

    for( auto& device : devices )
    {
        std::fill( copy_buffer.begin(), copy_buffer.end(), raw_t{ 0 } );

        size_t current_copy_buffer_size = size_t{ 0 };

        success = ::st_ManagedBuffer_init( orig_buffer.data(),
            &current_copy_buf_size, max_num_objects, max_num_slots,
                max_num_dataptrs, max_num_garbage, orig_buffer.size(),
                    slot_size );

        ASSERT_TRUE( success == 0 );
        ASSERT_TRUE( current_orig_buffer_size == current_copy_buffer_size );

        cl_int cl_ret = ::CL_SUCCESS;

        cl::Context context( device );
        cl::CommandQueue queue( context, device, CL_QUEUE_PROFILING_ENABLE );
        cl::Program program( context, PROGRAM_SOURCE_CODE );

        try
        {
            cl_ret = program.build( COMPILE_OPTIONS.c_str() );
        }
        catch( cl::Error const& e )
        {
            std::cerr << "OpenCL Compilation Error -> Stopping Unit-Test \r\n"
                      << program.getBuildInfo< CL_PROGRAM_BUILD_LOG >( device )
                      << "\r\n"
                      << std::endl;

            cl_ret = CL_FALSE;
            throw;
        }

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        cl::Buffer cl_orig_buffer(
            context, CL_MEM_READ_WRITE, current_orig_buffer_size );

        cl::Buffer cl_copy_buffer(
            context, CL_MEM_WRITE_ONLY, current_copy_buffer_size );


    }



}

/* end: tests/sixtracklib/opencl/test_managed_buffer_opencl.cpp */
