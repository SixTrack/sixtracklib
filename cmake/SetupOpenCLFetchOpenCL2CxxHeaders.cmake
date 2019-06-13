if( NOT SIXTRACKL_CMAKE_SETUP_OPENCL_FETCH_OPENCL2_CXX_HEADERS_FINISHED )
   set( SIXTRACKL_CMAKE_SETUP_OPENCL_FETCH_OPENCL2_CXX_HEADERS_FINISHED 1 )

    set( MSG "---- -- processing cmake/SetupOpenCLFetchOpenCL2CxxHeaders.cmake" )
    message( STATUS ${MSG} )
    message( STATUS "---- -- attempting to fetch OpenCL 2.x C++ headers" )

    if( ( NOT CONTRIBUTED_CXX_HEADER ) OR ( NOT EXT_OCLCXX_DIR ) )
       set( MSG "Do not include this cmake file directly, " )
       set( MSG "${MSG} include SetupOpenCL.cmake instead!" )
       message( FATAL_ERROR ${MSG} )
    endif()

    set( KHRONOS_OPENCL_CLHPP_GIT_REPOSITORY
        "https://github.com/KhronosGroup/OpenCL-CLHPP.git" )

    if( ${CMAKE_VERSION} VERSION_GREATER 3.11.0 )
        include( FetchContent )

        FetchContent_Declare( OpenCL_CLHPP
            SOURCE_DIR  "${EXT_OCLCXX_DIR}/src"
            BINARY_DIR  "${EXT_OCLCXX_DIR}/CL"
            GIT_REPOSITORY ${KHRONOS_OPENCL_CLHPP_GIT_REPOSITORY}
            GIT_TAG master )

        FetchContent_Populate( OpenCL_CLHPP )

        set( CL2_HPP_IN_FILE  "${EXT_OCLCXX_DIR}/src/input_cl2.hpp" )
        set( CL2_HPP_OUT_FILE "${EXT_OCLCXX_DIR}/CL/cl2.hpp" )

        if( EXISTS ${CL2_HPP_IN_FILE} )
            configure_file( ${CL2_HPP_IN_FILE} ${CL2_HPP_OUT_FILE} COPYONLY )
        endif()

    else()
       set( MSG "---- -- automatic downloading of cl2.hpp requires the cmake" )
       set( MSG "${MSG} FetchContent module (and in turn, CMake >= 3.11.0)" )
       message( STATUS ${MSG} )
       message( STATUS "---- -- unable to continue automatically" )

       set( MSG "----- -- the header should be available from the official " )
       set( MSG "${MSG} KhronosGroup repository; hint: try " )
       message( STATUS "${MSG} ${KHRONOS_OPENCL_CLHPP_GIT_REPOSITORY}" )

       set( MSG "please install/download cl2.hpp manually and re-run" )
       set( MSG "${MSG} cmake from the build directory" )
       message( SEND_ERROR ${MSG} )
    endif()

endif()
