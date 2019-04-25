if( NOT SIXTRACKL_CMAKE_SETUP_OPENCL_FETCH_OPENCL_CXX_HEADERS_FINISHED )
   set( SIXTRACKL_CMAKE_SETUP_OPENCL_FETCH_OPENCL_CXX_HEADERS_FINISHED 1 )

    set( MSG "---- -- processing cmake/SetupOpenCLFetchOpenCLCxxHeaders.cmake")
    message( STATUS ${MSG} )
    message( STATUS "---- -- attempting to fetch OpenCL 1.x legacy C++ header")

    if( ( NOT CONTRIBUTED_CXX_HEADER ) OR ( NOT EXT_OCLCXX_DIR ) )
       set( MSG "---- -- Do not include this cmake file directly, " )
       set( MSG "${MSG} include SetupOpenCL.cmake instead!" )
       message( FATAL_ERROR ${MSG} )
    endif()

    set( KHRONOS_OPENCL_CLHPP_GIT_REPOSITORY
        "https://github.com/KhronosGroup/OpenCL-CLHPP.git" )

    if( ${CMAKE_VERSION} VERSION_GREATER 3.11.0 )
        include( FetchContent )

        set( EXT_OCLCXX_SRC_DIR "${EXT_OCLCXX_DIR}/src" )
        set( EXT_OCLCXX_OUT_DIR "${EXT_OCLCXX_DIR}/CL"  )

        FetchContent_Declare( OpenCL_CLHPP
            SOURCE_DIR  ${EXT_OCLCXX_SRC_DIR}
            BINARY_DIR  ${EXT_OCLCXX_OUT_DIR}
            GIT_REPOSITORY ${KHRONOS_OPENCL_CLHPP_GIT_REPOSITORY}
            GIT_TAG master )

        FetchContent_Populate( OpenCL_CLHPP )

        if( NOT PYTHONINTERP_FOUND )
            set( MSG "---- -- please enable python for auto header generation")
            message( SEND_ERROR ${MSG} )
        endif()

        set( GENERATOR_SCRIPT_PY "gen_cl_hpp.py" )

        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} ${GENERATOR_SCRIPT_PY}
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${EXT_OCLCXX_SRC_DIR} )

        if( result == 0 )
            set( CL_HPP_IN_FILE  "${EXT_OCLCXX_SRC_DIR}/cl.hpp" )
            set( CL_HPP_OUT_FILE "${EXT_OCLCXX_OUT_DIR}/cl.hpp" )

            if( EXISTS ${CL_HPP_IN_FILE} )
                configure_file( ${CL_HPP_IN_FILE} ${CL_HPP_OUT_FILE} COPYONLY )
            endif()
        else()
            set( MSG "---- -- execution of python generator script failed:" )
            message( SEND_ERROR "${MSG} ${result}" )
        endif()

    else()
       set( MSG "---- -- automatic downloading of cl.hpp requires the cmake" )
       set( MSG "${MSG} FetchContent module (and in turn, CMake >= 3.11.0)" )
       message( STATUS ${MSG} )
       message( STATUS "---- -- unable to continue automatically" )

       set( MSG "----- -- the header should be available from the official" )
       set( MSG "${MSG} KhronosGroup repository; hint: try " )
       message( STATUS "${MSG} ${KHRONOS_OPENCL_CLHPP_GIT_REPOSITORY}" )

       set( MSG "please install/download cl.hpp manually *or* (recommended)" )
       set( MSG "${MSG} install the non-legacy cl2.hpp header instead and" )
       set( MSG "${MSG} re-run cmake from the build directory" )
       message( SEND_ERROR ${MSG} )
    endif()

endif()
