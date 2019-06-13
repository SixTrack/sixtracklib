if( NOT  SIXTRACKL_CMAKE_SETUP_OPENCL_FINISHED )
    set( SIXTRACKL_CMAKE_SETUP_OPENCL_FINISHED 1 )

    message(STATUS "---- Processing cmake/SetupOpenCL.cmake")

    # --------------------------------------------------------------------------
    # Add OPENCL to the list of supported modules and track its state:

    list( APPEND SIXTRACKLIB_SUPPORTED_MODULES "OPENCL" )

    if( SIXTRACKL_ENABLE_OPENCL )
        list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES "1" )
    else()
        list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES "0" )
    endif()

    # --------------------------------------------------------------------------
    # Provide include directories and library directories for OpenCL, if enabled

    if( NOT  SIXTRACKL_OPENCL_INCLUDE_DIR )
        set( SIXTRACKL_OPENCL_INCLUDE_DIR   )
    endif()

    if( NOT  SIXTRACKL_OPENCL_LIBRARY )
        set( SIXTRACKL_OPENCL_LIBRARY )
    endif()

    if( NOT  SIXTRACKL_OPENCL_VERSION_STR )
        set( SIXTRACKL_OPENCL_VERSION_STR "" )
    endif()

    if( SIXTRACKL_ENABLE_OPENCL AND NOT OpenCL_FOUND )
        find_package( OpenCL REQUIRED )

        if( OpenCL_FOUND )

            set( SIXTRACKL_OPENCL_LIBRARY
               ${SIXTRACKL_OPENCL_LIBRARY} ${OpenCL_LIBRARY} )

            set( SIXTRACKL_OPENCL_VERSION_STR
               ${SIXTRACKL_OPENCL_VERSION_STR} ${OpenCL_VERSION_STRING} )

            set( SIXTRACKL_OPENCL_INCLUDE_DIR
               ${SIXTRACKL_OPENCL_INCLUDE_DIR} ${OpenCL_INCLUDE_DIR} )

            if( NOT SIXTRACKL_USE_LEGACY_CL_HPP )
                set( CXX_OPENCL_HEADER_NAME "cl2.hpp" )
            else()
                set( CXX_OPENCL_HEADER_NAME "cl.hpp" )
            endif()

            set( CXX_OPENCL_HEADER
                 "${OpenCL_INCLUDE_DIR}/CL/${CXX_OPENCL_HEADER_NAME}" )

            if( NOT EXISTS ${CXX_OPENCL_HEADER} )
                set( MSG "---- -- Unable to find OpenCL 1.x C++ header " )
                message( STATUS "${MSG} ${CXX_OPENCL_HEADER}" )

                set( EXT_OCLCXX_DIR "${CMAKE_BINARY_DIR}/ext_opencl_clhpp" )
                set( CONTRIBUTED_CXX_HEADER
                     "${EXT_OCLCXX_DIR}/CL/${CXX_OPENCL_HEADER_NAME}" )

                if( EXISTS ${CONTRIBUTED_CXX_HEADER} )
                    set( MSG "---- -- Reuse existing externally contributed " )
                    set( MSG "${MSG} header ${CONTRIBUTED_CXX_HEADER}" )
                    message( STATUS ${MSG} )

                    set( CXX_OPENCL_HEADER ${CONTRIBUTED_CXX_HEADER} )

                elseif( NOT SIXTRACK_REQUIRE_OFFLINE_BUILD )
                    if( NOT SIXTRACKL_USE_LEGACY_CL_HPP )
                        include( SetupOpenCLFetchOpenCL2CxxHeaders )
                    else()
                        include( SetupOpenCLFetchOpenCLCxxHeaders )
                    endif()

                    if( EXISTS ${CONTRIBUTED_CXX_HEADER} )
                        set( MSG "---- -- successfully provisioned header at" )
                        message( STATUS "${MSG} ${CONTRIBUTED_CXX_HEADER}" )

                        set( CXX_OPENCL_HEADER ${CONTRIBUTED_CXX_HEADER} )
                    endif()
                endif()
            endif()

            if( NOT EXISTS ${CXX_OPENCL_HEADER} )
                set( FAIL_MSG "---- -- Please install the OpenCL 1.x c++" )
                set( FAIL_MSG "${FAIL_MSG} wrapper header " )
                set( FAIL_MSG "${FAIL_MSG} ${CXX_OPENCL_HEADER}" )
                message( FATAL_ERROR ${FAIL_MSG} )
            endif()
        endif()
    endif()
endif()

#end: cmake/SetupOpenCL.cmake
