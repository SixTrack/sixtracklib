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

            set( SIXTRACKL_OPENCL_INCLUDE_DIR
               ${SIXTRACKL_OPENCL_INCLUDE_DIR} ${OpenCL_INCLUDE_DIR} )

            set( SIXTRACKL_OPENCL_LIBRARY
               ${SIXTRACKL_OPENCL_LIBRARY} ${OpenCL_LIBRARY} )

            set( SIXTRACKL_OPENCL_VERSION_STR
               ${SIXTRACKL_OPENCL_VERSION_STR} ${OpenCL_VERSION_STRING} )

        endif()

    endif()

endif()

#end: cmake/SetupOpenCL.cmake
