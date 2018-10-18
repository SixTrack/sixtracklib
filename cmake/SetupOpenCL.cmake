if( NOT  SIXTRACKL_CMAKE_SETUP_OPENCL_FINISHED )
    set( SIXTRACKL_CMAKE_SETUP_OPENCL_FINISHED 1 )

    message(STATUS "---- Processing sixtracklib/cmake/SetupOpenCL.cmake")

    if( NOT SIXTRACKL_OPENCL_INCLUDE_DIR )
        set( SIXTRACKL_OPENCL_INCLUDE_DIR   )
    endif()

    if( NOT SIXTRACKL_OPENCL_LIBRARY )
        set( SIXTRACKL_OPENCL_LIBRARY )
    endif()

    if( NOT SIXTRACKL_OPENCL_VERSION_STR )
        set( SIXTRACKL_OPENCL_VERSION_STR "" )
    endif()

    if( NOT OpenCL_FOUND )
        find_package( OpenCL REQUIRED )

        if( OpenCL_FOUND )

            set( SIXTRACKL_OPENCL_INCLUDE_DIR ${SIXTRACKL_OPENCL_INCLUDE_DIR}
                 ${OpenCL_INCLUDE_DIR} )

            set( SIXTRACKL_OPENCL_LIBRARY     ${SIXTRACKL_OPENCL_LIBRARY}
                 ${OpenCL_LIBRARY} )

            set( SIXTRACKL_OPENCL_VERSION_STR ${SIXTRACKL_OPENCL_VERSION_STR}
                 ${OpenCL_VERSION_STRING} )

        endif()

    endif()

endif()

#end: sixtracklib/cmake/SetupOpenCL.cmake
