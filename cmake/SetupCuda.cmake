if( NOT  SIXTRACKL_CMAKE_SETUP_CUDA_FINISHED   )
    set( SIXTRACKL_CMAKE_SETUP_CUDA_FINISHED 1 )

    message(STATUS "---- Processing sixtracklib/cmake/SetupCuda.cmake")

    if( NOT  SIXTRACKL_CUDA_INCLUDE_DIRS )
        set( SIXTRACKL_CUDA_INCLUDE_DIRS   )
    endif()

    if( NOT  SIXTRACKL_CUDA_LIBRARIES )
        set( SIXTRACKL_CUDA_LIBRARIES )
    endif()

    if( NOT  SIXTRACKL_CUDA_VERSION_STR )
        set( SIXTRACKL_CUDA_VERSION_STR "" )
    endif()

    if( NOT CUDA_FOUND )
        find_package( CUDA REQUIRED )

        if( CUDA_FOUND )

            set( SIXTRACKL_CUDA_INCLUDE_DIRS ${SIXTRACKL_CUDA_INCLUDE_DIRS}
                 ${CUDA_INCLUDE_DIRS} )

            set( SIXTRACKL_CUDA_LIBRARIES ${SIXTRACKL_CUDA_LIBRARIES}
                 ${CUDA_LIBRARIES} )

            set( SIXTRACKL_CUDA_VERSION_STR ${SIXTRACKL_CUDA_VERSION_STR}
                 ${CUDA_VERSION_STRING} )

        endif()

    endif()

endif()

#end: sixtracklib/cmake/SetupOpenCL.cmake
