if( NOT  SIXTRACKL_CMAKE_SETUP_CUDA_FINISHED   )
    set( SIXTRACKL_CMAKE_SETUP_CUDA_FINISHED 1 )

    message( STATUS "---- Processing cmake/SetupCuda.cmake" )

    # --------------------------------------------------------------------------
    # Add CUDA to the list of supported modules and track its state:

    list( APPEND SIXTRACKLIB_SUPPORTED_MODULES "CUDA" )

    if( SIXTRACKL_ENABLE_CUDA )
        list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES "1" )
    else()
        list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES "0" )
    endif()

    # --------------------------------------------------------------------------
    # Provide include directories and library directories for Cuda, if enabled

    if( NOT  SIXTRACKL_CUDA_INCLUDE_DIRS )
        set( SIXTRACKL_CUDA_INCLUDE_DIRS   )
    endif()

    if( NOT  SIXTRACKL_CUDA_LIBRARIES )
        set( SIXTRACKL_CUDA_LIBRARIES )
    endif()

    if( NOT  SIXTRACKL_CUDA_VERSION_STR )
        set( SIXTRACKL_CUDA_VERSION_STR "" )
    endif()

    get_property( SIXTRACKL_ENABLED_LANGS GLOBAL PROPERTY ENABLED_LANGUAGES )

    if( SIXTRACKL_ENABLE_CUDA AND
        NOT ( SIXTRACKL_ENABLED_LANGS MATCHES "CUDA" ) AND
        NOT ( CUDA_FOUND ) )

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
