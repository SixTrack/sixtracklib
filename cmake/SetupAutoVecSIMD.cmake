if( NOT  SIXTRACKL_CMAKE_SETUP_AUTOVEC_SIMD_FINISHED )
    set( SIXTRACKL_CMAKE_SETUP_AUTOVEC_SIMD_FINISHED 1 )

    message( STATUS "---- Processing cmake/SetupAutoVecSIMD.cmake" )

    # --------------------------------------------------------------------------
    # Add AUTOVECTORIZATION and MANUAL_SIMD to the list of supported modules
    # and track its state:

    list( APPEND SIXTRACKLIB_SUPPORTED_MODULES "AUTOVECTORIZATION" )

    if( SIXTRACKL_ENABLE_AUTOVECTORIZATION )
        list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES "1" )
    else()
        list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES "0" )
    endif()


    list( APPEND SIXTRACKLIB_SUPPORTED_MODULES "MANUAL_SIMD" )

    if( SIXTRACKL_ENABLE_MANUAL_SIMD )
        list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES "1" )
    else()
        list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES "0" )
    endif()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # CPU/System architecture settings:

    if( SIXTRACKL_CPU_ARCH MATCHES "avx2" )
        message( STATUS "------ Optimizing for AVX2 architecture" )
        set( SIXTRACKLIB_CPU_FLAGS ${SIXTRACKLIB_CPU_FLAGS} -mavx2 )

    elseif( SIXTRACKL_CPU_ARCH MATCHES "avx" )
        message( STATUS "------ Optimizing for AVX architecture" )
        set( SIXTRACKLIB_CPU_FLAGS ${SIXTRACKLIB_CPU_FLAGS} -mavx )

    elseif( SIXTRACKL_CPU_ARCH MATCHES "sse2" )
        message( STATUS "------ Optimizing for SSE2 architecture" )
        set( SIXTRACKLIB_CPU_FLAGS ${SIXTRACKLIB_CPU_FLAGS} -msse2  )

    elseif( SIXTRACKL_CPU_ARCH MATCHES "native" )
        message( STATUS "------ Optimizing for native environment of the CPU" )
        set( SIXTRACKLIB_CPU_FLAGS ${SIXTRACKLIB_CPU_FLAGS} -march=native  )

    endif()

endif()

#end: cmake/SetupAutoVecSIMD.cmake
