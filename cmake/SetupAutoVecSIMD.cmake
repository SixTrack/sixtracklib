if( NOT  SIXTRACKL_CMAKE_SETUP_AUTOVEC_SIMD_FINISHED )
    set( SIXTRACKL_CMAKE_SETUP_AUTOVEC_SIMD_FINISHED 1 )

    message( STATUS "---- Processing cmake/SetupAutoVecSIMD.cmake" )

    # --------------------------------------------------------------------------
    # Add AUTOVECTORIZATION and MANUAL_SIMD to the list of supported modules
    # and track its state:

    list( APPEND SIXTRACKLIB_SUPPORTED_MODULES "AUTOVECTORIZATION" )
    set( SIXTRACKL_C_ENABLED_AUTOVEC_FLAGS )
    set( SIXTRACKL_C_DISABLED_AUTOVEC_FLAGS )

    if( CMAKE_C_COMPILER_ID STREQUAL "Clang" )
        set( SIXTRACKL_C_DISABLED_AUTOVEC_FLAGS -fno-slp-vectorize )
    elseif( CMAKE_C_COMPILER_ID STREQUAL "GNU" )
        set( SIXTRACKL_C_ENABLED_AUTOVEC_FLAGS
             -ftree-vectorize -ftree-vectorizer-verbose=6
             -fopt-info-loop
             -fno-fast-math
             --param vect-max-version-for-alias-checks=150 )
        set( SIXTRACKL_DEFAULT_C_NOAUTOVEC_FLAGS -fno-tree-vectorize )
    endif()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    set( SIXTRACKL_CXX_ENABLED_AUTOVEC_FLAGS )
    set( SIXTRACKL_CXX_DISABLED_AUTOVEC_FLAGS )

    if( SIXTRACKL_ENABLE_CXX )
        if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
            set( SIXTRACKL_CXX_DISABLED_AUTOVEC_FLAGS -fno-slp-vectorize )
        elseif( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
            set( SIXTRACKL_CXX_ENABLED_AUTOVEC_FLAGS
                 -ftree-vectorize -ftree-vectorizer-verbose=6
                 -fopt-info-loop
                 -fno-fast-math
                 --param vect-max-version-for-alias-checks=150 )
            set( SIXTRACKL_CXX_DISABLED_AUTOVEC_FLAGS -fno-tree-vectorize )
        endif()
    endif()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    set( SIXTRACKL_C99_AUTOVEC_FLAGS )
    set( SIXTRACKL_CXX_AUTOVEC_FLAGS )

    if( SIXTRACKL_ENABLE_AUTOVECTORIZATION )
        list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES "1" )
        set( SIXTRACKL_C99_AUTOVEC_FLAGS ${SIXTRACKL_C_ENABLED_AUTOVEC_FLAGS} )

        if( SIXTRACKL_ENABLE_CXX )
            set( SIXTRACKL_CXX_AUTOVEC_FLAGS
               ${SIXTRACKL_CXX_ENABLED_AUTOVEC_FLAGS} )
        endif()
    else()
        list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES "0" )
        set( SIXTRACKL_C99_AUTOVEC_FLAGS
           ${SIXTRACKL_C_DISABLED_AUTOVEC_FLAGS} )

        if( SIXTRACKL_ENABLE_C99 )
            set( SIXTRACKL_C99_AUTOVEC_FLAGS
               ${SIXTRACKL_C99_ENABLED_AUTOVEC_FLAGS} )
        endif()
    endif()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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
