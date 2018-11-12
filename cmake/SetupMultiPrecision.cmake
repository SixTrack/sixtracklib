if( NOT  SIXTRACKL_CMAKE_SETUP_MULTI_PRECISION_FINISHED )
    set( SIXTRACKL_CMAKE_SETUP_MULTI_PRECISION_FINISHED 1 )

    message( STATUS "---- Processing cmake/SetupMultiPrecision.cmake" )

    # --------------------------------------------------------------------------
    # Add MPFR4 to the list of supported modules and track its state:

    list( APPEND SIXTRACKLIB_SUPPORTED_MODULES "MPFR4" )

    if( SIXTRACKL_ENABLE_MPFR4 )
        list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES "1" )
    else()
        list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES "0" )
    endif()

    # --------------------------------------------------------------------------
    # Provide include directories and library directories for MPFR, if enabled

    if( NOT  SIXTRACKL_MULTIPREC_INCLUDE_DIRS )
        set( SIXTRACKL_MULTIPREC_INCLUDE_DIRS )
    endif()

    if( NOT  SIXTRACKL_MULTIPREC_LIBRARIES )
        set( SIXTRACKL_MULTIPREC_LIBRARIES )
    endif()

    # --------------------------------------------------------------------------
    # Find MPFR and Gmp packages, if required

    if( SIXTRACKL_ENABLE_MPFR4 OR SIXTRACKL_ENABLE_MPFR_ACCURACY_TESTS )

        if( NOT Gmp_FOUND )
            find_package( Gmp REQUIRED )
        endif()

        if( NOT MPFR_FOUND )
            find_package( MPFR REQUIRED )
        endif()

        if( MPFR_FOUND AND Gmp_FOUND )

            if( Gmp_INCLUDES OR MPFR_INCLUDES )
                set(   SIXTRACKL_MULTIPREC_INCLUDE_DIRS
                     ${SIXTRACKL_MULTIPREC_INCLUDE_DIRS}
                     ${Gmp_INCLUDES} ${MPFR_INCLUDES} )
            endif()

            if( Gmp_LIBRARIES OR MPFR_LIBRARIES )
                set(  SIXTRACKL_MULTIPREC_LIBRARIES
                    ${SIXTRACKL_MULTIPREC_LIBRARIES}
                    m ${MPFR_LIBRARIES} ${Gmp_LIBRARIES} )
            endif()

        endif()
    endif()
endif()

# end: cmake/SetupMultiPrecision.cmake
