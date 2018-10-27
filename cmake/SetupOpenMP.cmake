if( NOT  SIXTRACKL_CMAKE_SETUP_OPENMP_FINISHED )
    set( SIXTRACKL_CMAKE_SETUP_OPENMP_FINISHED 1 )

    message(STATUS "---- Processing cmake/SetupOpenMP.cmake")

     # --------------------------------------------------------------------------
    # Add OPENMP to the list of supported modules and track its state:

    list( APPEND SIXTRACKLIB_SUPPORTED_MODULES "OPENMP" )

    if( SIXTRACKL_ENABLE_OPENMP )
        list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES "1" )
    else()
        list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES "0" )
    endif()

    # --------------------------------------------------------------------------
    # Provide include directories and library directories for OpenMP, if enabled

    if( NOT  SIXTRACKL_OPENMP_C_FLAGS )
        set( SIXTRACKL_OPENMP_C_FLAGS   )
    endif()

    if( NOT  SIXTRACKL_OPENMP_CXX_FLAGS )
        set( SIXTRACKL_OPENMP_CXX_FLAGS )
    endif()

    if( NOT  SIXTRACKL_OPENMP_FORTRAN_FLAGS )
        set( SIXTRACKL_OPENMP_FORTRAN_FLAGS )
    endif()

    if( NOT  SIXTRACKL_OPENMP_EXE_LINKER_FLAGS )
        set( SIXTRACKL_OPENMP_EXE_LINKER_FLAGS )
    endif()

    if( SIXTRACKL_ENABLE_OPENMP AND NOT ( OpenMP_FOUND OR OPENMP_FOUND ) )
        find_package( OpenMP REQUIRED )

        message( STATUS "OpenMP_FOUND: ${OpenMP_FOUND}" )
        message( STATUS "OPENMP_FOUND: ${OPENMP_FOUND}" )

        if( OPENMP_FOUND AND NOT OpenMP_FOUND )
            set( OpenMP_FOUND ${OPENMP_FOUND} )
        endif()

        if( OpenMP_FOUND )

            if( OpenMP_C_FLAGS )
                set( SIXTRACKL_OPENMP_C_FLAGS ${SIXTRACKL_OPENMP_C_FLAGS}
                     ${OpenMP_C_FLAGS} )
            endif()

            if( OpenMP_CXX_FLAGS )
                set( SIXTRACKL_OPENMP_CXX_FLAGS ${SIXTRACKL_OPENMP_CXX_FLAGS}
                     ${OpenMP_CXX_FLAGS} )
            endif()

            if( OpenMP_Fortran_FLAGS )
                set( SIXTRACKL_OPENMP_FORTRAN_FLAGS ${SIXTRACKL_OPENMP_FORTRAN_FLAGS}
                     ${OpenMP_Fortran_FLAGS} )
            endif()

        endif()

    endif()

endif()
