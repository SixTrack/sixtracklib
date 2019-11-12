if(  NOT SETUP_C99_FINISHED )
    set( SETUP_C99_FINISHED 1 )

    message( STATUS "---- Processing cmake/SetupC99.cmake" )

    # --------------------------------------------------------------------------
    # Prepare default c99 compiler flags

    if( SIXTRACKL_DEFAULT_C99_FLAGS )
        string( REPLACE " " ";" SIXTRL_C99_FLAGS
                ${SIXTRACKL_DEFAULT_C99_FLAGS} )

        if( SIXTRL_C99_FLAGS )
            set( SIXTRACKLIB_C99_FLAGS
               ${SIXTRACKLIB_C99_FLAGS} ${SIXTRL_C99_FLAGS} )
        endif()

    endif()

endif()
