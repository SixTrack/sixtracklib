if(  NOT SETUP_C99_FINISHED )
    set( SETUP_C99_FINISHED 1 )

    message( STATUS "---- Processing cmake/SetupC99.cmake" )

    # --------------------------------------------------------------------------
    # Prepare default c99 compiler flags

    set( SIXTRACKLIB_C99_FLAGS )

    if( SIXTRACKL_DEFAULT_C99_FLAGS )
        string( REPLACE " " ";"
                SIXTRACKLIB_C99_FLAGS ${SIXTRACKL_DEFAULT_C99_FLAGS} )
    endif()
endif()
