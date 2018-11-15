if(  NOT SETUP_CXX_FINISHED )
    set( SETUP_CXX_FINISHED 1 )

    message( STATUS "---- Processing cmake/SetupCxx.cmake" )

    # --------------------------------------------------------------------------
    # Add C++11 to the list of supported bindings / languages

    list( APPEND SIXTRACKLIB_SUPPORTED_BINDINGS "CXX" )

    if( SIXTRACKL_ENABLE_CXX )
        list( APPEND SIXTRACKLIB_SUPPORTED_BINDINGS_VALUES "1" )
    else()
        list( APPEND SIXTRACKLIB_SUPPORTED_BINDINGS_VALUES "0" )
    endif()



endif()
