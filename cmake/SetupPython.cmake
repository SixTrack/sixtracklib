if(  NOT SETUP_PYTHON_FINISHED )
    set( SETUP_PYTHON_FINISHED 1 )

    message( STATUS "---- Processing cmake/SetupPython.cmake" )

    # --------------------------------------------------------------------------
    # Add Python to the list of supported bindings / languages

    list( APPEND SIXTRACKLIB_SUPPORTED_BINDINGS "PYTHON" )

    if( SIXTRACKL_ENABLE_PYTHON )
        list( APPEND SIXTRACKLIB_SUPPORTED_BINDINGS_VALUES "1" )
    else()
        list( APPEND SIXTRACKLIB_SUPPORTED_BINDINGS_VALUES "0" )
    endif()

    # ==========================================================================

    if( NOT  Python_ADDITIONAL_VERSIONS )
        set( Python_ADDITIONAL_VERSIONS 3.8 3.7 3.6 3 )
    endif()

    if( SIXTRACKL_ENABLE_PYTHON AND NOT PYTHONINTERP_FOUND )
        find_package( PythonInterp REQUIRED )
    endif()

    if( SIXTRACKL_ENABLE_PYTHON AND NOT PYTHONLIBS_FOUND )
        find_package( PythonLibs REQUIRED )
    endif()

endif()
