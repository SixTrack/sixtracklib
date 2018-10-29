# examples/python/CMakeLists.txt

message( STATUS "---- processing examples/python/CMakeLists.txt" )

set( PYTHON_EXAMPLES )
list( APPEND PYTHON_EXAMPLES __init__.py )

# -----------------------------------------------------------------------------
# track_bbsimple.py:

list( APPEND PYTHON_EXAMPLES track_bbsimple.py )

# -----------------------------------------------------------------------------
# track_beambeam.py:

list( APPEND PYTHON_EXAMPLES track_beambeam.py )

# -----------------------------------------------------------------------------
# track_lhc_no_bb.py:

list( APPEND PYTHON_EXAMPLES track_lhc_no_bb.py )


# =============================================================================
# Install:

if( SIXTRACKL_INSTALL_EXAMPLES )

    set( TESTDATA_PY_IN_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/testdata.py.template" )
    set( TESTDATA_PY_OUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/testdata.py" )
    set( TESTDATA_PY_INSTALL_PATH
        "${CMAKE_INSTALL_PREFIX}/examples/python/testdata.py" )

    configure_file( ${TESTDATA_PY_IN_PATH} ${TESTDATA_PY_OUT_PATH} @ONLY )

    install( CODE
            "set( SIXTRACKLIB_TESTDATA_DIR \"${CMAKE_INSTALL_PREFIX}/tests/testdata\" )
            configure_file( \"${TESTDATA_PY_IN_PATH}\"
                            \"${TESTDATA_PY_INSTALL_PATH}\" @ONLY )" )

    if( PYTHON_EXAMPLES )
        install( FILES ${PYTHON_EXAMPLES}
            DESTINATION "${CMAKE_INSTALL_PREFIX}/examples/python" )
    endif()

endif()

#end: tests/testdata/generators/CMakeLists.txt






