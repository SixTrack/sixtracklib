# python/CMakeLists.txt

if( NOT PYTHONINTERP_FOUND )
    message( FATAL_ERROR "Python interpreter not found -> stopping" )
endif()

add_subdirectory( pysixtracklib )

set( PYTHON_SIXTRACKLIB_SHARED_LIB  ${PY_SIXTRACKLIB_SHARED_LIB}  )
set( SETUP_PY_IN_PATH   "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.template" )
set( SETUP_PY_OUT_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/setup.py" )

configure_file( ${SETUP_PY_IN_PATH} ${SETUP_PY_OUT_PATH} @ONLY )

# ==============================================================================
# Install:

if( SIXTRACKL_INSTALL_PYSIXTRACKLIB )
    set( PYSIXTRACKLIB_INSTALL_STAMP
         "${CMAKE_BINARY_DIR}/python/pysixtracklib/pysixtracklib.stamp" )

    add_custom_command( OUTPUT ${PYSIXTRACKLIB_INSTALL_STAMP}
                        COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY_OUT_PATH} build
                        COMMAND ${CMAKE_COMMAND} -E touch ${PYSIXTRACKLIB_INSTALL_STAMP}
    )

    add_custom_target( pysixtracklib ALL DEPENDS
                       ${PYSIXTRACKLIB_INSTALL_STAMP} )

    install( CODE
            "execute_process( COMMAND ${PYTHON_EXECUTABLE}
                                      ${SETUP_PY_OUT_PATH} install )" )
endif()

# end: python/CMakeLists.txt
