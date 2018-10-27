# python/CMakeLists.txt

if( NOT PYTHONINTERP_FOUND )
    message( FATAL_ERROR "Python interpreter not found -> stopping" )
endif()


add_subdirectory( pysixtracklib )


set( SETUP_PY_IN_PATH      "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.template" )
set( SETUP_PY_OUT_PATH     "${CMAKE_CURRENT_SOURCE_DIR}/setup.py" )

configure_file( ${SETUP_PY_IN_PATH} ${SETUP_PY_OUT_PATH} @ONLY )

# end: python/CMakeLists.txt
