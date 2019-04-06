# python/pysixtracklib_test/CMakeLists.txt

# ------------------------------------------------------------------------------
# Generate config.py

set( PYTHON_SIXTRACKLIB_TEST_SHARED_LIB ${PY_SIXTRACKLIB_TEST_SHARED_LIB} )

string( REPLACE ";" "\r\n" PYTHON_SIXTRACKLIB_TEST_MODULES_STR
        "${PY_SIXTRACKLIB_MODULES_STR}" )

if( "${PY_PATH_TO_TESTDATA}" STREQUAL "" )
    set( PY_PATH_TO_TESTDATA "${CMAKE_SOURCE_DIR}/tests/testdata" )
endif()

if( EXISTS ${PY_PATH_TO_TESTDATA} )
    set( PYTHON_HAS_TESTDATA_DIR "" )
else()
    set( PYTHON_HAS_TESTDATA_DIR "#" )
endif()

set( SIXTRL_PY_IN_FILE  config.py.template )
set( SIXTRL_PY_OUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/config.py" )

configure_file( ${SIXTRL_PY_IN_FILE} ${SIXTRL_PY_OUT_FILE} @ONLY )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Install

set( SIXTRL_PY_INSTALL_PATH
     "${CMAKE_INSTALL_PREFIX}/python/pysixtracklib_test/config.py" )

install( CODE
    "set( PYTHON_SIXTRACKLIB_TEST_SHARED_LIB ${PYTHON_SIXTRACKLIB_TEST_SHARED_LIB} )
     set( PYTHON_SIXTRACKLIB_TEST_MODULES_STR ${PYTHON_SIXTRACKLIB_TEST_MODULES_STR} )
     set( PYTHON_HAS_TESTDATA_DIR ${PYTHON_HAS_TESTDATA_DIR} )

     message( STATUS \"Generating: ${SIXTRL_PY_INSTALL_PATH}\" )
     configure_file( \"${SIXTRL_PY_IN_FILE}\" \"${SIXTRL_PY_INSTALL_PATH}\"
     @ONLY ) "
)

# end: python/pysixtracklib_test/CMakeLists.txt
