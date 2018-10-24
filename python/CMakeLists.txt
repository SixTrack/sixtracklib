# python/CMakeLists.txt

set( SIXTRACKLIB_TESTDATA_DIR "${CMAKE_SOURCE_DIR}/tests/testdata" )

set( TESTDATA_PY_IN_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/testdata.py.template" )
set( TESTDATA_PY_OUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/testdata.py" )

configure_file( ${TESTDATA_PY_IN_PATH} ${TESTDATA_PY_OUT_PATH} @ONLY )

# end: python/CMakeLists.txt
