# tests/sixtracklib/testlib/testdata/CMakeLists.txt:

set( TESTLIB_TESTDATA_INSTALL_PATH
     "${CMAKE_INSTALL_PREFIX}/include/sixtracklib/testlib/testdata" )

set( TESTLIB_TESTDATA_GENERATED_HEADERS )

# ------------------------------------------------------------------------------
# generate the testdata_files.h header to be used with examples and unit-tests

set( SIXTRACKLIB_TESTDATA_DIR "${CMAKE_SOURCE_DIR}/tests/testdata" )

set( TESTLIB_TESTDATA_FILES_H_IN_PATH
     "${CMAKE_CURRENT_SOURCE_DIR}/testdata_files.h.template" )

set( TESTLIB_TESTDATA_FILES_H_OUT_PATH
     "${CMAKE_CURRENT_SOURCE_DIR}/testdata_files.h" )

set( TESTLIB_TESTDATA_FILES_H_INSTALL_PATH
     "${TESTLIB_TESTDATA_INSTALL_PATH}/testdata_files.h" )

configure_file( ${TESTLIB_TESTDATA_FILES_H_IN_PATH}
                ${TESTLIB_TESTDATA_FILES_H_OUT_PATH} @ONLY )

set(  TESTLIB_TESTDATA_GENERATED_HEADERS
    ${TESTLIB_TESTDATA_GENERATED_HEADERS}
    "testdata/testdata_files.h" )

# ------------------------------------------------------------------------------
# Install :

install( CODE "
         set( SIXTRACKLIB_TESTDATA_DIR \"${CMAKE_INSTALL_PREFIX}/tests/testdata\" )
         message( STATUS \"Generating: ${TESTLIB_TESTDATA_FILES_H_INSTALL_PATH}\" )
         configure_file( \"${TESTLIB_TESTDATA_FILES_H_IN_PATH}\"
                         \"${TESTLIB_TESTDATA_FILES_H_INSTALL_PATH}\" @ONLY ) " )

# ------------------------------------------------------------------------------
# Pass generated headers to parent scope for target dependency management :

set(  SIXTRACKL_TESTLIB_C99_HEADERS
    ${SIXTRACKL_TESTLIB_C99_HEADERS}
    ${TESTLIB_TESTDATA_GENERATED_HEADERS} PARENT_SCOPE )

# ------------------------------------------------------------------------------
# Build local submodule library sixtrack_test_testdata

set( TESTLIB_TESTDATA_C99_HEADERS
     track_testdata.h
)

set( TESTLIB_TESTDATA_C99_SOURCES
     track_testdata.c
)

set( TESTLIB_TESTDATA_CXX_HEADERS )
set( TESTLIB_TESTDATA_CXX_SOURCES )

add_library( sixtrack_test_testdata OBJECT
            ${TESTLIB_TESTDATA_C99_HEADERS}
            ${TESTLIB_TESTDATA_C99_SOURCES}
            ${TESTLIB_TESTDATA_CXX_HEADERS}
            ${TESTLIB_TESTDATA_CXX_SOURCES}
)

target_include_directories( sixtrack_test_testdata PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/tests> )

set_target_properties( sixtrack_test_testdata PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack_test_testdata PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties( sixtrack_test_testdata PROPERTIES C_STANDARD 99 )
set_target_properties( sixtrack_test_testdata PROPERTIES C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_test_testdata PRIVATE
                        -Wall -Werror -pedantic -ansi
                        ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_test_common as a module for sixtrack_test:

set(  SIXTRACKL_TESTLIB_LIBRARY_MODULES
    ${SIXTRACKL_TESTLIB_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_test_testdata> CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------
# Install local submodule library sixtrack_test_testdata:

if( TESTLIB_TESTDATA_C99_HEADERS )
    set( TESTLIB_TESTDATA_C99_INSTALL_PATH include/sixtracklib/testlib/testdata )

    install( FILES ${TESTLIB_TESTDATA_C99_HEADERS}
             DESTINATION ${TESTLIB_TESTDATA_C99_INSTALL_PATH} )

endif()

if( TESTLIB_TESTDATA_CXX_HEADERS )
    set( TESTLIB_TESTDATA_CXX_INSTALL_PATH include/sixtracklib/testlib/common )

    install( FILES ${TESTLIB_TESTDATA_CXX_HEADERS}
             DESTINATION ${TESTLIB_TESTDATA_CXX_INSTALL_PATH} )
endif()

#end: tests/sixtracklib/testlib/testdata/CMakeLists.txt:
