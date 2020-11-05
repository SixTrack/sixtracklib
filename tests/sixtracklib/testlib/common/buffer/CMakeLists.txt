# tests/sixtracklib/testlib/common/buffer/CMakeLists.txt

# -----------------------------------------------------------------------------
# c99 object library

set( SIXTRL_TESTLIB_COMMON_BUFFER_HEADERS
     buffer.h generic_buffer_obj.h binary_array_buffer.h assign_address_item.h )

set( SIXTRL_TESTLIB_COMMON_BUFFER_SOURCES
     buffer.c generic_buffer_obj.c assign_address_item.c )

add_library( sixtrack_test_common_buffer_c99 OBJECT
             ${SIXTRL_TESTLIB_COMMON_BUFFER_HEADERS}
             ${SIXTRL_TESTLIB_COMMON_BUFFER_SOURCES} )

target_include_directories( sixtrack_test_common_buffer_c99 PRIVATE
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests> )

set_target_properties( sixtrack_test_common_buffer_c99 PROPERTIES
    LINKER_LANGUAGE C POSITION_INDEPENDENT_CODE ON
    C_STANDARD 99 C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_test_common_buffer_c99 PRIVATE
    ${SIXTRACKLIB_C99_FLAGS} ${SIXTRACKLIB_CPU_FLAGS}
    ${SIXTRACKL_C99_AUTOVEC_FLAGS} )

# -----------------------------------------------------------------------------
# cxx object library

set( SIXTRL_TESTLIB_COMMON_BUFFER_CXX_HEADERS )
set( SIXTRL_TESTLIB_COMMON_BUFFER_CXX_SOURCES )

# -----------------------------------------------------------------------------
# pass on sixtrack_test_common as a module for sixtrack_test:

set(  SIXTRACKL_TESTLIB_LIBRARY_MODULES ${SIXTRACKL_TESTLIB_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_test_common_buffer_c99> CACHE INTERNAL "" FORCE )

# =============================================================================
# Install :

if( SIXTRL_TESTLIB_COMMON_BUFFER_HEADERS )
    install( FILES ${TESTLIB_COMMON_C99_HEADERS}
             DESTINATION "${SIXTRL_TESTLIB_COMMON_INSTALL_C99_PATH}/buffer" )
endif()

if( ( SIXTRL_TESTLIB_COMMON_BUFFER_CXX_HEADERS ) AND ( SIXTRACKL_ENABLE_CXX ) )
    install( FILES ${SIXTRL_TESTLIB_COMMON_BUFFER_CXX_HEADERS}
             DESTINATION "${SIXTRL_TESTLIB_COMMON_INSTALL_CXX_PATH}/buffer" )
endif()

#end: tests/sixtracklib/testlib/common/buffer/CMakeLists.txt
