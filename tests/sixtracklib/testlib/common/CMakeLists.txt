# ==============================================================================
# tests/sixtracklib/testlib/common/particles/CMakeLists.txt

message( STATUS "---- processing tests/sixtracklib/testlib/common/CMakeLists.txt" )

set( SIXTRL_TESTLIB_COMMON_INSTALL_C99_PATH
     "${SIXTRL_TESTLIB_INSTALL_C99_PATH}/common" )

set( SIXTRL_TESTLIB_COMMON_INSTALL_CXX_PATH
     "${SIXTRL_TESTLIB_INSTALL_CXX_PATH}/common" )

# ------------------------------------------------------------------------------

add_subdirectory( beam_elements )
add_subdirectory( particles )
add_subdirectory( track )
add_subdirectory( output )

# ------------------------------------------------------------------------------

set( TESTLIB_COMMON_C99_HEADERS
     gpu_kernel.h
     random.h
     time.h
)

set( TESTLIB_COMMON_C99_SOURCES
     gpu_kernel.c
     random.c
     time.c
)

add_library( sixtrack_test_common_c99 OBJECT
            ${TESTLIB_COMMON_C99_HEADERS}
            ${TESTLIB_COMMON_C99_SOURCES}
)

target_include_directories( sixtrack_test_common_c99 PRIVATE
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests> )

set_target_properties( sixtrack_test_common_c99 PROPERTIES
    LINKER_LANGUAGE C POSITION_INDEPENDENT_CODE ON
    C_STANDARD 99 C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_test_common_c99 PRIVATE
        ${SIXTRACKLIB_C99_FLAGS} ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_test_common as a module for sixtrack_test:

set(  SIXTRACKL_TESTLIB_LIBRARY_MODULES
    ${SIXTRACKL_TESTLIB_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_test_common_c99> CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------
# path.h :

set( PATH_IN_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/path.h.template" )
set( PATH_OUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/path.h" )

set( SIXTRACK_TESTLIB_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/tests/" )

configure_file( ${PATH_IN_PATH} ${PATH_OUT_PATH} @ONLY )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# install :

set( PATH_INSTALL_PATH
     "${CMAKE_INSTALL_PREFIX}/include/sixtracklib/testlib/common/path.h" )

install( CODE
    "set( SIXTRACK_TESTLIB_INCLUDE_DIR \"${CMAKE_INSTALL_PREFIX}/include/\" )
     message( STATUS \"Generating: ${PATH_INSTALL_PATH}\" )
     configure_file( \"${PATH_IN_PATH}\" \"${PATH_INSTALL_PATH}\" @ONLY )" )

# ==============================================================================
# Install :

if( TESTLIB_COMMON_C99_HEADERS )
    install( FILES ${TESTLIB_COMMON_C99_HEADERS}
             DESTINATION ${SIXTRL_TESTLIB_COMMON_INSTALL_C99_PATH} )
endif()


if( ( TESTLIB_COMMON_CXX_HEADERS ) AND ( SIXTRACKL_ENABLE_CXX ) )
    install( FILES ${TESTLIB_COMMON_CXX_HEADERS}
             DESTINATION ${SIXTRL_TESTLIB_COMMON_INSTALL_CXX_PATH} )
endif()

#end: tests/sixtracklib/testlib/common/CMakeLists.txt
