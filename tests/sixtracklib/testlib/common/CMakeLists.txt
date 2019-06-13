# tests/sixtracklib/testlib/common/CMakeLists.txt

add_subdirectory( beam_elements )

set( TESTLIB_COMMON_C99_HEADERS
     buffer.h
     generic_buffer_obj.h
     gpu_kernel.h
     particles.h
     random.h
     time.h
)

set( TESTLIB_COMMON_C99_SOURCES
     buffer.c
     generic_buffer_obj.c
     gpu_kernel.c
     particles.c
     random.c
     time.c
)

set( TESTLIB_COMMON_CXX_HEADERS )
set( TESTLIB_COMMON_CXX_SOURCES )


add_library( sixtrack_test_common OBJECT
            ${TESTLIB_COMMON_C99_HEADERS}
            ${TESTLIB_COMMON_C99_SOURCES}
            ${TESTLIB_COMMON_CXX_HEADERS}
            ${TESTLIB_COMMON_CXX_SOURCES}
)

target_include_directories( sixtrack_test_common PRIVATE
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests> )

set_target_properties( sixtrack_test_common PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack_test_common PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties( sixtrack_test_common PROPERTIES C_STANDARD 99 )
set_target_properties( sixtrack_test_common PROPERTIES C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_test_common PRIVATE
        -Wall -Werror -pedantic -ansi ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_test_common as a module for sixtrack_test:

set(  SIXTRACKL_TESTLIB_LIBRARY_MODULES
    ${SIXTRACKL_TESTLIB_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_test_common> CACHE INTERNAL "" FORCE )


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
    set( TESTLIB_COMMON_C99_INSTALL_PATH include/sixtracklib/testlib/common )

    install( FILES ${TESTLIB_COMMON_C99_HEADERS}
             DESTINATION ${TESTLIB_COMMON_C99_INSTALL_PATH} )

endif()


if( TESTLIB_COMMON_CXX_HEADERS )
    set( TESTLIB_COMMON_CXX_INSTALL_PATH include/sixtracklib/testlib/common )

    install( FILES ${TESTLIB_COMMON_CXX_HEADERS}
             DESTINATION ${TESTLIB_COMMON_CXX_INSTALL_PATH} )

endif()

#end: tests/sixtracklib/testlib/common/CMakeLists.txt
