# ==============================================================================
# tests/sixtracklib/testlib/common/output/CMakeLists.txt

set( SIXTRL_TESTLIB_COMMON_OUTPUT_C99_HEADERS
     assign_be_monitor_ctrl_arg.h
     assign_elem_by_elem_ctrl_arg.h
)

set( SIXTRL_TESTLIB_COMMON_OUTPUT_C99_SOURCES
     assign_be_monitor_ctrl_arg.c
     assign_elem_by_elem_ctrl_arg.c
)

add_library( sixoutput_test_common_output_c99 OBJECT
             ${SIXTRL_TESTLIB_COMMON_OUTPUT_C99_HEADERS}
             ${SIXTRL_TESTLIB_COMMON_OUTPUT_C99_SOURCES}
)

target_include_directories( sixoutput_test_common_output_c99 PRIVATE
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests> )

set_target_properties( sixoutput_test_common_output_c99 PROPERTIES
    LINKER_LANGUAGE C  POSITION_INDEPENDENT_CODE ON
    C_STANDARD 99 C_STANDARD_REQUIRED ON )

target_compile_options( sixoutput_test_common_output_c99 PRIVATE
        -Wall -Werror -pedantic -ansi ${SIXOUTPUTLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixoutput_test_common_output_c99 and
# sixoutput_test_common_output_cxx as a module for sixoutput_test:

set(  SIXOUTPUTL_TESTLIB_LIBRARY_MODULES
    ${SIXOUTPUTL_TESTLIB_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixoutput_test_common_output_c99>
    CACHE INTERNAL "" FORCE )

# ==============================================================================
# Install :

if( SIXTRL_TESTLIB_COMMON_OUTPUT_C99_HEADERS )
    install( FILES ${SIXTRL_TESTLIB_COMMON_OUTPUT_C99_HEADERS}
             DESTINATION "${SIXTRL_TESTLIB_COMMON_INSTALL_C99_PATH}/output" )
endif()

#end: tests/sixtracklib/testlib/common/output/CMakeLists.txt
