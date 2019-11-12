# sixtracklib/common/output/CMakeLists.txt

set( SIXTRACKLIB_COMMON_OUTPUT_C99_HEADERS
     elem_by_elem_config.h elem_by_elem_output_buffer.h
     elem_by_elem_kernel_impl.h output_buffer.h
)

set( SIXTRACKLIB_COMMON_OUTPUT_C99_SOURCES
     elem_by_elem_config.c
     elem_by_elem_output_buffer.c
     output_buffer.c
)

set( SIXTRACKLIB_COMMON_OUTPUT_CXX_SOURCES )
set( SIXTRACKLIB_COMMON_OUTPUT_CXX_HEADERS
     output_buffer.hpp
)

# ------------------------------------------------------------------------------
# build sixtrack_common_output  submodule object library

add_library( sixtrack_common_output OBJECT
             ${SIXTRACKLIB_COMMON_OUTPUT_C99_HEADERS}
             ${SIXTRACKLIB_COMMON_OUTPUT_C99_SOURCES}
             ${SIXTRACKLIB_COMMON_OUTPUT_CXX_HEADERS} )

target_include_directories( sixtrack_common_output PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_output PROPERTIES LINKER_LANGUAGE C
    POSITION_INDEPENDENT_CODE ON C_STANDARD 99 C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_output PRIVATE
                        ${SIXTRACKLIB_C99_FLAGS} ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_output as a module for sixtracklib:

set( SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_output> CACHE INTERNAL "" FORCE )

# ==============================================================================
# install :

if( SIXTRACKLIB_COMMON_OUTPUT_C99_HEADERS )
    install( FILES ${SIXTRACKLIB_COMMON_OUTPUT_C99_HEADERS}
             DESTINATION "${SIXTRL_COMMON_C99_INSTALL_PREFIX}/output" )
endif()

if( SIXTRACKL_ENABLE_CXX AND SIXTRACKLIB_COMMON_OUTPUT_CXX_HEADERS )
    install( FILES ${SIXTRACKLIB_COMMON_OUTPUT_CXX_HEADERS}
             DESTINATION "${SIXTRL_COMMON_CXX_INSTALL_PREFIX}/output" )
endif()

# end: sixtracklib/common/output/CMakeLists.txt
