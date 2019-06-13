set( SIXTRL_COMMON_CONTEXT_C99_HEADERS
     compute_arch.h
     definitions.h
)

set( SIXTRL_COMMON_CONTEXT_C99_SOURCES
    compute_arch.c
)

add_library( sixtrack_common_context OBJECT
    ${SIXTRL_COMMON_CONTEXT_C99_HEADERS}
    ${SIXTRL_COMMON_CONTEXT_C99_SOURCES}
)

target_include_directories( sixtrack_common_context PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_context PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack_common_context PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties( sixtrack_common_context PROPERTIES C_STANDARD 99 )
set_target_properties( sixtrack_common_context PROPERTIES C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_context PRIVATE
    -Wall -Werror -pedantic -ansi ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_internal as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES
    ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_context>
    CACHE INTERNAL "" FORCE )

# -----------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_COMMON_CONTEXT_INSTALL_PATH
     include/sixtracklib/common/context )

if( SIXTRL_COMMON_CONTEXT_C99_HEADERS )
    install( FILES ${SIXTRL_COMMON_CONTEXT_C99_HEADERS}
             DESTINATION ${SIXTRACKLIB_COMMON_CONTEXT_INSTALL_PATH} )
endif()

#end: sixtracklib/common/buffer/CMakeLists.txt

