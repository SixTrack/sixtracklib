set( SIXTRL_COMMON_BUFFER_HEADERS
     alignment.h
     buffer_dev_debug.h
     buffer_garbage.h
     buffer_generic.h
     buffer_object.h
     buffer_type.h
     managed_buffer_minimal.h
     managed_buffer_remap.h
     managed_buffer.h
     mem_pool.h
)

set( SIXTRL_COMMON_BUFFER_SOURCES
     alignment.c
     buffer.c
     mem_pool.c
)

add_library( sixtrack_common_buffer OBJECT
    ${SIXTRL_COMMON_BUFFER_HEADERS}
    ${SIXTRL_COMMON_BUFFER_SOURCES}
)

target_include_directories( sixtrack_common_buffer PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_buffer PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack_common_buffer PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties( sixtrack_common_buffer PROPERTIES C_STANDARD 99 )
set_target_properties( sixtrack_common_buffer PROPERTIES C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_buffer PRIVATE
                        -Wall -Werror -pedantic -ansi
                        ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_buffer as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES
    ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_buffer> CACHE INTERNAL "" FORCE )

# -----------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_COMMON_BUFFER_C99_INSTALL_PATH
     include/sixtracklib/common/buffer )

install( FILES ${SIXTRL_COMMON_BUFFER_HEADERS}
         DESTINATION ${SIXTRACKLIB_COMMON_BUFFER_C99_INSTALL_PATH} )

#end: sixtracklib/common/buffer/CMakeLists.txt

