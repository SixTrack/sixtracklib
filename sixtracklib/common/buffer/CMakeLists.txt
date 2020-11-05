set( SIXTRL_COMMON_BUFFER_CXX_HEADERS
     assign_address_item.hpp )

# -----------------------------------------------------------------------------

set( SIXTRL_COMMON_BUFFER_HEADERS
     alignment.h
     assign_address_item.h
     buffer_dev_debug.h
     buffer_garbage.h
     buffer_generic.h
     buffer_object.h
     buffer_type.h
     managed_buffer_minimal.h
     managed_buffer_remap.h
     managed_buffer.h
     mem_pool.h
     managed_buffer_handle.h
     buffer_string_object.h
     buffer_array_object.h
)

set( SIXTRL_COMMON_BUFFER_SOURCES
     alignment.c
     assign_address_item.c
     buffer.c
     mem_pool.c
     managed_buffer_handle.c
     buffer_string_object.c
     buffer_array_object.c
)

add_library( sixtrack_common_buffer OBJECT
    ${SIXTRL_COMMON_BUFFER_HEADERS} ${SIXTRL_COMMON_BUFFER_SOURCES} )

target_include_directories( sixtrack_common_buffer PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_buffer PROPERTIES LINKER_LANGUAGE C
    POSITION_INDEPENDENT_CODE ON C_STANDARD 99 C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_buffer PRIVATE
    ${SIXTRACKLIB_C99_FLAGS} ${SIXTRACKLIB_CPU_FLAGS}
    ${SIXTRACKL_C99_AUTOVEC_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_buffer as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_buffer> CACHE INTERNAL "" FORCE )

# -----------------------------------------------------------------------------
# install :

if( SIXTRL_COMMON_BUFFER_HEADERS )
    install( FILES ${SIXTRL_COMMON_BUFFER_HEADERS}
             DESTINATION "${SIXTRL_COMMON_C99_INSTALL_PREFIX}/buffer" )
endif()

if( SIXTRACKL_ENABLE_CXX AND SIXTRL_COMMON_BUFFER_CXX_HEADERS )
    install( FILES ${SIXTRL_COMMON_BUFFER_CXX_HEADERS}
             DESTINATION "${SIXTRL_COMMON_CXX_INSTALL_PREFIX}/buffer" )
endif()

#end: sixtracklib/common/buffer/CMakeLists.txt

