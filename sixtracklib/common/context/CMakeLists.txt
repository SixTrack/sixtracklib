set( SIXTRL_COMMON_CONTEXT_C99_HEADERS
     compute_arch.h
     arch_info.h
     arch_base.h
     node_id.h
     node_info.h
     context_base.h
     argument_base.h
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

set( SIXTRL_COMMON_CONTEXT_CXX_HEADERS
     definitions.h
     arch_info.hpp
     arch_base.hpp
     node_id.hpp
     node_info.hpp
     context_base.hpp
     context_base_with_nodes.hpp
     argument_base.hpp
)

set( SIXTRL_COMMON_CONTEXT_CXX_SOURCES
     arch_info.cpp
     arch_info_c99.cpp
     arch_base.cpp
     arch_base_c99.cpp
     node_id.cpp
     node_id_c99.cpp
     node_info.cpp
     node_info_c99.cpp
     context_base.cpp
     context_base_c99.cpp
     context_base_with_nodes.cpp
     context_base_with_nodes_c99.cpp
     argument_base.cpp
     argument_base_c99.cpp
)

add_library( sixtrack_common_context_cxx OBJECT
    ${SIXTRL_COMMON_CONTEXT_CXX_HEADERS}
    ${SIXTRL_COMMON_CONTEXT_CXX_SOURCES}
)

target_include_directories( sixtrack_common_context_cxx PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_context_cxx
                       PROPERTIES LINKER_LANGUAGE C )

set_target_properties( sixtrack_common_context_cxx
                       PROPERTIES POSITION_INDEPENDENT_CODE ON )

set_target_properties( sixtrack_common_context_cxx
                       PROPERTIES CXX_STANDARD 11 )

set_target_properties( sixtrack_common_context_cxx
                       PROPERTIES CXX_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_context_cxx PRIVATE
                        -Wall -Werror -pedantic -ansi
                        ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_internal as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES
    ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_context>
    $<TARGET_OBJECTS:sixtrack_common_context_cxx>
    CACHE INTERNAL "" FORCE )

# -----------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_COMMON_CONTEXT_INSTALL_PATH
     include/sixtracklib/common/context )

if( SIXTRL_COMMON_CONTEXT_C99_HEADERS )
    install( FILES ${SIXTRL_COMMON_CONTEXT_C99_HEADERS}
             DESTINATION ${SIXTRACKLIB_COMMON_CONTEXT_INSTALL_PATH} )
endif()

if( SIXTRL_COMMON_CONTEXT_CXX_HEADERS )
    install( FILES ${SIXTRL_COMMON_CONTEXT_CXX_HEADERS}
             DESTINATION ${SIXTRACKLIB_COMMON_CONTEXT_INSTALL_PATH} )
endif()

#end: sixtracklib/common/buffer/CMakeLists.txt

