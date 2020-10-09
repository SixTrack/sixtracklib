# ----------------------------------------------------------------------------
# sixtracklib/common/control/CMakeLists.txt

set( SIXTRL_COMMON_CONTROL_C99_SOURCES )
set( SIXTRL_COMMON_CONTROL_C99_HEADERS
     definitions.h debug_register.h arch_info.h arch_base.h node_id.h
     node_info.h kernel_config_base.h controller_base.h node_controller_base.h
     argument_base.h )

# add_library( sixtrack_common_control_c99 OBJECT
#     ${SIXTRL_COMMON_CONTROL_C99_HEADERS}
#     ${SIXTRL_COMMON_CONTROL_C99_SOURCES}
# )

# target_include_directories( sixtrack_common_control_c99 PRIVATE
#     $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
#     $<INSTALL_INTERFACE:include> )

# set_target_properties( sixtrack_common_control_c99
#     PROPERTIES LINKER_LANGUAGE C )
#
# set_target_properties( sixtrack_common_control_c99
#     PROPERTIES POSITION_INDEPENDENT_CODE ON )
#
# set_target_properties( sixtrack_common_control_c99
#     PROPERTIES C_STANDARD 99 )
#
# set_target_properties( sixtrack_common_control_c99
#     PROPERTIES C_STANDARD_REQUIRED ON )
#
# target_compile_options( sixtrack_common_control_c99 PRIVATE
#     -Wall -Werror -pedantic -ansi ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------

set( SIXTRL_COMMON_CONTROL_CXX_HEADERS arch_info.hpp arch_base.hpp node_id.hpp
    node_info.hpp kernel_config_base.hpp controller_base.hpp
    node_controller_base.hpp argument_base.hpp )

set( SIXTRL_COMMON_CONTROL_CXX_SOURCES arch_info.cpp arch_info_c99.cpp
     arch_base.cpp arch_base_c99.cpp node_id.cpp node_id_c99.cpp node_info.cpp
     node_info_c99.cpp kernel_config_base.cpp kernel_config_base_c99.cpp
     controller_base.cpp controller_base_c99.cpp node_controller_base.cpp
     node_controller_base_c99.cpp argument_base.cpp argument_base_c99.cpp )

add_library( sixtrack_common_control_cxx OBJECT
    ${SIXTRL_COMMON_CONTROL_C99_HEADERS}
    ${SIXTRL_COMMON_CONTROL_CXX_HEADERS}
    ${SIXTRL_COMMON_CONTROL_CXX_SOURCES} )

target_include_directories( sixtrack_common_control_cxx PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_control_cxx PROPERTIES CXX_STANDARD 11
    LINKER_LANGUAGE C POSITION_INDEPENDENT_CODE ON CXX_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_control_cxx
    PRIVATE ${SIXTRACKLIB_CXX_FLAGS} ${SIXTRACKLIB_CPU_FLAGS}
    ${SIXTRACKL_CXX_AUTOVEC_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_internal as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
#     $<TARGET_OBJECTS:sixtrack_common_control_c99>
    $<TARGET_OBJECTS:sixtrack_common_control_cxx> CACHE INTERNAL "" FORCE )

# -----------------------------------------------------------------------------
# install :

if( SIXTRL_COMMON_CONTROL_C99_HEADERS )
    install( FILES ${SIXTRL_COMMON_CONTROL_C99_HEADERS}
             DESTINATION "${SIXTRL_COMMON_C99_INSTALL_PREFIX}/control" )
endif()

if( SIXTACKL_ENABLE_CXX AND SIXTRL_COMMON_CONTROL_CXX_HEADERS )
    install( FILES ${SIXTRL_COMMON_CONTROL_CXX_HEADERS}
             DESTINATION "${SIXTRL_COMMON_C99_INSTALL_PREFIX}/control" )
endif()

#end: sixtracklib/common/buffer/CMakeLists.txt

