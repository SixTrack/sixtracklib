# =============================================================================
# end: sixtracklib/common/be_limit/CMakeLitst.txt

set( SIXTRL_COMMON_BE_LIMIT_CXX_HEADERS )

set( SIXTRL_COMMON_BE_LIMIT_C99_HEADERS )
set( SIXTRL_COMMON_BE_LIMIT_C99_SOURCES )

add_library( sixtrack_common_be_limit OBJECT
    ${SIXTRL_COMMON_BE_LIMIT_C99_HEADERS}
    ${SIXTRL_COMMON_BE_LIMIT_CXX_HEADERS}
    ${SIXTRL_COMMON_BE_LIMIT_C99_SOURCES}
)

target_include_directories( sixtrack_common_be_limit PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_be_limit PROPERTIES
    LINKER_LANGUAGE C POSITION_INDEPENDENT_CODE ON
    C_STANDARD 99 C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_be_limit PRIVATE
    -Wall -Werror -pedantic -ansi ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_be_beambeam as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_be_limit> CACHE INTERNAL "" FORCE )

# ==============================================================================
# Installation:

set( SIXTRL_COMMON_BE_LIMIT_C99_INSTALL_PATH
     "${CMAKE_INSTALL_PREFIX}/include/sixtracklib/common/be_limit" )

if( SIXTRL_COMMON_BE_LIMIT_C99_HEADERS )
    install( FILES ${SIXTRL_COMMON_BE_LIMIT_C99_HEADERS}
             DESTINATION ${SIXTRL_COMMON_BE_LIMIT_C99_INSTALL_PATH} )
endif()

if( SIXTRACKL_ENABLE_CXX AND SIXTRL_COMMON_BE_LIMIT_CXX_HEADERS )
    set( SIXTRL_COMMON_BE_LIMIT_CXX_INSTALL_PATH
       ${SIXTRL_COMMON_BE_LIMIT_C99_INSTALL_PATH} )

    install( FILES ${SIXTRL_COMMON_BE_LIMIT_CXX_HEADERS}
             DESTINATION ${SIXTRL_COMMON_BE_LIMIT_CXX_INSTALL_PATH} )
endif()

#end:
