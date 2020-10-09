# =============================================================================
# end: sixtracklib/common/be_dipedge/CMakeLitst.txt

set( SIXTRL_COMMON_BE_DIPEDGE_CXX_HEADERS be_dipedge.hpp )
set( SIXTRL_COMMON_BE_DIPEDGE_C99_HEADERS be_dipedge.h track.h )
set( SIXTRL_COMMON_BE_DIPEDGE_C99_SOURCES be_dipedge.c )

add_library( sixtrack_common_be_dipedge OBJECT
    ${SIXTRL_COMMON_BE_DIPEDGE_C99_HEADERS}
    ${SIXTRL_COMMON_BE_DIPEDGE_CXX_HEADERS}
    ${SIXTRL_COMMON_BE_DIPEDGE_C99_SOURCES} )

target_include_directories( sixtrack_common_be_dipedge PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_be_dipedge PROPERTIES
    LINKER_LANGUAGE C POSITION_INDEPENDENT_CODE ON
    C_STANDARD 99 C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_be_dipedge PRIVATE
    ${SIXTRACKLIB_C99_FLAGS} ${SIXTRACKLIB_CPU_FLAGS}
    ${SIXTRACKL_C99_AUTOVEC_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_be_beambeam as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_be_dipedge> CACHE INTERNAL "" FORCE )

# ==============================================================================
# Installation:

set( SIXTRL_COMMON_BE_DIPEDGE_C99_INSTALL_PATH
     "${CMAKE_INSTALL_PREFIX}/include/sixtracklib/common/be_dipedge" )

if( SIXTRL_COMMON_BE_DIPEDGE_C99_HEADERS )
    install( FILES ${SIXTRL_COMMON_BE_DIPEDGE_C99_HEADERS}
             DESTINATION ${SIXTRL_COMMON_BE_DIPEDGE_C99_INSTALL_PATH} )
endif()

if( SIXTRACKL_ENABLE_CXX AND SIXTRL_COMMON_BE_DIPEDGE_CXX_HEADERS )
    set( SIXTRL_COMMON_BE_DIPEDGE_CXX_INSTALL_PATH
       ${SIXTRL_COMMON_BE_DIPEDGE_C99_INSTALL_PATH} )

    install( FILES ${SIXTRL_COMMON_BE_DIPEDGE_CXX_HEADERS}
             DESTINATION ${SIXTRL_COMMON_BE_DIPEDGE_CXX_INSTALL_PATH} )
endif()

#end: sixtracklib/common/be_dipedge/CMakeLists.txt
