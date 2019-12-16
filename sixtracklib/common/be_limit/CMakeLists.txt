# =============================================================================
# end: sixtracklib/common/be_limit/CMakeLitst.txt

set( SIXTRL_COMMON_BE_LIMIT_CXX_HEADERS
     be_limit_rect.hpp
     be_limit_ellipse.hpp
)

set( SIXTRL_COMMON_BE_LIMIT_C99_HEADERS
     be_limit_rect.h
     be_limit_ellipse.h
     be_limit_rect_ellipse.h
     track.h
)
set( SIXTRL_COMMON_BE_LIMIT_C99_SOURCES
     be_limit_rect.c
     be_limit_ellipse.c
     be_limit_rect_ellipse.c
)

add_library( sixtrack_common_be_limit OBJECT
    ${SIXTRL_COMMON_BE_LIMIT_C99_HEADERS}
    ${SIXTRL_COMMON_BE_LIMIT_C99_SOURCES}
    ${SIXTRL_COMMON_BE_LIMIT_CXX_HEADERS}
)

target_include_directories( sixtrack_common_be_limit PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_be_limit PROPERTIES
    LINKER_LANGUAGE C POSITION_INDEPENDENT_CODE ON
    C_STANDARD 99 C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_be_limit PRIVATE
    ${SIXTRACKLIB_C99_FLAGS} ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_be_beambeam as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_be_limit> CACHE INTERNAL "" FORCE )

# ==============================================================================
# Installation:

if( SIXTRL_COMMON_BE_LIMIT_C99_HEADERS )
    install( FILES ${SIXTRL_COMMON_BE_LIMIT_C99_HEADERS}
             DESTINATION "${SIXTRACKLIB_C99_INSTALL_PREFIX}/be_limit" )
endif()

if( SIXTRACKL_ENABLE_CXX AND SIXTRL_COMMON_BE_LIMIT_CXX_HEADERS )
    install( FILES ${SIXTRL_COMMON_BE_LIMIT_CXX_HEADERS}
             DESTINATION "${SIXTRACKLIB_CXX_INSTALL_PREFIX}/be_limit" )
endif()

#end: sixtracklib/common/be_limit/CMakeLists.txt
