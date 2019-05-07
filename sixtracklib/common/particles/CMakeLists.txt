# ----------------------------------------------------------------------------
# sixtracklib/common/particles/CMakeLists.txt

set( SIXTRL_COMMON_PARTICLES_C99_HEADERS
     definitions.h
     particles_addr.h
)

set( SIXTRL_COMMON_PARTICLES_C99_SOURCES
     particles.c
)

add_library( sixtrack_common_particles_c99 OBJECT
    ${SIXTRL_COMMON_PARTICLES_C99_HEADERS}
    ${SIXTRL_COMMON_PARTICLES_C99_SOURCES}
)

target_include_directories( sixtrack_common_particles_c99 PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_particles_c99 PROPERTIES
    POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE C C_STANDARD 99
    C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_particles_c99 PRIVATE
    -Wall -Werror -pedantic -ansi ${SIXTRACKLIB_CPU_FLAGS} )

set(  SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_particles_c99>
    CACHE INTERNAL "" FORCE )

# -----------------------------------------------------------------------------

set( SIXTRL_COMMON_PARTICLES_CXX_HEADERS
     particles_addr.hpp
)

set( SIXTRL_COMMON_PARTICLES_CXX_SOURCES )

# ==============================================================================
# Handle install:

if( SIXTRL_COMMON_PARTICLES_C99_HEADERS )
    set( SIXTRACKLIB_COMMON_PARTICLES_C99_INSTALL_PATH
        "${CMAKE_INSTALL_PREFIX}/include/sixtracklib/common/particles" )

    install( FILES ${SIXTRL_COMMON_INTERNAL_C99_HEADERS}
             DESTINATION ${SIXTRACKLIB_COMMON_PARTICLES_C99_INSTALL_PATH} )
endif()

if( SIXTRL_COMMON_PARTICLES_CXX_HEADERS )
    set( SIXTRACKLIB_COMMON_PARTICLES_CXX_INSTALL_PATH
         "${CMAKE_INSTALL_PREFIX}/include/sixtracklib/common/particles" )

    install( FILES ${SIXTRL_COMMON_INTERNAL_CXX_HEADERS}
             DESTINATION ${SIXTRL_COMMON_PARTICLES_CXX_HEADERS} )
endif()

#end: sixtracklib/common/particles/CMakeLists.txt
