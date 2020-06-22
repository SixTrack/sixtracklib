# ==============================================================================
# tests/sixtracklib/testlib/common/particles/CMakeLists.txt

set( SIXTRL_TESTLIB_COMMON_PARTICLES_C99_HEADERS
     particles.h
     particles_addr.h )

set( SIXTRL_TESTLIB_COMMON_PARTICLES_C99_SOURCES
     particles.c )

add_library( sixtrack_test_common_particle_c99 OBJECT
             ${SIXTRL_TESTLIB_COMMON_PARTICLES_C99_HEADERS}
             ${SIXTRL_TESTLIB_COMMON_PARTICLES_C99_SOURCES}
)

target_include_directories( sixtrack_test_common_particle_c99 PRIVATE
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests> )

set_target_properties( sixtrack_test_common_particle_c99 PROPERTIES
    LINKER_LANGUAGE C  POSITION_INDEPENDENT_CODE ON
    C_STANDARD 99 C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_test_common_particle_c99 PRIVATE
    ${SIXTRACKLIB_C99_FLAGS} ${SIXTRACKLIB_CPU_FLAGS}
    ${SIXTRACKL_C99_AUTOVEC_FLAGS} )

# ------------------------------------------------------------------------------

set( SIXTRL_TESTLIB_COMMON_PARTICLES_CXX_HEADERS
     particles_addr.hpp
     particles_addr.h )

set( SIXTRL_TESTLIB_COMMON_PARTICLES_CXX_SOURCES
     particles_addr.cpp )

add_library( sixtrack_test_common_particle_cxx OBJECT
    ${SIXTRL_TESTLIB_COMMON_PARTICLES_CXX_HEADERS}
    ${SIXTRL_TESTLIB_COMMON_PARTICLES_CXX_SOURCES} )

target_include_directories( sixtrack_test_common_particle_cxx PRIVATE
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests> )

set_target_properties( sixtrack_test_common_particle_cxx PROPERTIES
    LINKER_LANGUAGE C  POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 11 C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_test_common_particle_cxx PRIVATE
    ${SIXTRACKLIB_CXX_FLAGS} ${SIXTRACKLIB_CPU_FLAGS}
    ${SIXTRACKL_C99_AUTOVEC_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_test_common_particle_c99 and
# sixtrack_test_common_particle_cxx as a module for sixtrack_test:

set(  SIXTRACKL_TESTLIB_LIBRARY_MODULES
    ${SIXTRACKL_TESTLIB_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_test_common_particle_c99>
    $<TARGET_OBJECTS:sixtrack_test_common_particle_cxx>
    CACHE INTERNAL "" FORCE )

# ==============================================================================
# Install :

if( SIXTRL_TESTLIB_COMMON_PARTICLES_C99_HEADERS )
    install( FILES ${SIXTRL_TESTLIB_COMMON_PARTICLES_C99_HEADERS}
             DESTINATION "${SIXTRL_TESTLIB_COMMON_INSTALL_C99_PATH}/particles" )
endif()


if( ( SIXTRL_TESTLIB_COMMON_PARTICLES_CXX_HEADERS ) AND
    ( SIXTRACKL_ENABLE_CXX ) )
    install( FILES ${SIXTRL_TESTLIB_COMMON_PARTICLES_CXX_HEADERS}
             DESTINATION "${SIXTRL_TESTLIB_COMMON_INSTALL_CXX_PATH}/particles" )

endif()

#end: tests/sixtracklib/testlib/common/particles/CMakeLists.txt
