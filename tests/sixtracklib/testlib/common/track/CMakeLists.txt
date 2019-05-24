# ==============================================================================
# tests/sixtracklib/testlib/common/track/CMakeLists.txt

set( SIXTRL_TESTLIB_COMMON_TRACK_C99_HEADERS
     track_job_setup.h
     track_particles_cpu.h
     track_particles_ctrl_arg.h
)

set( SIXTRL_TESTLIB_COMMON_TRACK_C99_SOURCES
     track_job_setup.c
     track_particles_cpu.c
     track_particles_ctrl_arg.c
)

add_library( sixtrack_test_common_track_c99 OBJECT
             ${SIXTRL_TESTLIB_COMMON_TRACK_C99_HEADERS}
             ${SIXTRL_TESTLIB_COMMON_TRACK_C99_SOURCES}
)

target_include_directories( sixtrack_test_common_track_c99 PRIVATE
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests> )

set_target_properties( sixtrack_test_common_track_c99 PROPERTIES
    LINKER_LANGUAGE C  POSITION_INDEPENDENT_CODE ON
    C_STANDARD 99 C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_test_common_track_c99 PRIVATE
        -Wall -Werror -pedantic -ansi ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------

set( SIXTRL_TESTLIB_COMMON_TRACK_CXX_HEADERS
     track_job_setup.hpp
)

set( SIXTRL_TESTLIB_COMMON_TRACK_CXX_SOURCES
     track_job_setup.cpp
)

add_library( sixtrack_test_common_track_cxx OBJECT
    ${SIXTRL_TESTLIB_COMMON_TRACK_CXX_HEADERS}
    ${SIXTRL_TESTLIB_COMMON_TRACK_CXX_SOURCES} )

target_include_directories( sixtrack_test_common_track_cxx PRIVATE
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests> )

set_target_properties( sixtrack_test_common_track_cxx PROPERTIES
    LINKER_LANGUAGE C  POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 11 C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_test_common_track_cxx PRIVATE
        -Wall -Werror -pedantic -ansi ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_test_common_track_c99 and
# sixtrack_test_common_track_cxx as a module for sixtrack_test:

set(  SIXTRACKL_TESTLIB_LIBRARY_MODULES
    ${SIXTRACKL_TESTLIB_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_test_common_track_c99>
    $<TARGET_OBJECTS:sixtrack_test_common_track_cxx>
    CACHE INTERNAL "" FORCE )

# ==============================================================================
# Install :

if( SIXTRL_TESTLIB_COMMON_TRACK_C99_HEADERS )
    install( FILES ${SIXTRL_TESTLIB_COMMON_TRACK_C99_HEADERS}
             DESTINATION "${SIXTRL_TESTLIB_COMMON_INSTALL_C99_PATH}/track" )
endif()


if( ( SIXTRL_TESTLIB_COMMON_TRACK_CXX_HEADERS ) AND
    ( SIXTRACKL_ENABLE_CXX ) )
    install( FILES ${SIXTRL_TESTLIB_COMMON_TRACK_CXX_HEADERS}
             DESTINATION "${SIXTRL_TESTLIB_COMMON_INSTALL_CXX_PATH}/track" )

endif()

#end: tests/sixtracklib/testlib/common/track/CMakeLists.txt
