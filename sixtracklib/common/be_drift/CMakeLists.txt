set( SIXTRL_COMMON_BE_DRIFT_C99_HEADERS be_drift.h track.h )
set( SIXTRL_COMMON_BE_DRIFT_C99_SOURCES be_drift.c )
set( SIXTRL_COMMON_BE_DRIFT_CXX_HEADERS be_drift.hpp )

add_library( sixtrack_common_be_drift OBJECT
    ${SIXTRL_COMMON_BE_DRIFT_C99_HEADERS}
    ${SIXTRL_COMMON_BE_DRIFT_C99_SOURCES} )

target_include_directories( sixtrack_common_be_drift PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_be_drift PROPERTIES
    LINKER_LANGUAGE C POSITION_INDEPENDENT_CODE ON C_STANDARD 99
    C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_be_drift PRIVATE
        ${SIXTRACKLIB_C99_FLAGS} ${SIXTRACKLIB_CPU_FLAGS}
        ${SIXTRACKL_C99_AUTOVEC_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_be_beambeam as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_be_drift> CACHE INTERNAL "" FORCE )

# ----------------------------------------------------------------------------
# install :

if( SIXTRL_COMMON_BE_DRIFT_C99_HEADERS )
    install( FILES ${SIXTRL_COMMON_BE_DRIFT_C99_HEADERS}
             DESTINATION "${SIXTRL_COMMON_C99_INSTALL_PREFIX}/be_drift" )
endif()

if( SIXTRACKL_ENABLE_CXX AND SIXTRL_COMMON_BE_DRIFT_CXX_HEADERS )
    install( FILES ${SIXTRL_COMMON_BE_DRIFT_CXX_HEADERS}
             DESTINATION "${SIXTRL_COMMON_CXX_INSTALL_PREFIX}/be_drift" )
 endif()
