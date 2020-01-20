set( SIXTRL_COMMON_BE_BEAMFIELDS_SOURCES
    be_beamfields.c
)

set( SIXTRL_COMMON_BE_BEAMFIELDS_HEADERS
     be_beamfields.h
     track.h
     faddeeva_cern.h
     gauss_fields.h
)

add_library( sixtrack_common_be_beamfields OBJECT
    ${SIXTRL_COMMON_BE_BEAMFIELDS_HEADERS}
    ${SIXTRL_COMMON_BE_BEAMFIELDS_SOURCES}
)


target_include_directories( sixtrack_common_be_beamfields PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_be_beamfields PROPERTIES
    LINKER_LANGUAGE C POSITION_INDEPENDENT_CODE ON C_STANDARD 99
    C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_be_beamfields PRIVATE
                        ${SIXTRACKLIB_C99_FLAGS} ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_be_beamfields as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES
    ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_be_beamfields> CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_COMMON_BE_BEAMFIELDS_C99_INSTALL_PATH
     include/sixtracklib/common/be_beamfields )

install( FILES ${SIXTRL_COMMON_BE_BEAMFIELDS_HEADERS}
         DESTINATION ${SIXTRACKLIB_COMMON_BE_BEAMFIELDS_C99_INSTALL_PATH} )

