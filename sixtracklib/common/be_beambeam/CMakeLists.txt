set( SIXTRL_COMMON_BE_BEAMBEAM_SOURCES
    be_beambeam.c
)

set( SIXTRL_COMMON_BE_BEAMBEAM_HEADERS
     be_beambeam.h
     track.h
     faddeeva_cern.h
     gauss_fields.h
)

add_library( sixtrack_common_be_beambeam OBJECT
    ${SIXTRL_COMMON_BE_BEAMBEAM_HEADERS}
    ${SIXTRL_COMMON_BE_BEAMBEAM_SOURCES}
)


target_include_directories( sixtrack_common_be_beambeam PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_be_beambeam PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack_common_be_beambeam PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties( sixtrack_common_be_beambeam PROPERTIES C_STANDARD 99 )
set_target_properties( sixtrack_common_be_beambeam PROPERTIES C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_be_beambeam PRIVATE
                        -Wall -Werror -pedantic -ansi
                        ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_be_beambeam as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES
    ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_be_beambeam> CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_COMMON_BE_BEAMBEAM_C99_INSTALL_PATH
     include/sixtracklib/common/be_beambeam )

install( FILES ${SIXTRL_COMMON_BE_BEAMBEAM_HEADERS}
         DESTINATION ${SIXTRACKLIB_COMMON_BE_BEAMBEAM_C99_INSTALL_PATH} )

