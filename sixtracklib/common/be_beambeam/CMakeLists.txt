set( SIXTRL_COMMON_BE_BEAM_BEAM_SOURCES
    be_beambeam4d.c
    be_beambeam6d.c
)

set( SIXTRL_COMMON_BE_BEAM_BEAM_HEADERS
     be_beambeam4d.h
     be_beambeam6d.h
     track_beambeam.h
)

add_library( sixtrack_common_beambeam OBJECT
    ${SIXTRL_COMMON_BE_BEAM_BEAM_HEADERS}
    ${SIXTRL_COMMON_BE_BEAM_BEAM_SOURCES}
)

target_include_directories( sixtrack_common_beambeam PRIVATE ${CMAKE_SOURCE_DIR} )

# ------------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_COMMON_BE_BEAMBEAM_INSTALL_PATH
     include/sixtracklib/common/be_beambeam )

install( FILES ${SIXTRL_COMMON_BE_BEAM_BEAM_HEADERS}
         DESTINATION ${SIXTRACKLIB_COMMON_BE_BEAMBEAM_INSTALL_PATH} )

