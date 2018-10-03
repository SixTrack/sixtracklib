set( SIXTRL_COMMON_BE_BEAMBEAM_SOURCES
    be_beambeam4d.c
    be_beambeam6d.c
)

set( SIXTRL_COMMON_BE_BEAMBEAM_HEADERS
     be_beambeam4d.h
     be_beambeam6d.h
     track_beambeam.h
     faddeeva_cern.h
     gauss_fields.h
)

add_library( sixtrack_common_be_beambeam OBJECT
    ${SIXTRL_COMMON_BE_BEAMBEAM_HEADERS}
    ${SIXTRL_COMMON_BE_BEAMBEAM_SOURCES}
)

set(  SIXTRACKLIB_COMMON_MODULES
    ${SIXTRACKLIB_COMMON_MODULES} sixtrack_common_be_beambeam PARENT_SCOPE )

target_include_directories( sixtrack_common_be_beambeam
                            PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> )

# ------------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_COMMON_BE_BEAMBEAM_INSTALL_PATH
     include/sixtracklib/common/be_beambeam )

install( FILES ${SIXTRL_COMMON_BE_BEAMBEAM_HEADERS}
         DESTINATION ${SIXTRACKLIB_COMMON_BE_BEAMBEAM_INSTALL_PATH} )

