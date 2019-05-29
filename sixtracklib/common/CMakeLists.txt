# ==============================================================================
# common sixtracklib submodule

message( STATUS "---- processing sixtracklib/common/CMakeLists.txt" )

set(  SIXTRACKL_LINK_LIBRARIES
    ${SIXTRACKL_LINK_LIBRARIES}  PARENT_SCOPE )

set( SIXTRACKLIB_COMMON_C99_HEADERS )
set( SIXTRACKLIB_COMMON_CXX_HEADERS )

add_subdirectory( generated      )

add_subdirectory( buffer         )
add_subdirectory( context        )
add_subdirectory( internal       )
add_subdirectory( be_beambeam    )
add_subdirectory( be_drift       )
add_subdirectory( be_cavity      )
add_subdirectory( be_multipole   )
add_subdirectory( be_rfmultipole )
add_subdirectory( be_srotation   )
add_subdirectory( be_xyshift     )
add_subdirectory( be_monitor     )
add_subdirectory( be_limit       )
add_subdirectory( be_dipedge     )
add_subdirectory( output         )

set( SIXTRACKLIB_COMMON_INTERNAL_CXX_HEADERS )
set( SIXTRACKLIB_COMMON_INTERNAL_CXX_SOURCES )

set(  SIXTRACKLIB_C99_HEADERS
    ${SIXTRACKLIB_C99_HEADERS}
      beam_elements.h
      buffer.h
      constants.h
      context.h
      definitions.h
      particles.h
      track.h
      track_job.h
      track_job_cpu.h
)

set(  SIXTRACKLIB_CXX_HEADERS
    ${SIXTRACKLIB_CXX_HEADERS}
      beam_elements.hpp
      buffer.hpp
      particles.hpp
)

# ------------------------------------------------------------------------------
# install :

if( SIXTRACKLIB_C99_HEADERS )

    set( SIXTRACKLIB_COMMON_C99_INSTALL_PATH
         include/sixtracklib/common )

    install( FILES ${SIXTRACKLIB_C99_HEADERS}
             DESTINATION ${SIXTRACKLIB_COMMON_C99_INSTALL_PATH} )
endif()

if( SIXTRACKLIB_CXX_HEADERS )

    set( SIXTRACKLIB_COMMON_CXX_INSTALL_PATH
         include/sixtracklib/common )

    install( FILES ${SIXTRACKLIB_CXX_HEADERS}
             DESTINATION ${SIXTRACKLIB_COMMON_CXX_INSTALL_PATH} )
endif()

#end sixtracklib/common/CMakeLists.txt
