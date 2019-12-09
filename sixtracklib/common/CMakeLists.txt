# ==============================================================================
# sixtracklib/common/CMakeLists.txt

message( STATUS "---- processing sixtracklib/common/CMakeLists.txt" )

set( SIXTRL_COMMON_C99_INSTALL_PREFIX
     "${SIXTRACKLIB_C99_INSTALL_PREFIX}/common" )

set( SIXTRL_COMMON_CXX_INSTALL_PREFIX
     "${SIXTRACKLIB_CXX_INSTALL_PREFIX}/common" )

set(  SIXTRACKL_LINK_LIBRARIES ${SIXTRACKL_LINK_LIBRARIES} PARENT_SCOPE )

set( SIXTRACKLIB_COMMON_C99_HEADERS )
set( SIXTRACKLIB_COMMON_CXX_HEADERS )

# add_subdirectory( architecture   )
add_subdirectory( generated      )
add_subdirectory( buffer         )
add_subdirectory( control        )
add_subdirectory( context        )
add_subdirectory( internal       )
add_subdirectory( be_beamfields  )
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
add_subdirectory( particles      )
add_subdirectory( track          )

set(  SIXTRACKLIB_C99_HEADERS ${SIXTRACKLIB_C99_HEADERS}
      beam_elements.h buffer.h constants.h controller.h definitions.h
      particles.h track_job.h track_job_cpu.h )

set(  SIXTRACKLIB_CXX_HEADERS ${SIXTRACKLIB_CXX_HEADERS}
      beam_elements.hpp buffer.hpp particles.hpp )

# ------------------------------------------------------------------------------
# install :

if( SIXTRACKLIB_C99_HEADERS OR SIXTRACKLIB_COMMON_C99_HEADERS )
    install( FILES ${SIXTRACKLIB_C99_HEADERS} ${SIXTRACKLIB_COMMON_C99_HEADERS}
             DESTINATION ${SIXTRL_COMMON_C99_INSTALL_PREFIX} )
endif()

if( SIXTRACKL_ENABLE_CXX AND
    ( SIXTRACKLIB_CXX_HEADERS OR SIXTRACKLIB_COMMON_CXX_HEADERS ) )

    install( FILES ${SIXTRACKLIB_CXX_HEADERS} ${SIXTRACKLIB_COMMON_CXX_HEADERS}
             DESTINATION ${SIXTRL_COMMON_CXX_INSTALL_PREFIX} )
endif()

# end: sixtracklib/common/CMakeLists.txt
