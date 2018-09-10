set( GENERATORS_C99_TARGETS )
set( GENERATORS_LINK_LIBRARIES sixtrack_test sixtrack m dl )

set( SIXTRACKLIB_TESTDATA_DIR "${CMAKE_SOURCE_DIR}/tests/testdata" )
set( TESTDATA_TARGET_OUTPUT   "{SIXTRACKLIB_TESTDATA_DIR}/testdata.generated" )

# -----------------------------------------------------------------------------
# generate_buffer_generic_obj:

add_executable( generate_buffer_generic_obj generate_buffer_generic_obj.c )
set( GENERATORS_C99_TARGETS ${GENERATORS_C99_TARGETS}
     generate_buffer_generic_obj )

# -----------------------------------------------------------------------------
# generate_track_be_drift_data:

add_executable( generate_track_be_drift_data generate_track_be_drift_data.c )
set( GENERATORS_C99_TARGETS ${GENERATORS_C99_TARGETS}
     generate_track_be_drift_data )

# -----------------------------------------------------------------------------
# generate_track_be_driftexact_data:

add_executable( generate_track_be_driftexact_data
                generate_track_be_driftexact_data.c )

set( GENERATORS_C99_TARGETS ${GENERATORS_C99_TARGETS}
     generate_track_be_driftexact_data )

# =============================================================================
# Add properties for all C99 targets:

set_property( TARGET ${GENERATORS_C99_TARGETS}
              PROPERTY LINK_LIBRARIES ${GENERATORS_LINK_LIBRARIES} )

set_property( TARGET ${GENERATORS_C99_TARGETS} PROPERTY C_STANDARD 99 )
set_property( TARGET ${GENERATORS_C99_TARGETS} PROPERTY C_STANDARD_REQUIRED ON )

set_property( TARGET ${UNIT_TEST_TARGETS}
              APPEND PROPERTY COMPILE_OPTIONS
              ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic )

# =============================================================================
# Create custom command to call all generators:

add_custom_command(
    OUTPUT  ${TESTDATA_TARGET_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E echo "1" >
        ${SIXTRACKLIB_TESTDATA_DIR}/testdata.generated
    COMMENT "Generate testdata ... "
    VERBATIM
)

foreach( GENERATOR_CMD ${GENERATORS_C99_TARGETS} )
    add_custom_command( OUTPUT  ${TESTDATA_TARGET_OUTPUT}
                        APPEND  COMMAND ${GENERATOR_CMD} )
endforeach()


add_custom_target( testdata DEPENDS ${TESTDATA_TARGET_OUTPUT} )


