set( GENERATORS_C99_TARGETS )
set( GENERATORS_LINK_LIBRARIES sixtrack_test sixtrack m dl )

# -----------------------------------------------------------------------------
# generate_buffer_generic_obj:

add_executable( generate_buffer_generic_obj generate_buffer_generic_obj.c )
set( GENERATORS_C99_TARGETS ${GENERATORS_C99_TARGETS}
     generate_buffer_generic_obj )


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

# add_custom_command(
#     OUTPUT  "${SIXTRACKLIB_TESTDATA_DIR}/testdata.generated"
#     COMMAND
#     DEPENDS ${GENERATORS_C99_TARGETS}

