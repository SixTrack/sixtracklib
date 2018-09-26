# examples/c99/CMakeLists.txt

set( C99_EXAMPLE_PROGRAMS )
set( EXAMPLE_LINK_LIBRARIES sixtrack_test sixtrack m dl )

# -----------------------------------------------------------------------------
# simple_drift_buffer:

add_executable( simple_drift_buffer_c99 simple_drift_buffer.c )
set( C99_EXAMPLE_PROGRAMS ${C99_EXAMPLE_PROGRAMS} simple_drift_buffer_c99 )

# -----------------------------------------------------------------------------
# simple_particles_buffer:

add_executable( simple_particles_buffer_c99 simple_particles_buffer.c )
set( C99_EXAMPLE_PROGRAMS ${C99_EXAMPLE_PROGRAMS} simple_particles_buffer_c99 )

# -----------------------------------------------------------------------------
# track_lhc_no_bb_c99:

add_executable( track_lhc_no_bb_c99 track_lhc_no_bb.c )
set( C99_EXAMPLE_PROGRAMS ${C99_EXAMPLE_PROGRAMS} track_lhc_no_bb_c99 )

# -----------------------------------------------------------------------------
# track_lhc_no_bb_from_python:

add_executable( track_lhc_no_bb_from_python_c99 track_lhc_no_bb_from_python.c )
set( C99_EXAMPLE_PROGRAMS ${C99_EXAMPLE_PROGRAMS} track_lhc_no_bb_from_python_c99 )

# -----------------------------------------------------------------------------
# track_lhc_no_bb_cuda_c99:

if( SIXTRACKL_ENABLE_CUDA )
    add_executable( track_lhc_no_bb_cuda_c99 track_lhc_no_bb_cuda.c )
    set( C99_EXAMPLE_PROGRAMS ${C99_EXAMPLE_PROGRAMS} track_lhc_no_bb_cuda_c99 )
endif()

# ==============================================================================
# Common settings for all programs:

set(  EXAMPLE_COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} )
list( APPEND EXAMPLE_COMPILE_OPTIONS -Wall )
list( APPEND EXAMPLE_COMPILE_OPTIONS -Werror )
list( APPEND EXAMPLE_COMPILE_OPTIONS -pedantic )

set_property( TARGET ${C99_EXAMPLE_PROGRAMS} PROPERTY LINKER_LANGUAGE C )
set_property( TARGET ${C99_EXAMPLE_PROGRAMS} PROPERTY C_STANDARD 99 )
set_property( TARGET ${C99_EXAMPLE_PROGRAMS} PROPERTY C_STANDARD_REQUIRED ON )
set_property( TARGET ${C99_EXAMPLE_PROGRAMS}
              APPEND PROPERTY COMPILE_OPTIONS ${EXAMPLE_COMPILE_OPTIONS} )

set_property( TARGET ${C99_EXAMPLE_PROGRAMS}
              PROPERTY LINK_LIBRARIES ${EXAMPLE_LINK_LIBRARIES} )

# end: examples/c99/CMakeLists.txt
