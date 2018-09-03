# examples/c99/CMakeLists.txt

set( EXAMPLE_PROGRAMS )
set( EXAMPLE_LINK_LIBRARIES sixtrack_test sixtrack m dl )

# -----------------------------------------------------------------------------
# simple_drift_buffer:

add_executable( simple_drift_buffer_c99 simple_drift_buffer.c )
set( EXAMPLE_PROGRAMS ${EXAMPLE_PROGRAMS} simple_drift_buffer_c99 )

target_compile_options( simple_drift_buffer_c99 PRIVATE
    -fno-fast-math -fno-tree-vectorize )

# -----------------------------------------------------------------------------
# simple_particles_buffer:

add_executable( simple_particles_buffer_c99 simple_particles_buffer.c )
set( EXAMPLE_PROGRAMS ${EXAMPLE_PROGRAMS} simple_particles_buffer_c99 )

target_compile_options( simple_particles_buffer_c99 PRIVATE
    -fno-fast-math -fno-tree-vectorize )

# -----------------------------------------------------------------------------
# simple_drift_autovec:

# if( SIXTRACKL_ENABLE_AUTOVECTORIZATION )
#     add_executable( simple_drift_autovec simple_drift.c )
#     set( EXAMPLE_PROGRAMS ${EXAMPLE_PROGRAMS} simple_drift_autovec )
#
#     target_compile_options( simple_drift_autovec PRIVATE
#         -ftree-vectorize -ftree-vectorizer-verbose=6 -fopt-info-loop
#         -fno-fast-math --param vect-max-version-for-alias-checks=50 )
# endif()

# -----------------------------------------------------------------------------
# simple_drift_opencl:

# # if( SIXTRACKL_ENABLE_OPENCL )
#     add_executable( simple_drift_opencl simple_drift_opencl.c )
#     set( EXAMPLE_PROGRAMS ${EXAMPLE_PROGRAMS} simple_drift_opencl )
#
# endif()

# -----------------------------------------------------------------------------
# simple_drift_cuda:

# if( SIXTRACKL_ENABLE_CUDA )
#     add_executable( simple_drift_cuda simple_drift_cuda.c )
#     set( EXAMPLE_PROGRAMS ${EXAMPLE_PROGRAMS} simple_drift_cuda )
#
# endif()


# ==============================================================================
# Common settings for all programs:

set(  EXAMPLE_COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} )
list( APPEND EXAMPLE_COMPILE_OPTIONS -Wall )
list( APPEND EXAMPLE_COMPILE_OPTIONS -Werror )
list( APPEND EXAMPLE_COMPILE_OPTIONS -pedantic )

set_property( TARGET ${EXAMPLE_PROGRAMS} PROPERTY LINKER_LANGUAGE C )
set_property( TARGET ${EXAMPLE_PROGRAMS} PROPERTY C_STANDARD 99 )
set_property( TARGET ${EXAMPLE_PROGRAMS} PROPERTY C_STANDARD_REQUIRED ON )
set_property( TARGET ${EXAMPLE_PROGRAMS}
              APPEND PROPERTY COMPILE_OPTIONS ${EXAMPLE_COMPILE_OPTIONS} )

set_property( TARGET ${EXAMPLE_PROGRAMS}
              PROPERTY LINK_LIBRARIES ${EXAMPLE_LINK_LIBRARIES} )

# end: examples/c99/CMakeLists.txt
