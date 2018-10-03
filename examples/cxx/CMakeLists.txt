# examples/cxx/CMakeLists.txt

set( CXX_EXAMPLE_PROGRAMS )
set( EXAMPLE_LINK_LIBRARIES sixtrack_test sixtrack m dl )

# -----------------------------------------------------------------------------
# simple_drift_buffer_cxx:

add_executable( simple_drift_buffer_cxx simple_drift_buffer.cpp )
set( CXX_EXAMPLE_PROGRAMS ${CXX_EXAMPLE_PROGRAMS} simple_drift_buffer_cxx )

# -----------------------------------------------------------------------------
# simple_particles_buffer:

# add_executable( simple_particles_buffer_cxx simple_particles_buffer.cpp )
# set( CXX_EXAMPLE_PROGRAMS ${CXX_EXAMPLE_PROGRAMS} simple_particles_buffer_cxx )

# -----------------------------------------------------------------------------
# track_lhc_no_bb_cxx:

add_executable( track_lhc_no_bb_cxx track_lhc_no_bb.cpp )
set( CXX_EXAMPLE_PROGRAMS ${CXX_EXAMPLE_PROGRAMS} track_lhc_no_bb_cxx )

# -----------------------------------------------------------------------------
# track_lhc_no_bb_from_python:

# add_executable( track_simple_bb_from_python_c_cxx track_simple_bb_from_python.cpp )
# set( CXX_EXAMPLE_PROGRAMS ${CXX_EXAMPLE_PROGRAMS} track_simple_bb_from_python_c_cxx )

# -----------------------------------------------------------------------------
# track_lhc_no_bb_cuda_cxx:

# if( SIXTRACKL_ENABLE_CUDA )
#     add_executable( track_lhc_no_bb_cuda_cxx track_lhc_no_bb_cuda.cpp )
#     set( CXX_EXAMPLE_PROGRAMS ${CXX_EXAMPLE_PROGRAMS} track_lhc_no_bb_cuda_cxx )
# endif()

# -----------------------------------------------------------------------------
# track_lhc_no_bb_opencl_cxx:

if( SIXTRACKL_ENABLE_OPENCL )
    add_executable( track_lhc_no_bb_opencl_cxx track_lhc_no_bb_opencl.cpp )
    set( CXX_EXAMPLE_PROGRAMS ${CXX_EXAMPLE_PROGRAMS} track_lhc_no_bb_opencl_cxx )
endif()

# ==============================================================================
# Common settings for all programs:

set(  EXAMPLE_COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} )
list( APPEND EXAMPLE_COMPILE_OPTIONS -Wall )
list( APPEND EXAMPLE_COMPILE_OPTIONS -Werror )
list( APPEND EXAMPLE_COMPILE_OPTIONS -pedantic )

set_property( TARGET ${CXX_EXAMPLE_PROGRAMS} PROPERTY LINKER_LANGUAGE CXX )
set_property( TARGET ${CXX_EXAMPLE_PROGRAMS} PROPERTY CXX_STANDARD 11 )
set_property( TARGET ${CXX_EXAMPLE_PROGRAMS} PROPERTY CXX_STANDARD_REQUIRED ON )
set_property( TARGET ${CXX_EXAMPLE_PROGRAMS}
              APPEND PROPERTY COMPILE_OPTIONS ${EXAMPLE_COMPILE_OPTIONS} )

set_property( TARGET ${CXX_EXAMPLE_PROGRAMS}
              PROPERTY LINK_LIBRARIES ${EXAMPLE_LINK_LIBRARIES} )

# end: examples/cxx/CMakeLists.txt
