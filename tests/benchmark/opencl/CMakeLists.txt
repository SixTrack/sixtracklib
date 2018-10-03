# tests/benchmark/opencl/CMakeLists.txt

set( C99_BENCHMARKS )
set( CXX_BENCHMARKS )

# ------------------------------------------------------------------------------
# benchmark_lhc_no_bb_opencl_c99:

add_executable( benchmark_lhc_no_bb_opencl_c99
                benchmark_lhc_no_bb_opencl_c99.c )

set( C99_BENCHMARKS ${C99_BENCHMARKS} benchmark_lhc_no_bb_opencl_c99 )

# ------------------------------------------------------------------------------
# benchmark_lhc_no_bb_priv_particles_optimized_opencl_c99:

add_executable( benchmark_lhc_no_bb_priv_particles_optimized_opencl_c99
                benchmark_lhc_no_bb_priv_particles_optimized_opencl_c99.c )

set( C99_BENCHMARKS ${C99_BENCHMARKS}
     benchmark_lhc_no_bb_priv_particles_optimized_opencl_c99 )

# ==============================================================================
# Set properties

if( C99_BENCHMARKS )

    set_property( TARGET ${C99_BENCHMARKS}
                  PROPERTY LINK_LIBRARIES ${BENCHMARK_LINK_LIBRARIES} )

    set_property( TARGET ${C99_BENCHMARKS} PROPERTY C_STANDARD 99 )
    set_property( TARGET ${C99_BENCHMARKS} PROPERTY C_STANDARD_REQUIRED ON )

    set( C99_BENCHMARKS_COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} )
    list( APPEND C99_BENCHMARKS_COMPILE_OPTIONS -Wall )
    list( APPEND C99_BENCHMARKS_COMPILE_OPTIONS -Werror )
    list( APPEND C99_BENCHMARKS_COMPILE_OPTIONS -pedantic )

    set_property( TARGET ${C99_BENCHMARKS}
        APPEND PROPERTY COMPILE_OPTIONS ${C99_BENCHMARKS_COMPILE_OPTIONS} )
endif()

if( CXX_BENCHMARKS )

    set_property( TARGET ${CXX_BENCHMARKS}
                  PROPERTY LINK_LIBRARIES ${BENCHMARK_LINK_LIBRARIES} )

    set_property( TARGET ${CXX_BENCHMARKS} PROPERTY CXX_STANDARD 11 )
    set_property( TARGET ${CXX_BENCHMARKS} PROPERTY CXX_STANDARD_REQUIRED ON )

    set( CXX_BENCHMARKS_COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} )
    list( APPEND CXX_BENCHMARKS_COMPILE_OPTIONS -Wall )
    list( APPEND CXX_BENCHMARKS_COMPILE_OPTIONS -Werror )
    list( APPEND CXX_BENCHMARKS_COMPILE_OPTIONS -pedantic )

    set_property( TARGET ${CXX_BENCHMARKS}
        APPEND PROPERTY COMPILE_OPTIONS ${CXX_BENCHMARKS_COMPILE_OPTIONS} )

    set_property( TARGET ${CXX_BENCHMARKS} PROPERTY COMPILE_OPTIONS
                  APPEND PROPERTY COMPILE_OPTIONS
                  ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic )
endif()

#end: tests/benchmark/opencl/CMakeLists.txt
