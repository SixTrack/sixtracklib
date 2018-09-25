# tests/sixtracklib/benchmark/cuda/CMakeLists.txt

set( BENCHMARK_LINK_LIBRARIES sixtrack_test sixtrack m )
set( BENCHMARK_C99_PROGRAMS )
set( BENCHMARK_CXX_PROGRAMS )


# ------------------------------------------------------------------------------
# benchmark_lhc_no_bb_cuda_c99:

add_executable( benchmark_lhc_no_bb_cuda_c99
                benchmark_lhc_no_bb_cuda_c99.c )

set( BENCHMARK_C99_PROGRAMS ${BENCHMARK_C99_PROGRAMS}
     benchmark_lhc_no_bb_cuda_c99 )


# ==============================================================================
# set properties:

if( BENCHMARK_CXX_PROGRAMS )

    set_property( TARGET ${BENCHMARK_CXX_PROGRAMS}
                  PROPERTY LINK_LIBRARIES ${BENCHMARK_LINK_LIBRARIES} )

    set_property( TARGET ${BENCHMARK_CXX_PROGRAMS} PROPERTY CXX_STANDARD 11 )
    set_property( TARGET ${BENCHMARK_CXX_PROGRAMS} PROPERTY CXX_STANDARD_REQUIRED ON )

endif()

if( BENCHMARK_C99_PROGRAMS )

    set_property( TARGET ${BENCHMARK_C99_PROGRAMS}
                  PROPERTY LINK_LIBRARIES ${BENCHMARK_LINK_LIBRARIES} )

    set_property( TARGET ${BENCHMARK_C99_PROGRAMS} PROPERTY C_STANDARD 99 )
    set_property( TARGET ${BENCHMARK_C99_PROGRAMS} PROPERTY C_STANDARD_REQUIRED ON )

endif()


