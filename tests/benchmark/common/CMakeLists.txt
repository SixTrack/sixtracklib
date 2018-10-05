# tests/benchmark/common/CMakeLists.txt

set( C99_BENCHMARKS )
set( CXX_BENCHMARKS )

# -----------------------------------------------------------------------------
# benchmark_lhc_no_bb_common_c99:

add_executable( benchmark_lhc_no_bb_common_c99 benchmark_lhc_no_bb_c99.c )
set( C99_BENCHMARKS ${C99_BENCHMARKS} benchmark_lhc_no_bb_common_c99 )

target_compile_options( benchmark_lhc_no_bb_common_c99 BEFORE PUBLIC
    -fno-fast-math -fno-tree-vectorize )

# -----------------------------------------------------------------------------
# benchmark_lhc_no_bb_common_autovec_c99:

if( SIXTRACKL_ENABLE_AUTOVECTORIZATION )

    add_executable( benchmark_lhc_no_bb_common_autovec_c99
                    benchmark_lhc_no_bb_c99.c )

    set( C99_BENCHMARKS ${C99_BENCHMARKS}
         benchmark_lhc_no_bb_common_autovec_c99 )

    target_compile_options( benchmark_lhc_no_bb_common_autovec_c99
        BEFORE PRIVATE -ftree-vectorize -ftree-vectorizer-verbose=6 -fopt-info-loop
        -fno-fast-math --param vect-max-version-for-alias-checks=50 )

endif()

# =============================================================================

if( C99_BENCHMARKS )
    set_property( TARGET ${C99_BENCHMARKS}
                APPEND PROPERTY LINK_LIBRARIES ${BENCHMARK_LINK_LIBRARIES} )

    set_property( TARGET ${C99_BENCHMARKS} PROPERTY C_STANDARD 99 )
    set_property( TARGET ${C99_BENCHMARKS} PROPERTY C_STANDARD_REQUIRED ON )

    set( C99_BENCHMARKS_COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} )
    list( APPEND C99_BENCHMARKS_COMPILE_OPTIONS -Wall )
    list( APPEND C99_BENCHMARKS_COMPILE_OPTIONS -Werror )
    list( APPEND C99_BENCHMARKS_COMPILE_OPTIONS -pedantic )

    set_property( TARGET ${C99_BENCHMARKS}
        APPEND PROPERTY COMPILE_OPTIONS ${C99_BENCHMARKS_COMPILE_OPTIONS} )

endif()

#end: tests/benchmark/common/CMakeLists.txt
