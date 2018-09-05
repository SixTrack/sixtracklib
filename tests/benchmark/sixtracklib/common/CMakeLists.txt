# tests/benchmark/sixtracklib/common/CMakeLists.txt

set( BENCHMARK_LINK_LIBRARIES sixtrack_test sixtrack m dl )

set( BENCHMARK_C99_PROGRAMS )

# -----------------------------------------------------------------------------
# benchmark_lhc_no_bb_common_c99:

add_executable( benchmark_lhc_no_bb_common_c99 benchmark_lhc_no_bb_c99.c )

set( BENCHMARK_C99_PROGRAMS ${BENCHMARK_C99_PROGRAMS}
     benchmark_lhc_no_bb_common_c99 )

target_compile_options( benchmark_lhc_no_bb_common_c99 BEFORE PUBLIC
    -fno-fast-math -fno-tree-vectorize )

# -----------------------------------------------------------------------------
# benchmark_lhc_no_bb_common_autovec_c99:

if( SIXTRACKL_ENABLE_AUTOVECTORIZATION )
    add_executable( benchmark_lhc_no_bb_common_autovec_c99
                    benchmark_lhc_no_bb_c99.c )

    set( BENCHMARK_C99_PROGRAMS ${BENCHMARK_C99_PROGRAMS}
         benchmark_lhc_no_bb_common_autovec_c99 )

    target_compile_options( benchmark_lhc_no_bb_common_autovec_c99
        BEFORE PRIVATE -ftree-vectorize -ftree-vectorizer-verbose=6 -fopt-info-loop
        -fno-fast-math --param vect-max-version-for-alias-checks=50 )
endif()

# =============================================================================

if( BENCHMARK_C99_PROGRAMS )
    set_property( TARGET ${BENCHMARK_C99_PROGRAMS}
                APPEND PROPERTY LINK_LIBRARIES ${BENCHMARK_LINK_LIBRARIES} )

    set_property( TARGET ${BENCHMARK_C99_PROGRAMS}
                PROPERTY C_STANDARD 99 )

    set_property( TARGET ${BENCHMARK_C99_PROGRAMS}
                PROPERTY C_STANDARD_REQUIRED ON )

    set( BENCHMARK_C99_COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} )
    list( APPEND BENCHMARK_C99_COMPILE_OPTIONS -Wall )
    list( APPEND BENCHMARK_C99_COMPILE_OPTIONS -Werror )
    list( APPEND BENCHMARK_C99_COMPILE_OPTIONS -pedantic )

    set_property( TARGET ${BENCHMARK_C99_PROGRAMS}
                  APPEND PROPERTY COMPILE_OPTIONS
                    ${BENCHMARK_C99_COMPILE_OPTIONS} )
endif()