# sixtracklib/common/benchmarks/CMakeLists.txt 

# ----------------------------------------------------------------------------
# test_drift_base:

add_executable( benchmark_drift_base benchmark_drift.cpp )
target_link_libraries( benchmark_drift_base ${SIXTRACKL_TEST_LIBRARIES} )
target_compile_options( benchmark_drift_base BEFORE PUBLIC 
    -O3 -fno-fast-math -fno-tree-vectorize -std=c++11 -Wall -Werror -pedantic )

set_property( TARGET benchmark_drift_base PROPERTY CXX_STANDARD 11 )
set_property( TARGET benchmark_drift_base PROPERTY CXX_STANDARD_REQUIRED ON )

add_test( CommonBlockDriftBaseBenchmarks benchmark_drift_base )


# ----------------------------------------------------------------------------
# test_drift_autovec_sse2:

add_executable( benchmark_drift_autovec_sse2 benchmark_drift.cpp )
target_link_libraries( benchmark_drift_autovec_sse2 ${SIXTRACKL_TEST_LIBRARIES} )
target_compile_options( benchmark_drift_autovec_sse2 BEFORE PUBLIC 
    -O3 -fno-fast-math -ftree-vectorize -msse2 -ftree-vectorizer-verbose=6 -std=c++11 -Wall -Werror -pedantic )

set_property( TARGET benchmark_drift_autovec_sse2 PROPERTY CXX_STANDARD 11 )
set_property( TARGET benchmark_drift_autovec_sse2 PROPERTY CXX_STANDARD_REQUIRED ON )

add_test( CommonBlockDriftAutoVecSSE2Benchmarks benchmark_drift_autovec_sse2 )


