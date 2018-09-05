# tests/benchmark/sixtracklib/common/CMakeLists.txt

set( BENCHMARK_COMMON_LINK_LIBRARIES sixtrack_test sixtrack m dl )

# ****************************************************************************
# ****   Baseline Benchmark for SingleParticle:

# -----------------------------------------------------------------------------
# test_drift_baseline:

add_executable( benchmark_common_drift_baseline benchmark_drift_baseline.cpp )
target_link_libraries( benchmark_common_drift_baseline ${BENCHMARK_COMMON_LINK_LIBRARIES} )
target_compile_options( benchmark_common_drift_baseline BEFORE PUBLIC
    -fno-fast-math -fno-tree-vectorize -std=c++11 -Wall -Werror -pedantic )

set_property( TARGET benchmark_common_drift_baseline PROPERTY CXX_STANDARD 11 )
set_property( TARGET benchmark_common_drift_baseline PROPERTY CXX_STANDARD_REQUIRED ON )

