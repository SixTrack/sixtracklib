# sixtracklib/common/benchmarks/CMakeLists.txt 

# ****************************************************************************
# ****   Baseline Benchmark for SingleParticle:

# -----------------------------------------------------------------------------
# test_drift_baseline:

add_executable( benchmark_drift_baseline benchmark_drift_baseline.cpp )
target_link_libraries( benchmark_drift_baseline ${SIXTRACKL_TEST_LIBRARIES} )
target_compile_options( benchmark_drift_baseline BEFORE PUBLIC 
    -fno-fast-math -fno-tree-vectorize -std=c++11 -Wall -Werror -pedantic )

set_property( TARGET benchmark_drift_baseline PROPERTY CXX_STANDARD 11 )
set_property( TARGET benchmark_drift_baseline PROPERTY CXX_STANDARD_REQUIRED ON )

add_test( CommonBlockDriftBaseLineBenchmarks benchmark_drift_baseline )

# -----------------------------------------------------------------------------
# test_drift_baseline_autovec_sse:

add_executable( benchmark_drift_baseline_autov_sse2 benchmark_drift_baseline.cpp )
target_link_libraries( benchmark_drift_baseline_autov_sse2 ${SIXTRACKL_TEST_LIBRARIES} )
target_compile_options( benchmark_drift_baseline_autov_sse2 BEFORE PUBLIC 
    -fno-fast-math -ftree-vectorize -march=native -mtune=native 
    #-fopt-info-all -fopt-info-all-missed 
    -std=c++11 -Wall -Werror -pedantic )

set_property( TARGET benchmark_drift_baseline_autov_sse2 
              PROPERTY CXX_STANDARD 11 )
              
set_property( TARGET benchmark_drift_baseline_autov_sse2 
              PROPERTY CXX_STANDARD_REQUIRED ON )

add_test( CommonBlockDriftBaseLineAutoVecSSE2Benchmarks 
          benchmark_drift_baseline_autov_sse2 )

# -----------------------------------------------------------------------------
# benchmark_drift_baseline_openmp:

if( SIXTRACKL_ENABLE_OPENMP AND OpenMP_FOUND )
    add_executable( benchmark_drift_baseline_openmp benchmark_drift_baseline.cpp )
    target_link_libraries( benchmark_drift_baseline_openmp ${SIXTRACKL_TEST_LIBRARIES} )
    target_compile_options( benchmark_drift_baseline_openmp BEFORE PUBLIC 
        -fno-fast-math -march=native -mtune=native ${SIXTRACKL_OPENMP_CXX_FLAGS} 
        -std=c++11 -Wall -Werror -pedantic )

    set_target_properties( benchmark_drift_baseline_openmp PROPERTIES LINK_FLAGS 
                        ${SIXTRACKL_OPENMP_CXX_FLAGS} )
        
    set_property( TARGET benchmark_drift_baseline_openmp 
                PROPERTY CXX_STANDARD 11 )
                
    set_property( TARGET benchmark_drift_baseline_openmp 
                PROPERTY CXX_STANDARD_REQUIRED ON )

    add_test( CommonBlockDriftBaseLineOpenMPBenchmarks 
            benchmark_drift_baseline_openmp )
endif()
          
# ****************************************************************************
# ****   Benchmark for Particles:

# -----------------------------------------------------------------------------
# test_drift:

add_executable( benchmark_drift benchmark_drift.cpp )
target_link_libraries( benchmark_drift ${SIXTRACKL_TEST_LIBRARIES} )
target_compile_options( benchmark_drift BEFORE PUBLIC 
    -fno-fast-math -fno-tree-vectorize -std=c++11 -Wall -Werror -pedantic )

set_property( TARGET benchmark_drift PROPERTY CXX_STANDARD 11 )
set_property( TARGET benchmark_drift PROPERTY CXX_STANDARD_REQUIRED ON )

add_test( CommonBlockDriftSingleBenchmarks benchmark_drift )


# ----------------------------------------------------------------------------
# test_drift_autovec_sse2:

add_executable( benchmark_drift_autovec_sse2 benchmark_drift.cpp )
target_link_libraries( benchmark_drift_autovec_sse2 ${SIXTRACKL_TEST_LIBRARIES} )
target_compile_options( benchmark_drift_autovec_sse2 BEFORE PUBLIC 
    -fno-fast-math -ftree-vectorize -march=native -mtune=native 
    #-fopt-info-all -fopt-info-all-missed 
    -std=c++11 -Wall -Werror -pedantic )

set_property( TARGET benchmark_drift_autovec_sse2 PROPERTY CXX_STANDARD 11 )
set_property( TARGET benchmark_drift_autovec_sse2 PROPERTY CXX_STANDARD_REQUIRED ON )

add_test( CommonBlockDriftAutoVecSSE2Benchmarks benchmark_drift_autovec_sse2 )


# -----------------------------------------------------------------------------
# test_drift_openmp

if( SIXTRACKL_ENABLE_OPENMP AND OpenMP_FOUND )
    add_executable( benchmark_drift_openmp benchmark_drift.cpp )
    target_link_libraries( benchmark_drift_openmp ${SIXTRACKL_TEST_LIBRARIES} )
    target_compile_options( benchmark_drift_openmp BEFORE PUBLIC 
        -fno-fast-math -march=native -mtune=native ${SIXTRACKL_OPENMP_CXX_FLAGS} 
        #-fopt-info-all -fopt-info-all-missed 
        -std=c++11 -Wall -Werror -pedantic )
        
    set_target_properties( benchmark_drift_openmp PROPERTIES LINK_FLAGS 
                          ${SIXTRACKL_OPENMP_CXX_FLAGS} )

    set_property( TARGET benchmark_drift_openmp PROPERTY CXX_STANDARD 11 )
    set_property( TARGET benchmark_drift_openmp PROPERTY CXX_STANDARD_REQUIRED ON )

    add_test( CommonBlockDriftOpenMPBenchmarks benchmark_drift_openmp )

    
endif()



