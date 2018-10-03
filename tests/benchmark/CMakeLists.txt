# tests/benchmark/CMakeLists.txt

if( SIXTRACKL_ENABLE_TIMING_TESTS )

    set( BENCHMARK_LINK_LIBRARIES sixtrack_test sixtrack m dl )

    add_subdirectory( common )

    if( SIXTRACKL_ENABLE_OPENCL )
        add_subdirectory( opencl )
    endif()

    if( SIXTRACKL_ENABLE_CUDA )
        add_subdirectory( cuda )
    endif()

endif()

#end: tests/benchmark/CMakeLists.txt
