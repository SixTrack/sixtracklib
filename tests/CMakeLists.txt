# sixtracklib/tests/CMakeLists.txt

# testdata and generators for the testdata:
add_subdirectory( testdata )

# Unit tests for the C/C++ version of the library
add_subdirectory( sixtracklib )

# Unit tests for the python bindings, if enabled
if( SIXTRACKL_ENABLE_PYTHON )
    add_subdirectory( python )
endif()

# Enable benchmarking tests
if( SIXTRACKL_ENABLE_BENCHMARK_TESTS )
    add_subdirectory( benchmark )
endif()
