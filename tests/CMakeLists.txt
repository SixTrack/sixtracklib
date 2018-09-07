# sixtracklib/tests/CMakeLists.txt

# testdata and generators for the testdata:
add_subdirectory( testdata )

# Unit tests for the C/C++ version of the library
add_subdirectory( sixtracklib )

# Enable benchmarking tests
if( SIXTRACKL_ENABLE_TIMING_TESTS )
    add_subdirectory( benchmark )
endif()
