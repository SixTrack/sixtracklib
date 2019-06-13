# ==============================================================================
# tests/sixtracklib/cuda/CMakeLists.txt

message( STATUS "---- processing tests/sixtracklib/cuda/CMakeLists.txt" )

# ==============================================================================
# Setup compilation options and parameters:

if( SIXTRACKL_CUDA_COMPUTE_ARCH_LIST )

    foreach( A IN LISTS SIXTRACKL_CUDA_COMPUTE_ARCH_LIST )
        string( APPEND CMAKE_CUDA_FLAGS
                " -gencode=arch=compute_${A},code=compute_${A}" )
    endforeach()

endif()

string( APPEND CMAKE_CUDA_FLAGS " -D_FORCE_INLINES" )

if( CMAKE_BUILD_TYPE MATCHES "Debug" )
    string( APPEND CMAKE_CUDA_FLAGS " -g -G" )
else()
    string( APPEND CMAKE_CUDA_FLAGS " -lineinfo" )
endif()

# ==============================================================================
# add unit-tests for submodules:

add_subdirectory( control )
add_subdirectory( wrappers )
add_subdirectory( track )

