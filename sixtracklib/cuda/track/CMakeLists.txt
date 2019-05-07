# ==============================================================================
# sixtracklib/cuda/track submodule

set( SIXTRACKLIB_CUDA_TRACK_CXX_HEADERS
)

set( SIXTRACKLIB_CUDA_TRACK_CXX_SOURCES
     track_job.cpp
)

set( SIXTRACKLIB_CUDA_TRACK_C99_HEADERS
)

add_library( sixtrack_cuda_cxx_track OBJECT
             ${SIXTRACKLIB_CUDA_TRACK_CXX_HEADERS}
             ${SIXTRACKLIB_CUDA_TRACK_C99_HEADERS}
             ${SIXTRACKLIB_CUDA_TRACK_CXX_SOURCES}
)

target_compile_definitions( sixtrack_cuda_cxx_track
    PUBLIC ${SIXTRACKLIB_NAMESPACE_FLAGS} )

target_include_directories( sixtrack_cuda_cxx_track
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> )

set_target_properties( sixtrack_cuda_cxx_track PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    LINKER_LANGUAGE C
    CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON )

# ------------------------------------------------------------------------------
# pass on sixtrack_cuda_cxx_track as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES
    ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_cuda_cxx_track> CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------