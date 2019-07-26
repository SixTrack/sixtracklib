# ==============================================================================
# sixtracklib/cuda/track submodule

# ------------------------------------------------------------------------------

set( SIXTRACKLIB_CUDA_TRACK_C99_INSTALL_PREFIX
     "${SIXTRACKLIB_CUDA_C99_INSTALL_PREFIX}/track" )

set( SIXTRACKLIB_CUDA_TRACK_CXX_INSTALL_PREFIX
     "${SIXTRACKLIB_CUDA_CXX_INSTALL_PREFIX}/track" )

 # ------------------------------------------------------------------------------

set( SIXTRACKLIB_CUDA_TRACK_CXX_HEADERS )
set( SIXTRACKLIB_CUDA_TRACK_CXX_SOURCES
     track_job.cpp
     track_job_c99.cpp
)

set( SIXTRACKLIB_CUDA_TRACK_C99_HEADERS )

add_library( sixtrack_cuda_cxx_track OBJECT
             ${SIXTRACKLIB_CUDA_TRACK_CXX_HEADERS}
             ${SIXTRACKLIB_CUDA_TRACK_C99_HEADERS}
             ${SIXTRACKLIB_CUDA_TRACK_CXX_SOURCES}
)

target_compile_definitions( sixtrack_cuda_cxx_track
    PUBLIC ${SIXTRACKLIB_NAMESPACE_FLAGS} )

target_include_directories( sixtrack_cuda_cxx_track
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    PUBLIC  $<BUILD_INTERFACE:${SIXTRACKL_CUDA_INCLUDE_DIRS}> )

set_target_properties( sixtrack_cuda_cxx_track PROPERTIES
    POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE C
    CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON )

# ------------------------------------------------------------------------------
# pass on sixtrack_cuda_cxx_track as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_cuda_cxx_track> CACHE INTERNAL "" FORCE )

# ==============================================================================
# Handle install

if( SIXTRACKLIB_CUDA_TRACK_C99_HEADERS )
    install( FILES ${SIXTRACKLIB_CUDA_TRACK_C99_HEADERS}
             DESTINATION ${SIXTRACKLIB_CUDA_TRACK_C99_INSTALL_PREFIX} )
endif()


if( SIXTRACKLIB_CUDA_TRACK_CXX_HEADERS AND SIXTRACKLIB_ENABLE_CXX )
    install( FILES ${SIXTRACKLIB_CUDA_TRACK_CXX_HEADERS}
             DESTINATION ${SIXTRACKLIB_CUDA_TRACK_CXX_INSTALL_PREFIX} )
endif()

#end: sixtracklib/cuda/track/CMakeLists.txt
