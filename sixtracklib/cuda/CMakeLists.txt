string( APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_30,code=compute_30" )
string( APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_50,code=compute_50" )
string( APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_60,code=compute_60" )
string( APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=compute_61" )
string( APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_62,code=compute_62" )

set( SIXTRACKL_CUDA_DEFINITIONS "_GPUCODE=1" )
list( APPEND SIXTRACKL_CUDA_DEFINITIONS "__NAMESPACE=st_" )

set( SIXTRACKLIB_CUDA_PART_HEADERS
     cuda_env.h
     impl/track_particles_kernel.cuh
)

set( SIXTRACKLIB_CUDA_PART_SOURCES 
     details/cuda_env.cu
     details/track_particles_kernel.cu
)

add_library( sixtrack_cuda OBJECT
             ${SIXTRACKLIB_CUDA_PART_HEADERS}
             ${SIXTRACKLIB_CUDA_PART_SOURCES}
)

target_compile_definitions( sixtrack_cuda PUBLIC ${SIXTRACKL_CUDA_DEFINITIONS} )

set_target_properties( sixtrack_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties( sixtrack_cuda PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack_cuda PROPERTIES CXX_STANDARD 11 )
set_target_properties( sixtrack_cuda PROPERTIES CXX_STANDARD_REQUIRED ON )
set_target_properties( sixtrack_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION  ON  )
set_target_properties( sixtrack_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
set_target_properties( sixtrack_cuda PROPERTIES CUDA_STANDARD 11 )
set_target_properties( sixtrack_cuda PROPERTIES CUDA_STANDARD_REQUIRED 11 )

# ------------------------------------------------------------------------------

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )
    add_subdirectory( tests )
endif()

# ------------------------------------------------------------------------------

# set( CUDA_PROPAGATE_HOST_FLAGS ${SAVED_CUDA_PROPAGATE_HOST_FLAGS} )

