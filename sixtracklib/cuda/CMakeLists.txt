
set( SIXTRACKLIB_CUDA_PART_HEADERS
     cuda_env.h
)

set( SIXTRACKLIB_CUDA_PART_SOURCES 
     details/cuda_env.c
)

# add_library( sixtrack_cuda_part OBJECT 
#              ${SIXTRACKLIB_CUDA_PART_HEADERS}
#              ${SIXTRACKLIB_CUDA_PART_SOURCES} 
# )

# target_include_directories( sixtrack_cuda_part PUBLIC 
#     ${CMAKE_SOURCE_DIR} ${SIXTRACKL_CUDA_INCLUDE_DIRS} )
# 
# set_target_properties(  sixtrack_cuda_part PROPERTIES LINKER_LANGUAGE C )
# set_target_properties(  sixtrack_cuda_part PROPERTIES POSITION_INDEPENDENT_CODE True )
# set_target_properties(  sixtrack_cuda_part PROPERTIES CXX_STANDARD 11 )
# target_compile_options( sixtrack_cuda_part BEFORE PUBLIC 
#                         ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic -ansi )
    
# ------------------------------------------------------------------------------

set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-g;-G;-O0;-arch=sm_50;-D_GPUCODE=1;" )

cuda_add_library( sixtrack_cuda SHARED 
                  cuda_env.h
                  details/cuda_env.cu 
                  track_particles_kernel.cu
                  track_particles_kernel.cuh )

target_link_libraries( sixtrack_cuda sixtrack )

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )
    add_subdirectory( tests )
endif()
