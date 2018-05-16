# sixtracklib/sixtracklib/common/CMakeLists.txt

set( SIXTRACKLIB_COMMON_HEADERS 
     details/tools.h 
     details/gpu_kernel_tools.h
     alignment.h
     impl/alignment_impl.h
     be_drift.h
     impl/be_drift_impl.h
     block_info.h
     impl/block_info_impl.h
     beam_elements.h
     blocks_container.h
     mem_pool.h     
     particles.h
     single_particle.h
     tests/test_particles_tools.h     
     details/random.h     
)


set( SIXTRACKLIB_COMMON_SOURCES
     details/alignment.c
     details/be_drift.c
     details/block_info.c
     details/blocks_container.c
     details/gpu_kernel_tools.c
     details/mem_pool.c
     details/particles.c
     details/random.c
     details/single_particle.c
     tests/test_particles_tools.c
)

add_library( sixtrack_common OBJECT 
             ${SIXTRACKLIB_COMMON_HEADERS}
             ${SIXTRACKLIB_COMMON_SOURCES} )
             
target_include_directories( sixtrack_common PUBLIC ${CMAKE_SOURCE_DIR} )
             
set_target_properties(  sixtrack_common PROPERTIES LINKER_LANGUAGE C )
set_target_properties(  sixtrack_common PROPERTIES POSITION_INDEPENDENT_CODE True )
set_target_properties(  sixtrack_common PROPERTIES C_STANDARD 99 )
target_compile_options( sixtrack_common BEFORE PUBLIC 
                        ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic -ansi )

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )
    add_subdirectory( tests )
endif()

if( SIXTRACKL_ENABLE_TIMING_TESTS )
#     add_subdirectory( tests_benchmark )
endif()

#end sixtracklib/sixtracklib/common/CMakeLists.txt
