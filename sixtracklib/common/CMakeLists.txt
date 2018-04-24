# sixtracklib/sixtracklib/common/CMakeLists.txt

set( SIXTRACKLIB_COMMON_HEADERS 
     details/tools.h 
     details/gpu_kernel_tools.h
     block.h
     impl/block_type.h     
     block_drift.h
     impl/block_drift_type.h
     mem_pool.h     
     particles.h
     particles_sequence.h     
     single_particle.h
     impl/particles_type.h     
     tests/test_particles_tools.h     
     details/random.h     
     track.h
     impl/track_single.h          
     values.h
)


set( SIXTRACKLIB_COMMON_SOURCES
     details/block.c
     details/block_drift.c
     details/gpu_kernel_tools.c
     details/mem_pool.c
     details/particles.c
     details/particles_sequence.c
     details/random.c
     details/single_particle.c
     details/track_single.c     
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
    add_subdirectory( tests_benchmark )
endif()

#end sixtracklib/sixtracklib/common/CMakeLists.txt
