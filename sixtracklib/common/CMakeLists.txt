# sixtracklib/sixtracklib/common/CMakeLists.txt

set( SIXTRACKLIB_COMMON_HEADERS 
     alignment.h
     impl/alignment_impl.h
     blocks.h
     beam_elements.h
     impl/beam_elements_api.h
     mem_pool.h     
     particles.h
     impl/particles_api.h
     tests/test_particles_tools.h     
     details/tools.h 
     details/gpu_kernel_tools.h
     details/random.h     
)


set( SIXTRACKLIB_COMMON_SOURCES
     details/alignment.c
     details/blocks.c
     details/beam_elements.c
     details/gpu_kernel_tools.c
     details/mem_pool.c
     details/particles.c
     details/random.c
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
