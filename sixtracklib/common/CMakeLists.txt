# sixtracklib/sixtracklib/common/CMakeLists.txt

set( SIXTRACKLIB_COMMON_IMPL_HEADERS
     impl/alignment_impl.h
     impl/beam_beam_element_6d.h
     impl/beam_elements_type.h
     impl/beam_elements_api.h
     impl/faddeeva.h
     impl/faddeeva_mit.h
     impl/particles_type.h
     impl/particles_api.h
)

set( SIXTRACKLIB_COMMON_DETAILS_HEADERS
     details/tools.h
     details/gpu_kernel_tools.h
     details/random.h
)

set( SIXTRACKLIB_COMMON_TESTS_HEADERS
     tests/test_particles_tools.h
     tests/test_track_tools.h
)

set( SIXTRACKLIB_COMMON_HEADERS
     alignment.h
     beam_elements.h
     blocks.h
     compute_arch.h
     mem_pool.h
     particles.h
)

set( SIXTRACKLIB_COMMON_SOURCES
     details/alignment.c
     details/beam_elements.c
     details/blocks.c
     details/compute_arch.c
     details/faddeeva_mit.c
     details/gpu_kernel_tools.c
     details/mem_pool.c
     details/particles.c
     details/random.c
     tests/test_particles_tools.c
     tests/test_track_tools.c
)

add_library( sixtrack_common OBJECT
             ${SIXTRACKLIB_COMMON_IMPL_HEADERS}
             ${SIXTRACKLIB_COMMON_DETAILS_HEADERS}
             ${SIXTRACKLIB_COMMON_TESTS_HEADERS}
             ${SIXTRACKLIB_COMMON_HEADERS}
             ${SIXTRACKLIB_COMMON_SOURCES} )

target_include_directories( sixtrack_common
                            PUBLIC $<INSTALL_INTERFACE:include>
                                   $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> )

set_target_properties(  sixtrack_common PROPERTIES LINKER_LANGUAGE C )
set_target_properties(  sixtrack_common PROPERTIES POSITION_INDEPENDENT_CODE True )
set_target_properties(  sixtrack_common PROPERTIES C_STANDARD 99 )

target_compile_definitions( sixtrack_common PUBLIC
                            ${SIXTRACKLIB_NAMESPACE_FLAGS} )

target_compile_options( sixtrack_common BEFORE PUBLIC
                        ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic -ansi )

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )
    add_subdirectory( tests )
endif()

if( SIXTRACKL_ENABLE_TIMING_TESTS )
#     add_subdirectory( tests_benchmark )
endif()


set( SIXTRACKLIB_COMMON_INSTALL_PATH include/sixtracklib/common )

install( FILES ${SIXTRACKLIB_COMMON_HEADERS}
         DESTINATION ${SIXTRACKLIB_COMMON_INSTALL_PATH} )

install( FILES ${SIXTRACKLIB_COMMON_IMPL_HEADERS}
         DESTINATION "${SIXTRACKLIB_COMMON_INSTALL_PATH}/impl" )

#end sixtracklib/sixtracklib/common/CMakeLists.txt
