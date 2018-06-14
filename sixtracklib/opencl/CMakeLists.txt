set( SIXTRACKLIB_COMMON_HEADERS
     ocl_environment.h
     tests/test_opencl_tools.h
)

set( SIXTRACKLIB_COMMON_SOURCES 
     tests/test_opencl_tools.c
     details/ocl_environment.c
)

add_library( sixtrack_opencl OBJECT 
             ${SIXTRACKLIB_COMMON_HEADERS}
             ${SIXTRACKLIB_COMMON_SOURCES} 
)
 
target_include_directories( sixtrack_opencl PUBLIC 
    ${CMAKE_SOURCE_DIR} 
    ${SIXTRACKL_OPENCL_INCLUDE_DIR} )
             
set_target_properties(  sixtrack_opencl PROPERTIES LINKER_LANGUAGE C )
set_target_properties(  sixtrack_opencl PROPERTIES POSITION_INDEPENDENT_CODE True )
set_target_properties(  sixtrack_opencl PROPERTIES C_STANDARD 99 )

target_compile_definitions( sixtrack_opencl PUBLIC 
        ${SIXTRACKLIB_NAMESPACE_FLAGS} -DCL_USE_DEPRECATED_OPENCL_1_2_APIS )

target_compile_options( sixtrack_opencl BEFORE PUBLIC 
        ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic -ansi )
                        
if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )                                                                                                                                                                             
    add_subdirectory( tests )                                                                                                                                                                                     
endif()                                                                                                                                                                                                           
                                                                                                                                                                                                                  
if( SIXTRACKL_ENABLE_TIMING_TESTS )                                                                                                                                                                               
    add_subdirectory( tests_benchmark )                                                                                                                                                                           
endif()     

# end: sixtracklib/opencl/CMakeLists.txt 
