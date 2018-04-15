# sixtracklib/sixtracklib/common/CMakeLists.txt

set( SIXTRACKLIB_COMMON_HEADERS 
     details/tools.h
     impl/block_type.h
     impl/block_drift_type.h
     impl/particles_type.h
     impl/track_single.h     
     block.h
     block_drift.h
     mem_pool.h
     particles.h
     restrict.h
     single_particle.h
     track.h
     values.h
)


set( SIXTRACKLIB_COMMON_SOURCES
     details/block.c
     details/block_drift.c
     details/mem_pool.c
     details/particles.c
     details/single_particle.c
     details/track_single.c
)

add_library( sixtrack_common OBJECT 
             ${SIXTRACKLIB_COMMON_HEADERS}
             ${SIXTRACKLIB_COMMON_SOURCES} )
             
target_include_directories( sixtrack_common PUBLIC ${CMAKE_SOURCE_DIR} )
             
set_target_properties(  sixtrack_common PROPERTIES LINKER_LANGUAGE C )
set_target_properties(  sixtrack_common PROPERTIES POSITION_INDEPENDENT_CODE True )
set_target_properties(  sixtrack_common PROPERTIES C_STANDARD 99 )
target_compile_options( sixtrack_common BEFORE PUBLIC -Wall -Werror -pedantic -ansi )

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )
    add_subdirectory( tests )
endif()

#end sixtracklib/sixtracklib/common/CMakeLists.txt
