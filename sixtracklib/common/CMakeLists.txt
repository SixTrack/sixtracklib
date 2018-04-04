# sixtracklib/sixtracklib/common/CMakeLists.txt

set( SIXTRACKLIB_COMMON_HEADERS 
     single_particle.h
     particles.h
     restrict.h
     values.h
)


set( SIXTRACKLIB_COMMON_SOURCES
     details/single_particle.c
     details/particles.c
)


add_library( sixtrack_common OBJECT 
             ${SIXTRACKLIB_COMMON_HEADERS}
             ${SIXTRACKLIB_COMMON_SOURCES} )
             
target_include_directories( sixtrack_common PUBLIC ${CMAKE_SOURCE_DIR} )
             
set_target_properties(  sixtrack_common PROPERTIES LINKER_LANGUAGE C )
set_target_properties(  sixtrack_common PROPERTIES POSITION_INDEPENDENT_CODE True )
set_target_properties(  sixtrack_common PROPERTIES C_STANDARD 99 )
target_compile_options( sixtrack_common BEFORE PUBLIC -Wall -Werror -pedantic -ansi )

#end sixtracklib/sixtracklib/common/CMakeLists.txt
