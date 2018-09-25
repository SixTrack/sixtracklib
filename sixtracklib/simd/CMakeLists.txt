# set( SIXTRACKLIB_SIMD_HEADERS
#      track.h
# )
#
# set( SIXTRACKLIB_SIMD_SOURCES
#      details/track.c
# )
#
# add_library( sixtrack_simd OBJECT
#              ${SIXTRACKLIB_SIMD_HEADERS}
#              ${SIXTRACKLIB_SIMD_SOURCES} )
#
# target_include_directories( sixtrack_simd PUBLIC ${CMAKE_SOURCE_DIR} )
#
# set_target_properties(  sixtrack_simd PROPERTIES LINKER_LANGUAGE C )
# set_target_properties(  sixtrack_simd PROPERTIES POSITION_INDEPENDENT_CODE True )
# set_target_properties(  sixtrack_simd PROPERTIES C_STANDARD 99 )
# target_compile_options( sixtrack_simd BEFORE PUBLIC
#     ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic )
#
# if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )
#     add_subdirectory( tests )
# endif()
#
# if( SIXTRACKL_ENABLE_TIMING_TESTS )
#     add_subdirectory( tests_benchmark )
# endif()

#end: sixtracklib/simd/CMakeLists.txt
