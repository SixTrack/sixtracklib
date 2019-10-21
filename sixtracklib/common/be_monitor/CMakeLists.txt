# sixtracklib/common/be_monitor/CMakeLists.txt

set( SIXTRACKLIB_COMMON_BE_MONITOR_C99_HEADERS
     be_monitor.h be_monitor_kernel_impl.h output_buffer.h track.h )

set( SIXTRACKLIB_COMMON_BE_MONITOR_C99_SOURCES
     be_monitor.c output_buffer.c )

set( SIXTRACKLIB_COMMON_BE_MONITOR_CXX_HEADERS be_monitor.hpp )

add_library( sixtrack_common_be_monitor OBJECT
    ${SIXTRACKLIB_COMMON_BE_MONITOR_C99_HEADERS}
    ${SIXTRACKLIB_COMMON_BE_MONITOR_C99_SOURCES}
    ${SIXTRACKLIB_COMMON_BE_MONITOR_CXX_HEADERS} )

target_include_directories( sixtrack_common_be_monitor PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_be_monitor PROPERTIES LINKER_LANGUAGE C
    POSITION_INDEPENDENT_CODE ON C_STANDARD 99 C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_be_monitor PRIVATE
    ${SIXTRACKLIB_C99_FLAGS} ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_be_beambeam as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_be_monitor> CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------
# install:

if( SIXTRACKLIB_COMMON_BE_MONITOR_C99_HEADERS )
    install( FILES ${SIXTRACKLIB_COMMON_BE_MONITOR_C99_HEADERS}
             DESTINATION "${SIXTRL_COMMON_C99_INSTALL_PREFIX}/be_monitor" )
endif()

if( SIXTRACKL_ENABLE_CXX AND SIXTRACKLIB_COMMON_BE_MONITOR_CXX_HEADERS )
    install( FILES ${SIXTRACKLIB_COMMON_BE_MONITOR_CXX_HEADERS}
             DESTINATION "${SIXTRL_COMMON_CXX_INSTALL_PREFIX}/be_monitor" )
endif()

#end: sixtracklib/common/be_monitor/CMakeLists.txt
