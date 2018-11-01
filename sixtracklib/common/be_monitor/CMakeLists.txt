# sixtracklib/common/be_monitor/CMakeLists.txt

set( SIXTRACKLIB_COMMON_BE_MONITOR_C99_HEADERS
     be_monitor.h
     io_buffer.h
     track.h
)

set( SIXTRACKLIB_COMMON_BE_MONITOR_C99_SOURCES
     io_buffer.c
)

set( SIXTRACKLIB_COMMON_BE_MONITOR_CXX_HEADERS
)

# ------------------------------------------------------------------------------
# build sixtrack_common_be_monitor  submodule object library

add_library( sixtrack_common_be_monitor OBJECT
             ${SIXTRACKLIB_COMMON_BE_MONITOR_C99_HEADERS}
             ${SIXTRACKLIB_COMMON_BE_MONITOR_C99_SOURCES}
)

target_include_directories( sixtrack_common_be_monitor PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_be_monitor PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack_common_be_monitor PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties( sixtrack_common_be_monitor PROPERTIES C_STANDARD 99 )
set_target_properties( sixtrack_common_be_monitor PROPERTIES C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_be_monitor PRIVATE
                        -Wall -Werror -pedantic -ansi ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_be_monitor as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES
    ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_be_monitor> CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------
# Pass in C++ headers to parent scope to track as a depenency for the whole lib

if( SIXTRACKLIB_COMMON_BE_MONITOR_CXX_HEADERS )
    string( REGEX REPLACE "([^;]+)" "be_monitor/\\1" LOCAL_CXX_HEADERS
            "${SIXTRACKLIB_COMMON_BE_MONITOR_CXX_HEADERS}" )

    set(  SIXTRACKLIB_COMMON_CXX_HEADERS
        ${SIXTRACKLIB_COMMON_CXX_HEADERS}
        ${LOCAL_CXX_HEADERS} PARENT_SCOPE )
endif()

# ==============================================================================
# install :

if( SIXTRACKLIB_COMMON_BE_MONITOR_C99_HEADERS )

    set( SIXTRACKLIB_COMMON_BE_MONITOR_C99_INSTALL_PATH
         include/sixtracklib/common/be_monitor )

    install( FILES ${SIXTRACKLIB_COMMON_BE_MONITOR_C99_HEADERS}
             DESTINATION ${SIXTRACKLIB_COMMON_BE_MONITOR_C99_INSTALL_PATH} )
endif()


if( SIXTRACKLIB_COMMON_BE_MONITOR_CXX_HEADERS )

    set( SIXTRACKLIB_COMMON_BE_MONITOR_CXX_INSTALL_PATH
         include/sixtracklib/common/be_monitor )

    install( FILES ${SIXTRACKLIB_COMMON_BE_MONITOR_CXX_HEADERS}
             DESTINATION ${SIXTRACKLIB_COMMON_BE_MONITOR_CXX_INSTALL_PATH} )
endif()

# end: sixtracklib/common/be_monitor/CMakeLists.txt
