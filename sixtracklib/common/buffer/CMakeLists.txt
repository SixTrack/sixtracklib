set( SIXTRACKLIB_MODULES ${SIXTRACKLIB_MODULES} PARENT_SCOPE )

set( SIXTRL_COMMON_BUFFER_HEADERS
     alignment.h
     buffer_garbage.h
     buffer_generic.h
     buffer_object.h
     buffer_type.h
     managed_buffer_minimal.h
     managed_buffer_remap.h
     managed_buffer.h
     mem_pool.h
)

set( SIXTRL_COMMON_BUFFER_SOURCES
     alignment.c
     buffer.c
     mem_pool.c
)

add_library( sixtrack_common_buffer OBJECT
    ${SIXTRL_COMMON_BUFFER_HEADERS}
    ${SIXTRL_COMMON_BUFFER_SOURCES}
)

set(  SIXTRACKLIB_MODULES
    ${SIXTRACKLIB_MODULES} sixtrack_common_buffer PARENT_SCOPE )

target_include_directories( sixtrack_common_buffer
                            PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> )

# -----------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_COMMON_BUFFER_INSTALL_PATH
     include/sixtracklib/common/buffer )

install( FILES ${SIXTRL_COMMON_BUFFER_HEADERS}
         DESTINATION ${SIXTRACKLIB_COMMON_BE_BEAMBEAM_INSTALL_PATH} )

#end: sixtracklib/common/buffer/CMakeLists.txt

