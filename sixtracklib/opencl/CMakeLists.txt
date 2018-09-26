set( SIXTRACKLIB_OPENCL_HEADERS
     private/base_context.h
     argument.h
)

set( SIXTRACKLIB_OPENCL_KERNEL_HEADERS

)

set( SIXTRACKLIB_OPENCL_SOURCES
     code/base_context.cpp
     code/argument.cpp
)

add_library( sixtrack_opencl OBJECT
             ${SIXTRACKLIB_OPENCL_HEADERS}
             ${SIXTRACKLIB_OPENCL_SOURCES}
)

target_include_directories( sixtrack_opencl PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${SIXTRACKL_OPENCL_INCLUDE_DIR} )

set_target_properties( sixtrack_opencl PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack_opencl PROPERTIES POSITION_INDEPENDENT_CODE True )
set_target_properties( sixtrack_opencl PROPERTIES CXX_STANDARD 11 )
set_target_properties( sixtrack_opencl PROPERTIES CXX_STANDARD_REQUIRED ON   )

target_compile_definitions( sixtrack_opencl PUBLIC
        ${SIXTRACKLIB_NAMESPACE_FLAGS} -DCL_USE_DEPRECATED_OPENCL_1_2_APIS )

target_compile_options( sixtrack_opencl BEFORE PUBLIC
        ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic -ansi )


# ------------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_OPENCL_INSTALL_PATH include/sixtracklib/opencl )

install( FILES ${SIXTRACKLIB_OPENCL_HEADERS}
         DESTINATION ${SIXTRACKLIB_OPENCL_INSTALL_PATH} )

# end: sixtracklib/opencl/CMakeLists.txt
