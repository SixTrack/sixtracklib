# ==============================================================================
# sixtracklib/cuda/control submodule

set( SIXTRACKLIB_CUDA_CONTROL_CXX_HEADERS
     kernel_config.hpp
     node_info.hpp
     argument_base.hpp
)

set( SIXTRACKLIB_CUDA_CONTROL_CXX_SOURCES
     kernel_config.cpp
     kernel_config_c99.cpp
     node_info.cpp
     argument_base.cpp
     argument_base_c99.cpp
     argument.cpp
     argument_c99.cpp
     controller.cpp
     controller_c99.cpp
)

set( SIXTRACKLIB_CUDA_CONTROL_C99_HEADERS
     argument_base.h
     kernel_config.h
)

add_library( sixtrack_cuda_cxx_control OBJECT
             ${SIXTRACKLIB_CUDA_CONTROL_CXX_HEADERS}
             ${SIXTRACKLIB_CUDA_CONTROL_C99_HEADERS}
             ${SIXTRACKLIB_CUDA_CONTROL_CXX_SOURCES}
)

target_compile_definitions( sixtrack_cuda_cxx_control
    PUBLIC ${SIXTRACKLIB_NAMESPACE_FLAGS} )

target_include_directories( sixtrack_cuda_cxx_control
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> )

set_target_properties( sixtrack_cuda_cxx_control PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    LINKER_LANGUAGE C
    CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON )

# ------------------------------------------------------------------------------
# pass on sixtrack_cuda_cxx_control as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES
    ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_cuda_cxx_control> CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------