add_subdirectory( common )

if( SIXTRACKL_ENABLE_OPENCL )
    add_subdirectory( opencl )
endif()

if( SIXTRACKL_ENABLE_CUDA )
    add_subdirectory( cuda )
endif()
