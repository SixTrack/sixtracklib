# =============================================================================
# external/CMakeLists.txt

message( STATUS "---- processing external/CMakeLists.txt" )

if( SIXTRACKL_ENABLE_OPENCL )
    add_subdirectory( CL )
endif()

